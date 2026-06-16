#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train detector prompt-completion SFT with Transformers.

Examples::

  export ASCEND_RT_VISIBLE_DEVICES=8,9
  torchrun --standalone --nproc_per_node=2 --master_port=47086 \
    -m AnomaMind.train.sft \
    --train_csv /path/AnomaMind/train/sft/abl_know/cross/final_merge.csv \
    --output_dir /path/sft/abl_0.6B_know \
    --bf16 \
    --gradient_checkpointing \
    --fsdp

  python3 -m AnomaMind.train.sft \
    --train_csv /path/data/WSD/result/train/wsd_detector_sft.csv \
    --output_dir /path/sft/wsd_debug_1.7B \
    --max_train_samples 32 \
    --eval_split_ratio 0.1
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from datasets import load_dataset


SYSTEM_PROMPT = "You are a professional time series anomaly detection expert."


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detector prompt-completion SFT (transformers).")
    p.add_argument("--model_path", required=True, help="Local HF snapshot path for base model.")
    p.add_argument(
        "--train_csv",
        required=True,
        help="CSV file containing prompt/answer columns, typically from AnomaMind.train.prepare_sft_data.",
    )
    p.add_argument("--prompt_col", default="prompt", help="Column name for prompt text.")
    p.add_argument("--answer_col", default="answer", help="Column name for answer text.")
    p.add_argument("--output_dir", required=True, help="Output HF directory.")

    p.add_argument("--max_length", type=int, default=9000, help="Max sequence length (tokens).")
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--save_total_limit", type=int, default=2)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataloader_num_workers", type=int, default=8)
    p.add_argument("--bf16", action="store_true", help="Use bf16 if supported.")
    p.add_argument("--fp16", action="store_true", help="Use fp16 if supported.")
    p.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    p.add_argument("--fsdp", action="store_true", help="Enable Hugging Face Trainer FSDP full_shard auto_wrap.")
    p.add_argument(
        "--fsdp_transformer_layer_cls_to_wrap",
        default="Qwen3DecoderLayer",
        help="Transformer layer class name for FSDP auto wrapping.",
    )
    p.add_argument(
        "--ddp_backend",
        default="",
        help="DDP backend override. For Ascend NPU typically use 'hccl'. Leave empty to auto-select.",
    )

    p.add_argument("--eval_split_ratio", type=float, default=0.0, help="If >0, split train into train/eval.")
    p.add_argument("--max_train_samples", type=int, default=0, help="0=all; else cap for quick runs.")
    return p.parse_args(argv)


def _iter_csv_records_robust(path: str, *, prompt_col: str, answer_col: str) -> Iterator[Tuple[str, str]]:
    import csv

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        header = f.readline()
        if not header:
            return
        header_cols = next(csv.reader([header]))
        if not header_cols:
            raise ValueError("CSV header is empty")
        try:
            pidx = header_cols.index(str(prompt_col))
        except ValueError as exc:
            raise ValueError(f"prompt_col not found in CSV header: {prompt_col}; header={header_cols}") from exc
        try:
            aidx = header_cols.index(str(answer_col))
        except ValueError as exc:
            raise ValueError(f"answer_col not found in CSV header: {answer_col}; header={header_cols}") from exc

        buf_lines: List[str] = []
        quote_count = 0

        def _flush_if_complete() -> Optional[Tuple[str, str]]:
            nonlocal buf_lines, quote_count
            if not buf_lines:
                return None
            chunk = "".join(buf_lines)
            try:
                row = next(csv.reader([chunk]))
            except Exception:
                return None
            if len(row) <= max(pidx, aidx):
                return None
            prompt = row[pidx]
            answer = row[aidx]
            buf_lines = []
            quote_count = 0
            return prompt, answer

        for line in f:
            buf_lines.append(line)
            quote_count += line.count('"')
            if quote_count % 2 == 0:
                rec = _flush_if_complete()
                if rec is not None:
                    yield rec
                else:
                    buf_lines = []
                    quote_count = 0


def _convert_csv_to_jsonl(
    *,
    csv_path: str,
    jsonl_path: str,
    prompt_col: str,
    answer_col: str,
    max_samples: int = 0,
) -> int:
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    n = 0
    with open(jsonl_path, "w", encoding="utf-8") as out:
        for prompt, answer in _iter_csv_records_robust(csv_path, prompt_col=prompt_col, answer_col=answer_col):
            out.write(json.dumps({"prompt": prompt, "answer": answer}, ensure_ascii=False) + "\n")
            n += 1
            if max_samples and n >= int(max_samples):
                break
    return n


def _apply_chat_template_or_fallback(
    tokenizer,
    messages: List[Dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    parts: List[str] = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role in {"system", "user", "assistant"}:
            parts.append(content.strip())
    if add_generation_prompt:
        return "\n\n".join(parts[:-1]).strip() + "\n\n"
    return "\n\n".join(parts).strip()


def _tokenize_with_labels(tokenizer, prompt: str, answer: str, max_length: int) -> Dict[str, List[int]]:
    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    prompt_text = _apply_chat_template_or_fallback(tokenizer, base_messages, add_generation_prompt=True)
    full_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    full_text = _apply_chat_template_or_fallback(tokenizer, full_messages, add_generation_prompt=False)

    if tokenizer.eos_token and not full_text.endswith(tokenizer.eos_token):
        full_text = full_text + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)
    input_ids = full["input_ids"]
    attn = full["attention_mask"]

    pl = min(len(prompt_ids), len(input_ids))
    if input_ids[:pl] != prompt_ids[:pl]:
        mx = 0
        for k in range(pl, 0, -1):
            if input_ids[:k] == prompt_ids[:k]:
                mx = k
                break
        pl = mx

    labels = [-100] * pl + input_ids[pl:]
    labels = labels[: len(input_ids)]
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: object

    def __call__(self, features: List[Dict[str, List[int]]]):
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        def _pad(seq: List[int], pad_value: int) -> List[int]:
            return seq + [pad_value] * (max_len - len(seq))

        input_ids = torch.tensor([_pad(f["input_ids"], pad_id) for f in features], dtype=torch.long)
        attention_mask = torch.tensor([_pad(f["attention_mask"], 0) for f in features], dtype=torch.long)
        labels = torch.tensor([_pad(f["labels"], -100) for f in features], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _default_ddp_backend(torch) -> str:
    if hasattr(torch, "npu") and getattr(torch.npu, "is_available", lambda: False)():
        return "hccl"
    if torch.cuda.is_available():
        return "nccl"
    return "gloo"


def _build_training_args_kwargs(
    *,
    args: argparse.Namespace,
    ddp_backend: str,
    has_eval: bool,
    training_arguments_cls,
) -> Dict[str, object]:
    ta_kwargs: Dict[str, object] = dict(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=float(args.num_train_epochs),
        learning_rate=float(args.learning_rate),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        warmup_ratio=float(args.warmup_ratio),
        weight_decay=float(args.weight_decay),
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        save_total_limit=int(args.save_total_limit),
        seed=int(args.seed),
        dataloader_num_workers=int(args.dataloader_num_workers),
        bf16=bool(args.bf16),
        fp16=bool(args.fp16),
        ddp_backend=ddp_backend,
        lr_scheduler_type="constant",
        evaluation_strategy=("steps" if has_eval else "no"),
        eval_steps=(int(args.save_steps) if has_eval else None),
        report_to=[],
    )
    if getattr(args, "fsdp", False):
        ta_kwargs["fsdp"] = "full_shard auto_wrap"
        ta_kwargs["fsdp_config"] = {"transformer_layer_cls_to_wrap": str(args.fsdp_transformer_layer_cls_to_wrap)}
    sig = inspect.signature(training_arguments_cls.__init__)
    allowed = set(sig.parameters.keys())
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in allowed}
    if "evaluation_strategy" not in allowed and "do_eval" in allowed:
        ta_kwargs["do_eval"] = bool(has_eval)
    return ta_kwargs


def _ensure_process_group_before_training_args(torch, ddp_backend: str) -> None:
    import torch.distributed as dist

    if not dist.is_available() or dist.is_initialized():
        return

    try:
        ws = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        ws = 1

    if ws > 1:
        dist.init_process_group(backend=ddp_backend)
        return

    for k in ("LOCAL_RANK", "RANK", "GROUP_RANK", "ROLE_RANK", "WORLD_SIZE", "GROUP_WORLD_SIZE", "ROLE_WORLD_SIZE"):
        os.environ.pop(k, None)

    try:
        store = dist.HashStore()  # type: ignore[attr-defined]
        dist.init_process_group(backend="gloo", rank=0, world_size=1, store=store)
    except AttributeError:
        import tempfile

        fd, fpath = tempfile.mkstemp(prefix="pt_pg_", suffix=".init")
        os.close(fd)
        try:
            dist.init_process_group(backend="gloo", init_method="file://" + fpath, rank=0, world_size=1)
        finally:
            try:
                os.remove(fpath)
            except OSError:
                pass


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    import torch

    try:  # pragma: no cover
        import torch_npu  # type: ignore  # noqa: F401
    except Exception:
        torch_npu = None  # type: ignore[assignment]

    ddp_backend = str(args.ddp_backend).strip().lower() if args.ddp_backend else ""
    if not ddp_backend:
        ddp_backend = _default_ddp_backend(torch)
    _ensure_process_group_before_training_args(torch, ddp_backend)

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if hasattr(torch, "npu") and getattr(torch.npu, "is_available", lambda: False)():
        try:
            torch.npu.set_device(local_rank)
        except Exception:
            pass

    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tmp_jsonl = os.path.join(args.output_dir, "_train_data.jsonl")
    written = _convert_csv_to_jsonl(
        csv_path=str(args.train_csv),
        jsonl_path=tmp_jsonl,
        prompt_col=str(args.prompt_col),
        answer_col=str(args.answer_col),
        max_samples=int(args.max_train_samples) if args.max_train_samples else 0,
    )
    if written <= 0:
        raise SystemExit(f"No training rows parsed from CSV: {args.train_csv}")
    ds = load_dataset("json", data_files={"train": tmp_jsonl})["train"]

    if float(args.eval_split_ratio) > 0:
        split = ds.train_test_split(test_size=float(args.eval_split_ratio), seed=int(args.seed))
        train_ds = split["train"]
        eval_ds = split["test"]
    else:
        train_ds = ds
        eval_ds = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    max_length = int(args.max_length)

    def _map_fn(ex: Dict[str, str]) -> Dict[str, List[int]]:
        return _tokenize_with_labels(tokenizer, ex.get("prompt", ""), ex.get("answer", ""), max_length=max_length)

    train_tok = train_ds.map(_map_fn, remove_columns=train_ds.column_names, num_proc=None)
    eval_tok = eval_ds.map(_map_fn, remove_columns=eval_ds.column_names, num_proc=None) if eval_ds is not None else None

    ta_kwargs = _build_training_args_kwargs(
        args=args,
        ddp_backend=ddp_backend,
        has_eval=bool(eval_tok is not None),
        training_arguments_cls=TrainingArguments,
    )
    targs = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=DataCollatorForCausalLMWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
