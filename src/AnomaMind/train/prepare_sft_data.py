#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build detector SFT CSV from inference result JSON files.

Examples::

  python3 -m AnomaMind.train.prepare_sft_data \
      --record_dir /path/data/KPI/result/train/8B-in \
      --output_csv /path/data/KPI/result/train/kpi_detector_sft.csv \
      --max_prompt_tokens 7000

  python3 -m AnomaMind.train.prepare_sft_data \
      --record_dir /path/data/WSD/result/train \
      --output_csv /path/data/WSD/result/train/wsd_detector_sft.csv \
      --max_samples 500

  python3 -m AnomaMind.train.prepare_sft_data \
      --mode multi \
      --res_root /path/data \
      --datasets KPI WSD IOPS YAHOO \
      --output_csv /path/data/sft/mixed_detector_sft.csv \
      --balance downsample \
      --shuffle
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from AnomaMind.core.Detector import Detector
from AnomaMind.core.knowledge_base import KnowledgeBase
from AnomaMind.helpers.message_io import import_messages
from AnomaMind.helpers.prompt_token_filter import build_text_token_counter, keep_text_prompt
from AnomaMind.workflows.detector_agent import restore_detector_training_sample


@dataclass(frozen=True)
class DatasetStat:
    name: str
    record_dir: Path
    valid_records: int


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"record is not a JSON object: {path}")
    return obj


def _records_to_dict_list(records: Any) -> List[Dict[str, Any]]:
    if not isinstance(records, list):
        return []
    return [dict(one) for one in records if isinstance(one, dict)]


def _extract_actor_records(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    actor_records = _records_to_dict_list(record.get("actor_messages"))
    if actor_records:
        return actor_records

    messages = _records_to_dict_list(record.get("messages"))
    if not messages:
        return []

    # New aggregated format: actor_messages + [detector user prompt, detector assistant answer]
    if len(messages) >= 2:
        tail_user = str(messages[-2].get("role", "") or "").lower() == "user"
        tail_assistant = str(messages[-1].get("role", "") or "").lower() == "assistant"
        if tail_user and tail_assistant:
            return messages[:-2]

    # Older aggregated format may only append the final detector answer.
    if messages and str(messages[-1].get("role", "") or "").lower() == "assistant":
        return messages[:-1]
    return messages


def _extract_detector_answer(record: Dict[str, Any]) -> str:
    detector_records = _records_to_dict_list(record.get("detector_messages"))
    for message in reversed(detector_records):
        if str(message.get("role", "") or "").lower() == "assistant":
            content = message.get("content", "")
            return content if isinstance(content, str) else str(content or "")

    messages = _records_to_dict_list(record.get("messages"))
    for message in reversed(messages):
        role = str(message.get("role", "") or "").lower()
        if role != "assistant":
            continue
        if isinstance(message.get("tool_calls"), list) and message.get("tool_calls"):
            continue
        content = message.get("content", "")
        return content if isinstance(content, str) else str(content or "")
    return ""


def _fill_missing_confidence_in_answer(answer: str) -> str:
    """Old infer answers may omit confidence; SFT label should include it (default 2)."""
    text = str(answer or "")
    if not text.strip():
        return text

    reasoning_match = re.search(
        r"<(think|redacted_reasoning|redacted_thinking)>(.*?)</\1>",
        text,
        re.DOTALL,
    )
    search_from = reasoning_match.end() if reasoning_match else 0
    tail = text[search_from:]
    decoder = json.JSONDecoder()
    json_start = json_end = None
    anomalies_list = None

    for source, base in ((tail, search_from), (text, 0)):
        pos = source.find("[")
        while pos != -1:
            try:
                parsed, end = decoder.raw_decode(source[pos:])
            except json.JSONDecodeError:
                pos = source.find("[", pos + 1)
                continue
            if isinstance(parsed, list):
                json_start = base + pos
                json_end = base + pos + end
                anomalies_list = parsed
                break
            pos = source.find("[", pos + 1)
        if anomalies_list is not None:
            break

    if not isinstance(anomalies_list, list) or json_start is None or json_end is None:
        return text
    if not anomalies_list:
        return text

    changed = False
    for item in anomalies_list:
        if not isinstance(item, dict):
            continue
        if item.get("confidence") is None:
            item["confidence"] = 2
            changed = True

    if not changed:
        return text

    new_json = json.dumps(anomalies_list, ensure_ascii=False, separators=(",", ":"))
    return text[:json_start] + new_json + text[json_end:]


def _rebuild_detector_prompt(record_path: str | Path) -> str:
    restored = restore_detector_training_sample(record_path)
    record = restored["record"]
    state = dict(restored["initial_state"])
    actor_records = _extract_actor_records(record)
    state["actor_messages"] = import_messages(actor_records)

    detector = Detector(knowledge_base=KnowledgeBase(), llm=None)
    rebuilt_messages = detector._build_messages(state)  # type: ignore[attr-defined]
    if not rebuilt_messages:
        return ""
    prompt_message = rebuilt_messages[0]
    prompt_content = getattr(prompt_message, "content", "") or ""
    return prompt_content if isinstance(prompt_content, str) else str(prompt_content)


def build_detector_sft_sample(record_path: str | Path) -> Dict[str, Any]:
    restored = restore_detector_training_sample(record_path)
    record = restored["record"]
    prompt = _rebuild_detector_prompt(record_path)
    answer = _fill_missing_confidence_in_answer(_extract_detector_answer(record))
    if not prompt.strip():
        raise ValueError(f"empty detector prompt: {record_path}")
    if not answer.strip():
        raise ValueError(f"empty detector answer: {record_path}")

    return {
        "record_path": str(Path(record_path).resolve()),
        "segment_dir": str(restored["segment_dir"]),
        "type": str(restored["initial_state"]["type"]),
        "prompt": prompt,
        "answer": answer,
        "ground_truth_intervals": json.dumps(
            record.get("ground_truth_intervals", restored.get("ground_truth", [])),
            ensure_ascii=False,
        ),
    }


def list_detector_sft_records(
    record_dir: str | Path,
    *,
    max_prompt_tokens: int = 5000,
    tokenizer_path: str,
    prompt_token_counter: Optional[Callable[[str], int]] = None,
) -> List[str]:
    record_dir = Path(record_dir).resolve()
    if not record_dir.exists():
        raise FileNotFoundError(f"record_dir does not exist: {record_dir}")

    counter = prompt_token_counter
    if int(max_prompt_tokens) > 0 and counter is None:
        counter = build_text_token_counter(str(tokenizer_path))

    out: List[str] = []
    for path in sorted(record_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix != ".json" or path.name.startswith("_"):
            continue
        try:
            sample = build_detector_sft_sample(path)
        except Exception:
            continue
        if (
            sample.get("prompt")
            and sample.get("answer")
            and (
                counter is None
                or keep_text_prompt(
                    str(sample.get("prompt") or ""),
                    max_prompt_tokens=int(max_prompt_tokens),
                    token_counter=counter,
                )
            )
        ):
            out.append(str(path.resolve()))
    return out


def export_detector_sft_csv(
    *,
    record_dir: str | Path,
    output_csv: str | Path,
    max_samples: int = 0,
    max_prompt_tokens: int = 5000,
    tokenizer_path: str,
    prompt_token_counter: Optional[Callable[[str], int]] = None,
) -> int:
    records = list_detector_sft_records(
        record_dir,
        max_prompt_tokens=int(max_prompt_tokens),
        tokenizer_path=str(tokenizer_path),
        prompt_token_counter=prompt_token_counter,
    )
    if max_samples > 0:
        records = records[: int(max_samples)]

    output_csv = Path(output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["prompt", "answer", "record_path", "segment_dir", "type", "ground_truth_intervals"]
    written = 0
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record_path in records:
            sample = build_detector_sft_sample(record_path)
            writer.writerow(sample)
            written += 1
    return written


def _discover_datasets(res_root: Path, explicit: Optional[Sequence[str]]) -> List[Path]:
    if explicit:
        out: List[Path] = []
        for name in explicit:
            p = (res_root / name).resolve()
            if not p.is_dir():
                raise FileNotFoundError(f"dataset dir not found: {p}")
            out.append(p)
        return out
    return sorted([p for p in res_root.iterdir() if p.is_dir()])


def _sample_paths(
    paths: List[str],
    *,
    target: int,
    rng: random.Random,
    replace: bool,
) -> List[str]:
    if target <= 0 or not paths:
        return []
    if not replace:
        if target >= len(paths):
            return list(paths)
        return rng.sample(paths, target)
    return [rng.choice(paths) for _ in range(target)]


def _choose_target(
    *,
    counts: Dict[str, int],
    balance: str,
    target_per_dataset: int,
) -> int:
    if not counts or balance == "none":
        return 0
    if target_per_dataset > 0:
        return int(target_per_dataset)
    if balance == "downsample":
        return min(counts.values())
    if balance == "upsample":
        return max(counts.values())
    raise ValueError(f"unknown balance mode: {balance!r}")


def _print_stats(stats: List[DatasetStat]) -> None:
    total = sum(s.valid_records for s in stats) or 1
    print("[dataset_stats]")
    for s in stats:
        ratio = 100.0 * float(s.valid_records) / float(total)
        print(f"  {s.name:<10} valid_records={s.valid_records:<6} ratio={ratio:>6.2f}% dir={s.record_dir}")
    print(f"  {'TOTAL':<10} valid_records={sum(s.valid_records for s in stats)}")


def export_detector_sft_csv_multi(
    *,
    res_root: str | Path,
    datasets: Optional[Sequence[str]],
    output_csv: str | Path,
    balance: str,
    target_per_dataset: int,
    seed: int,
    shuffle: bool,
    dry_run: bool,
    max_prompt_tokens: int,
    tokenizer_path: str,
) -> Tuple[int, List[DatasetStat], Dict[str, int]]:
    res_root = Path(res_root).resolve()
    output_csv = Path(output_csv).resolve()

    ds_dirs = _discover_datasets(res_root, datasets)
    stats: List[DatasetStat] = []
    records_by_ds: Dict[str, List[str]] = {}

    for d in ds_dirs:
        name = d.name
        recs = list_detector_sft_records(
            d,
            max_prompt_tokens=int(max_prompt_tokens),
            tokenizer_path=str(tokenizer_path),
        )
        records_by_ds[name] = recs
        stats.append(DatasetStat(name=name, record_dir=d, valid_records=len(recs)))

    _print_stats(stats)

    counts = {k: len(v) for k, v in records_by_ds.items()}
    target = _choose_target(counts=counts, balance=balance, target_per_dataset=target_per_dataset)
    rng = random.Random(int(seed))

    picked: List[Tuple[str, str]] = []
    for ds_name, recs in records_by_ds.items():
        if balance == "none":
            chosen = list(recs)
        elif balance == "downsample":
            chosen = _sample_paths(recs, target=target, rng=rng, replace=False)
        elif balance == "upsample":
            chosen = _sample_paths(recs, target=target, rng=rng, replace=len(recs) < target)
        else:
            raise ValueError(f"unknown balance={balance!r}")
        picked.extend((ds_name, p) for p in chosen)

    if shuffle:
        rng.shuffle(picked)

    picked_counts: Dict[str, int] = {}
    for ds_name, _ in picked:
        picked_counts[ds_name] = picked_counts.get(ds_name, 0) + 1

    print("[sampling]")
    print(f"  balance={balance} target_per_dataset={target_per_dataset} seed={seed} shuffle={shuffle}")
    if balance != "none":
        print(f"  chosen_target_per_dataset={target}")
    for ds_name in sorted(picked_counts.keys()):
        print(f"  picked[{ds_name}]={picked_counts[ds_name]}")
    print(f"  picked_total={len(picked)}")

    if dry_run:
        print(f"[dry_run] not writing CSV: {output_csv}")
        return 0, stats, picked_counts

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "prompt",
        "answer",
        "record_path",
        "segment_dir",
        "type",
        "ground_truth_intervals",
    ]
    written = 0
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ds_name, record_path in picked:
            sample = build_detector_sft_sample(record_path)
            sample["dataset"] = ds_name
            writer.writerow({k: sample.get(k, "") for k in fieldnames})
            written += 1

    print(f"[build_sft_data_multi] wrote={written} output={output_csv}")
    return written, stats, picked_counts


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build detector SFT CSV from detector result JSON records.")
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
        help="single=one result dir to one CSV; multi=merge multiple dataset result dirs into one CSV.",
    )
    parser.add_argument(
        "--record_dir",
        "--input-dir",
        "-r",
        dest="record_dir",
        default=None,
        help="Input result directory for single mode.",
    )
    parser.add_argument(
        "--res_root",
        default=None,
        help="Root directory that contains multiple dataset result dirs for multi mode.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset subdir names under --res_root for multi mode.",
    )
    parser.add_argument(
        "--output_csv",
        "--output-file",
        "-o",
        dest="output_csv",
        required=True,
        help="Output CSV path with prompt/answer columns",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="0=all")
    parser.add_argument(
        "--balance",
        choices=["none", "downsample", "upsample"],
        default="none",
        help="Sampling strategy for multi mode.",
    )
    parser.add_argument(
        "--target_per_dataset",
        type=int,
        default=0,
        help="Explicit per-dataset sample target for multi mode. 0 means infer from --balance.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for multi mode.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle merged rows in multi mode.")
    parser.add_argument("--dry_run", action="store_true", help="Only print multi-mode stats without writing output.")
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=7000,
        help="Drop samples whose prompt token count exceeds this limit. 0 disables filtering.",
    )
    parser.add_argument(
        "--tokenizer_path",
        required=True,
        help="Tokenizer path used for prompt token counting.",
    )
    args = parser.parse_args(argv)
    if args.mode == "single" and not args.record_dir:
        parser.error("--record_dir is required in single mode")
    if args.mode == "multi" and not args.res_root:
        parser.error("--res_root is required in multi mode")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "multi":
        written, _stats, _picked = export_detector_sft_csv_multi(
            res_root=args.res_root,
            datasets=args.datasets,
            output_csv=args.output_csv,
            balance=str(args.balance),
            target_per_dataset=int(args.target_per_dataset),
            seed=int(args.seed),
            shuffle=bool(args.shuffle),
            dry_run=bool(args.dry_run),
            max_prompt_tokens=int(args.max_prompt_tokens),
            tokenizer_path=str(args.tokenizer_path),
        )
        if not args.dry_run:
            print(f"[build_sft_data_multi] wrote={written} output={Path(args.output_csv).resolve()}")
        return

    written = export_detector_sft_csv(
        record_dir=args.record_dir,
        output_csv=args.output_csv,
        max_samples=int(args.max_samples),
        max_prompt_tokens=int(args.max_prompt_tokens),
        tokenizer_path=str(args.tokenizer_path),
    )
    print(f"[build_sft_data] wrote={written} output={Path(args.output_csv).resolve()}")


if __name__ == "__main__":
    main()
