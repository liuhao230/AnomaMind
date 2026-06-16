#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert SFT CSV (prompt/answer + metadata) into VERL parquet files.

This is for RL training with VERL where each example contains:
- prompt: chat messages (system + user)
- reward_model.ground_truth: JSON string of ground-truth intervals

It is designed to handle CSV rows with very long multiline fields.

Examples::

  python3 -m AnomaMind.train.prepare_rl_data \
      --mode csv \
      --csv /path/data/KPI/result/train/kpi_detector_sft.csv \
      --out-dir /path/verl/data/detector_rl_kpi \
      --data-source anoma/kpi_detector_sft \
      --ability ts_anomaly_detection \
      --val-ratio 0.05

  python3 -m AnomaMind.train.prepare_rl_data \
      --mode merge \
      --source kpi=/path/verl/data/detector_rl_kpi \
      --source wsd=/path/verl/data/detector_rl_wsd \
      --ratio kpi=1 \
      --ratio wsd=1 \
      --out-dir /path/verl/data/cross/kpi_wsd_equal
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

import datasets
import pyarrow.parquet as pq

from AnomaMind.helpers.prompt_token_filter import build_text_token_counter, keep_text_prompt

SYSTEM_PROMPT = "You are a professional time series anomaly detection expert."


def parse_kv_paths(items: Iterable[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for raw in items:
        if "=" not in str(raw):
            raise ValueError(f"invalid source mapping (expected name=/path): {raw}")
        name, path = str(raw).split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"invalid source mapping: {raw}")
        out[name] = Path(path).resolve()
    return out


def parse_kv_ints(items: Iterable[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for raw in items:
        if "=" not in str(raw):
            raise ValueError(f"invalid numeric mapping (expected name=int): {raw}")
        name, value = str(raw).split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            raise ValueError(f"invalid numeric mapping: {raw}")
        iv = int(value)
        if iv <= 0:
            raise ValueError(f"value must be > 0: {raw}")
        out[name] = iv
    return out


def _iter_csv_rows_robust(path: Path) -> Iterator[Dict[str, str]]:
    """
    Robust CSV reader that can handle huge multiline quoted fields.
    Returns each row as a dict(header -> value).
    """
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        header_line = f.readline()
        if not header_line:
            return
        header = next(csv.reader([header_line]))
        if not header:
            return

        buf_lines: List[str] = []
        quote_count = 0

        def _try_parse(chunk: str) -> Optional[List[str]]:
            try:
                return next(csv.reader([chunk]))
            except Exception:
                return None

        for line in f:
            buf_lines.append(line)
            quote_count += line.count('"')
            if quote_count % 2 != 0:
                continue
            chunk = "".join(buf_lines)
            row = _try_parse(chunk)
            buf_lines = []
            quote_count = 0
            if not row:
                continue
            # Some broken lines may have wrong column count; skip them.
            if len(row) < len(header):
                continue
            yield {header[i]: row[i] for i in range(len(header))}


def _coerce_json_intervals(text: str) -> str:
    """
    Ensure ground_truth_intervals is a JSON string.
    If text is invalid, fall back to '[]'.
    """
    t = (text or "").strip()
    if not t:
        return "[]"
    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass
    return "[]"


def build_examples(
    *,
    csv_path: Path,
    data_source: str,
    ability: str,
    max_samples: int,
    max_prompt_tokens: int = 5000,
    tokenizer_path: str,
    prompt_token_counter: Optional[Callable[[str], int]] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    counter = prompt_token_counter
    if int(max_prompt_tokens) > 0 and counter is None:
        counter = build_text_token_counter(str(tokenizer_path))

    for i, row in enumerate(_iter_csv_rows_robust(csv_path)):
        if max_samples and len(out) >= int(max_samples):
            break

        prompt = row.get("prompt", "") or ""
        if counter is not None and not keep_text_prompt(
            prompt,
            max_prompt_tokens=int(max_prompt_tokens),
            token_counter=counter,
        ):
            continue
        gt = _coerce_json_intervals(row.get("ground_truth_intervals", "") or "")

        ex: Dict[str, Any] = {
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "ability": ability,
            "reward_model": {
                "style": "rule",
                "ground_truth": gt,
            },
            "extra_info": {
                "split": "train",
                "index": int(i),
                "record_path": row.get("record_path", "") or "",
                "segment_dir": row.get("segment_dir", "") or "",
                "dataset_type": row.get("type", "") or "",
            },
        }
        out.append(ex)
    return out


def build_parquet_datasets(
    *,
    csv_path: Path,
    data_source: str,
    ability: str,
    max_samples: int,
    max_prompt_tokens: int = 5000,
    tokenizer_path: str,
    prompt_token_counter: Optional[Callable[[str], int]] = None,
    val_ratio: float = 0.0,
    seed: int = 42,
    shuffle_before_write: bool = True,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    examples = build_examples(
        csv_path=csv_path,
        data_source=str(data_source),
        ability=str(ability),
        max_samples=int(max_samples),
        max_prompt_tokens=int(max_prompt_tokens),
        tokenizer_path=str(tokenizer_path),
        prompt_token_counter=prompt_token_counter,
    )
    if not examples:
        raise ValueError(f"no examples generated from CSV: {csv_path}")

    ds = datasets.Dataset.from_list(examples)
    if bool(shuffle_before_write):
        ds = ds.shuffle(seed=int(seed))

    if float(val_ratio) > 0:
        split = ds.train_test_split(test_size=float(val_ratio), seed=int(seed))
        return split["train"], split["test"]
    return ds, ds


def write_parquet_datasets(
    *,
    csv_path: Path,
    out_dir: Path,
    data_source: str,
    ability: str,
    max_samples: int,
    max_prompt_tokens: int = 5000,
    tokenizer_path: str,
    prompt_token_counter: Optional[Callable[[str], int]] = None,
    val_ratio: float = 0.0,
    seed: int = 42,
    shuffle_before_write: bool = True,
) -> tuple[Path, Path]:
    train_ds, val_ds = build_parquet_datasets(
        csv_path=csv_path,
        data_source=str(data_source),
        ability=str(ability),
        max_samples=int(max_samples),
        max_prompt_tokens=int(max_prompt_tokens),
        tokenizer_path=str(tokenizer_path),
        prompt_token_counter=prompt_token_counter,
        val_ratio=float(val_ratio),
        seed=int(seed),
        shuffle_before_write=bool(shuffle_before_write),
    )
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    train_ds.to_parquet(str(train_path))
    val_ds.to_parquet(str(val_path))
    return train_path, val_path


def _load_rows_from_parquet(path: Path) -> List[Dict[str, Any]]:
    table = pq.read_table(path)
    return table.to_pylist()


def _expand_rows(
    rows: List[Dict[str, Any]],
    *,
    target_count: int,
    full_repeats: int,
    remainder: int,
    rng,
) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    for _ in range(full_repeats):
        expanded.extend(rows)
    if remainder > 0:
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        expanded.extend(rows[i] for i in indices[:remainder])
    if len(expanded) != target_count:
        raise AssertionError(f"expanded rows mismatch: got={len(expanded)} target={target_count}")
    return expanded


def build_merge_plan(
    *,
    dataset_sizes: Dict[str, int],
    ratios: Dict[str, int],
    repeats: Dict[str, int],
) -> Dict[str, Dict[str, int]]:
    import math

    if ratios and repeats:
        raise ValueError("use either ratios or repeats, not both")
    if not ratios and not repeats:
        raise ValueError("must provide ratios or repeats")

    missing = set(ratios or repeats) - set(dataset_sizes)
    if missing:
        raise ValueError(f"datasets not found in sizes: {sorted(missing)}")

    plan: Dict[str, Dict[str, int]] = {}
    if ratios:
        unit = max(math.ceil(dataset_sizes[name] / ratios[name]) for name in ratios)
        for name, ratio in ratios.items():
            size = int(dataset_sizes[name])
            target = int(ratio) * int(unit)
            plan[name] = {
                "source_count": size,
                "target_count": target,
                "full_repeats": target // size,
                "remainder": target % size,
            }
        return plan

    for name, repeat in repeats.items():
        size = int(dataset_sizes[name])
        plan[name] = {
            "source_count": size,
            "target_count": size * int(repeat),
            "full_repeats": int(repeat),
            "remainder": 0,
        }
    return plan


def merge_split_tables(
    tables: Dict[str, List[Dict[str, Any]]],
    *,
    ratios: Dict[str, int],
    repeats: Dict[str, int],
    shuffle: bool,
    seed: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    import random

    sizes = {name: len(rows) for name, rows in tables.items()}
    plan = build_merge_plan(dataset_sizes=sizes, ratios=ratios, repeats=repeats)
    rng = random.Random(int(seed))
    merged: List[Dict[str, Any]] = []
    for name in sorted(plan):
        item = plan[name]
        merged.extend(
            _expand_rows(
                tables[name],
                target_count=int(item["target_count"]),
                full_repeats=int(item["full_repeats"]),
                remainder=int(item["remainder"]),
                rng=rng,
            )
        )
    if shuffle:
        rng.shuffle(merged)
    return merged, plan


def write_merged_parquet_datasets(
    *,
    sources: Dict[str, Path],
    out_dir: Path,
    ratios: Dict[str, int],
    repeats: Dict[str, int],
    shuffle: bool,
    seed: int,
) -> Dict[str, Dict[str, Dict[str, int]]]:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict[str, Dict[str, int]]] = {}
    for split in ("train", "val"):
        tables = {
            name: _load_rows_from_parquet(path / f"{split}.parquet")
            for name, path in sources.items()
        }
        merged_rows, plan = merge_split_tables(
            tables,
            ratios=ratios,
            repeats=repeats,
            shuffle=shuffle,
            seed=int(seed) if split == "train" else int(seed) + 1,
        )
        datasets.Dataset.from_list(merged_rows).to_parquet(str(out_dir / f"{split}.parquet"))
        summary[split] = plan
    return summary


def render_merge_summary(summary: Dict[str, Dict[str, Dict[str, int]]]) -> str:
    lines: List[str] = []
    for split in ("train", "val"):
        lines.append(f"[{split}]")
        plan = summary[split]
        for name in sorted(plan):
            item = plan[name]
            lines.append(
                f"{name}: source={item['source_count']} target={item['target_count']} "
                f"full_repeats={item['full_repeats']} remainder={item['remainder']}"
            )
    return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare VERL parquet from SFT CSV, or merge multiple VERL parquet dirs.")
    p.add_argument(
        "--mode",
        choices=("csv", "merge"),
        default="csv",
        help="csv: build parquet from SFT CSV; merge: merge existing parquet dirs.",
    )
    p.add_argument(
        "--csv",
        "--input-file",
        dest="csv",
        type=Path,
        default=None,
        help="Input CSV (e.g. /path/data/KPI/result/train/kpi_detector_sft.csv)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output dir; writes train.parquet and val.parquet",
    )
    p.add_argument("--data-source", default="anoma/detector_sft", help="data_source field")
    p.add_argument("--ability", default="ts_anomaly_detection", help="ability field")
    p.add_argument("--max-samples", type=int, default=0, help="0=all; else cap")
    p.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=7000,
        help="Drop samples whose prompt token count exceeds this limit. 0 disables filtering.",
    )
    p.add_argument(
        "--tokenizer-path",
        required=True,
        help="Tokenizer path used for prompt token counting.",
    )
    p.add_argument("--val-ratio", type=float, default=0.0, help="0=val=train; else split ratio")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--shuffle-before-write",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle examples before writing parquet. Default: enabled.",
    )
    p.add_argument("--source", action="append", default=[], help="Dataset mapping: name=/path/to/parquet_dir")
    p.add_argument("--ratio", action="append", default=[], help="Target mix ratio: name=int")
    p.add_argument("--repeat", action="append", default=[], help="Integer full-dataset repeat count: name=int")
    p.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable final shuffle after merging parquet datasets.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).resolve()

    if args.mode == "csv":
        if args.csv is None:
            raise SystemExit("--csv is required when --mode=csv")
        train_path, val_path = write_parquet_datasets(
            csv_path=Path(args.csv).resolve(),
            out_dir=out_dir,
            data_source=str(args.data_source),
            ability=str(args.ability),
            max_samples=int(args.max_samples),
            max_prompt_tokens=int(args.max_prompt_tokens),
            tokenizer_path=str(args.tokenizer_path),
            val_ratio=float(args.val_ratio),
            seed=int(args.seed),
            shuffle_before_write=bool(args.shuffle_before_write),
        )
        train_rows = pq.read_table(train_path).num_rows
        val_rows = pq.read_table(val_path).num_rows
        print(f"saved train={train_path} rows={train_rows}")
        print(f"saved val={val_path} rows={val_rows}")
        return 0

    sources = parse_kv_paths(args.source)
    ratios = parse_kv_ints(args.ratio)
    repeats = parse_kv_ints(args.repeat)
    if not sources:
        raise SystemExit("no sources provided")
    for name, path in sources.items():
        if not path.exists():
            raise SystemExit(f"source dir does not exist: {name}={path}")
        for split in ("train.parquet", "val.parquet"):
            if not (path / split).is_file():
                raise SystemExit(f"missing {split}: {path}")

    summary = write_merged_parquet_datasets(
        sources=sources,
        out_dir=out_dir,
        ratios=ratios,
        repeats=repeats,
        shuffle=not bool(args.no_shuffle),
        seed=int(args.seed),
    )
    print(render_merge_summary(summary))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
