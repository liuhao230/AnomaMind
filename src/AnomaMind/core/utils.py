from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_csv_full_lock = threading.Lock()
_csv_full_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
_MAX_CSV_CACHE_FILES = 24
_segment_folder_lock = threading.Lock()
_segment_folder_cache: Dict[str, Optional[str]] = {}


def _read_global_csv_cached(full_path: Path) -> pd.DataFrame:
    p = full_path.resolve()
    key = str(p)
    try:
        mtime = float(p.stat().st_mtime)
    except OSError:
        return pd.read_csv(p)
    with _csv_full_lock:
        hit = _csv_full_cache.get(key)
        if hit is not None and hit[0] == mtime:
            return hit[1]
    df = pd.read_csv(p)
    with _csv_full_lock:
        hit2 = _csv_full_cache.get(key)
        if hit2 is not None and hit2[0] == mtime:
            return hit2[1]
        _csv_full_cache[key] = (mtime, df)
        while len(_csv_full_cache) > _MAX_CSV_CACHE_FILES:
            try:
                drop = next(iter(_csv_full_cache))
                del _csv_full_cache[drop]
            except (StopIteration, KeyError):
                break
    return df


def _resolve_segment_folder_with_meta(folder: Path) -> Path:
    folder = Path(folder).resolve()
    meta_path = folder / "meta_info.json"
    if meta_path.is_file():
        return folder

    key = str(folder)
    with _segment_folder_lock:
        cached = _segment_folder_cache.get(key)
    if cached is not None:
        return Path(cached) if cached else folder

    chosen: Optional[Path] = None
    parts = folder.parts
    if "processed" in parts:
        processed_index = parts.index("processed")
        processed_root = Path(*parts[: processed_index + 1])
        if processed_root.is_dir():
            matches = sorted(
                p
                for p in processed_root.rglob(folder.name)
                if p.is_dir() and (p / "meta_info.json").is_file()
            )
            if matches:
                chosen = matches[0].resolve()

    with _segment_folder_lock:
        _segment_folder_cache[key] = str(chosen) if chosen is not None else ""
    return chosen or folder


def load_segment_meta(folder: Path) -> Dict[str, Any]:
    meta_path = _resolve_segment_folder_with_meta(Path(folder)) / "meta_info.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception as exc:
        logger.warning(f"load meta_info.json error ({meta_path}): {exc}")
        return {}


def meta_indexes_global_series(folder: Path) -> bool:
    meta = load_segment_meta(folder)
    if not meta.get("global_data_file") or not isinstance(meta.get("index"), dict):
        return False
    idx = meta["index"]
    return "start" in idx and "end" in idx


def resolve_dataset_root(segment_folder: Path, meta: Dict[str, Any]) -> Path:
    rel = meta.get("dataset_root", "..")
    return (Path(segment_folder) / rel).resolve()


def resolve_global_statistics_path(segment_folder: Path) -> Optional[Path]:
    folder = _resolve_segment_folder_with_meta(Path(segment_folder))
    meta = load_segment_meta(folder)
    if not meta:
        return None
    try:
        root = resolve_dataset_root(folder, meta)
        rel = meta.get("global_statistics_file")
        if not rel:
            stem = meta.get("series_stem", "")
            if not stem:
                return None
            rel = f"_global/statistics/{stem}_global_statistics.json"
        path = (root / rel).resolve()
        if path.is_file():
            return path
    except Exception as exc:
        logger.warning(f"resolve global_statistics path error ({folder}): {exc}")
    return None


def _merged_intervals_from_sorted_indices(idxs: List[int]) -> List[List[int]]:
    if not idxs:
        return []
    out: List[List[int]] = []
    start = end = int(idxs[0])
    for x in idxs[1:]:
        x = int(x)
        if x == end + 1:
            end = x
        else:
            out.append([start, end])
            start = end = x
    out.append([start, end])
    return out


def ground_truth_intervals_from_labeled_segment(segment_df: pd.DataFrame) -> List[List[int]]:
    if segment_df.empty or "Label" not in segment_df.columns:
        return []
    index_col = "Index" if "Index" in segment_df.columns else None
    if index_col is None:
        return []
    sub = segment_df[segment_df["Label"].astype(int) == 1]
    if sub.empty:
        return []
    idxs = sorted(sub[index_col].astype(int).tolist())
    return _merged_intervals_from_sorted_indices(idxs)


def _read_segment_type(folder: Path) -> str:
    meta = load_segment_meta(folder)
    segment_type = meta.get("type") or meta.get("dataset_type")
    if isinstance(segment_type, str) and segment_type.strip():
        return segment_type.strip().upper()
    return "UNKNOWN"


def load_segment_data(segment_folder: Path) -> Dict[str, Any]:
    folder = _resolve_segment_folder_with_meta(Path(segment_folder))
    if not meta_indexes_global_series(folder):
        raise FileNotFoundError(
            f"need meta_info.json with global_data_file and index.start/index.end: {folder}"
        )

    meta = load_segment_meta(folder)
    root = resolve_dataset_root(folder, meta)
    rel_data = meta.get("global_data_file")
    if not rel_data:
        raise FileNotFoundError(f"meta missing global_data_file: {folder}")
    full_path = (root / rel_data).resolve()
    if not full_path.is_file():
        raise FileNotFoundError(f"global data file not found: {full_path}")

    idx = meta.get("index") or {}
    start = int(idx["start"])
    end = int(idx["end"])

    df_full = _read_global_csv_cached(full_path)
    n = len(df_full)
    if start < 0 or end > n or start >= end:
        raise ValueError(f"invalid meta index {{start:{start}, end:{end}}} for full length {n}")

    segment_df = df_full.iloc[start:end].copy()
    if "Index" not in segment_df.columns:
        segment_df.insert(0, "Index", np.arange(start, end, dtype=int))
    else:
        segment_df["Index"] = segment_df["Index"].astype(int)

    segment_type = _read_segment_type(folder)
    ground_truth = ground_truth_intervals_from_labeled_segment(segment_df)

    return {
        "segment_data": segment_df,
        "path": str(folder.resolve()),
        "ground_truth": ground_truth,
        "type": segment_type,
    }


def get_column_names(data: pd.DataFrame) -> Tuple[str, str]:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(data).__name__}")
    if data.empty:
        raise ValueError("DataFrame is empty")
    if len(data.columns) == 0:
        raise ValueError("DataFrame has no columns")

    timestamp_col = None
    for col in data.columns:
        if col.lower() in ["index", "time", "timestamp", "date"]:
            timestamp_col = col
            break

    value_col = None
    for col in data.columns:
        if col.lower() in ["value", "data", "val", "values"]:
            value_col = col
            break

    if timestamp_col is None:
        timestamp_col = "Index" if "Index" in data.columns else data.columns[0]
    if value_col is None:
        remaining_cols = [c for c in data.columns if c != timestamp_col and c.lower() != "label"]
        value_col = remaining_cols[0] if remaining_cols else data.columns[-1]

    timestamp_col = str(timestamp_col)
    value_col = str(value_col)
    if not isinstance(timestamp_col, str) or not isinstance(value_col, str):
        raise ValueError("get_column_names returned invalid types")
    return timestamp_col, value_col


def round_value(value: float, decimals: int = 4) -> float:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return 0.0
    return round(float(value), decimals)
