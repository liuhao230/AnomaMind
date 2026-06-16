from __future__ import annotations

import json
from pathlib import Path
from typing import AbstractSet, Any, Dict, List, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import StructuredTool

from AnomaMind.core.utils import get_column_names, resolve_global_statistics_path, round_value


'''
### `grep`
- **Args:** range `[start, end]`.
- **Returns:** JSON object `{{ "<index>": <value>, ... }}` for points whose index lies in the range. Use to zoom into Locator-suggested or Tool-detected intervals.
- **Important:** Visual localization may be misaligned. Use a **broader** index range than the localization suggested windows.(Don't be too large) 
If other tools flag points outside those suspicious intervals, inspect those values as well.
#### Important
Only call the interval with evidence, do not include to those intervals without evidence, which costs token(eg. DO NOT call whole segment though no anomaly envidence detect).
If there are two anomaly intervals appearantly seperate, do not call in a single but much larger grep.

'''


def get_detector_tools_prompt_block() -> str:
    return """
## Rule
**Parallelize independent calls:** In one tool round, call `stats`, `value_zscore`, and `diff_zscore` **together** (no ordering dependency). 
If Locator already gives usable global index bounds, add `grep(start,end)` in the **same** batch;

""".strip()


def get_detector_tool_result_reference_block() -> str:
    return """
## Tool result reference

### `stats`
- Provides local segment summary statistics and global baseline statistics.
- Use it to understand the overall value range, spread, and whether a candidate anomaly is globally extreme or only locally unusual.

### `grep`
- Returns raw values for a selected global index range.
- Use it to confirm exact anomalous points, refine interval boundaries, and verify whether suspicious points align with nearby context.

### `value_zscore`
- Measures each point against the full-series value baseline.
- Use it mainly for global level outliers or unusually large absolute values.

### `diff_zscore`
- Measures local changes against the full-series diff baseline.
- Use it mainly for sharp spikes, dips, and context anomalies that are obvious in change magnitude.
""".strip()


def _normalize_thresholds(thresholds: Any) -> List[float]:
    if thresholds is None:
        return []
    if isinstance(thresholds, (int, float)):
        return [float(thresholds)]
    if isinstance(thresholds, (list, tuple)):
        out: List[float] = []
        for x in thresholds:
            try:
                out.append(float(x))
            except Exception:
                continue
        out = sorted(set([t for t in out if t > 0]))
        return out
    return []


def _bucket_points_by_thresholds(
    indices: List[int],
    zscores: List[float],
    thresholds: List[float],
    z_round: int = 2,
) -> Dict[str, Any]:
    th = _normalize_thresholds(thresholds)
    if not th:
        return {"thresholds": [], "bins": []}
    absz = [abs(float(z)) for z in zscores]
    kept = [(int(i), float(z), float(a)) for i, z, a in zip(indices, zscores, absz) if a > th[0]]
    bins: List[Dict[str, Any]] = []
    for i, lo in enumerate(th):
        hi = th[i + 1] if i + 1 < len(th) else None
        pts: List[List[Any]] = []
        for idx, z, a in kept:
            if a <= lo:
                continue
            if hi is not None and a > hi:
                continue
            pts.append([idx, round_value(z, z_round)])
        if hi is None:
            r = f"({round_value(lo, 2)},inf)"
        else:
            r = f"({round_value(lo, 2)},{round_value(hi, 2)}]"
        bins.append({"range": r, "points": pts})
    return {"thresholds": [round_value(x, 2) for x in th], "bins": bins}


def _load_global_statistics_json(segment_folder_path: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    gp = resolve_global_statistics_path(Path(segment_folder_path))
    if gp is None or not gp.is_file():
        return None, "global statistics file not found (expected meta_info.json + _global/statistics/...)"
    try:
        with open(gp, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return None, "global statistics JSON is not an object"
        return obj, None
    except Exception as e:
        return None, f"Failed to load global statistics: {str(e)}"


def create_detector_tools(
    data: pd.DataFrame,
    segment_folder_path: Optional[str] = None,
) -> List[StructuredTool]:
    """Detector AgentLoop：stats + grep + value_zscore + diff_zscore。"""
    timestamp_col, value_col = get_column_names(data)
    tools: List[StructuredTool] = []

    def stats() -> str:
        if segment_folder_path is None:
            return json.dumps(
                {"error": "segment_folder_path not provided"},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        try:
            local_values = data[value_col]
        except Exception as e:
            return json.dumps(
                {"error": f"Failed to read local segment values: {str(e)}"},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        global_stats, err = _load_global_statistics_json(segment_folder_path)
        if err or global_stats is None:
            return json.dumps(
                {"error": err or "missing global statistics"},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        if "mean" not in global_stats or "std" not in global_stats or "median" not in global_stats:
            return json.dumps(
                {"error": "global statistics JSON does not contain mean, std, or median"},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        stats_obj: Dict[str, Any] = {
            "local": {
                "mean": round_value(local_values.mean(), 3),
                "std": round_value(local_values.std(), 3),
                "min": round_value(local_values.min(), 3),
                "max": round_value(local_values.max(), 3),
                "median": round_value(local_values.median(), 3),
                "quantiles": {
                    "p5": round_value(local_values.quantile(0.05), 3),
                    "p10": round_value(local_values.quantile(0.10), 3),
                    "p25": round_value(local_values.quantile(0.25), 3),
                    "p75": round_value(local_values.quantile(0.75), 3),
                    "p90": round_value(local_values.quantile(0.90), 3),
                    "p95": round_value(local_values.quantile(0.95), 3),
                },
            },
            "global": {
                "mean": round_value(global_stats["mean"], 3),
                "std": round_value(global_stats["std"], 3),
                "median": round_value(global_stats["median"], 3),
            },
        }
        return json.dumps(stats_obj, ensure_ascii=False, separators=(",", ":"))

    tools.append(
        StructuredTool.from_function(
            func=stats,
            name="stats",
            description=(
                "Summary statistics. Args: empty object {}. "
                "Returns local {mean,std,min,max,median,quantiles:{p5,p10,p25,p75,p90,p95}} "
                "and global {mean,std,median}. Use this to understand the segment's overall value range, "
                "spread, and whether suspicious behavior is only locally unusual or globally extreme."
            ),
        )
    )

    def _index_bounds() -> tuple[int, int]:
        if timestamp_col in data.columns:
            idxs = data[timestamp_col].astype(int).tolist()
            return int(min(idxs)), int(max(idxs))
        return 0, len(data) - 1

    def grep(start: int, end: int) -> str:
        lo, hi = _index_bounds()
        a = int(max(min(start, end), lo))
        b = int(min(max(start, end), hi))
        if timestamp_col in data.columns:
            ic = data[timestamp_col].astype(int)
            sub = data.loc[(ic >= a) & (ic <= b)]
        else:
            sub = data.iloc[a : b + 1]
        out: Dict[str, Any] = {}
        for _, row in sub.iterrows():
            try:
                ix = int(row[timestamp_col]) if timestamp_col in sub.columns else int(row.name)
            except Exception:
                continue
            try:
                v = float(row[value_col])
                out[str(ix)] = round(float(v), 3)
            except Exception:
                out[str(ix)] = row[value_col]
        return json.dumps(out, ensure_ascii=False, separators=(",", ":"))

    tools.append(
        StructuredTool.from_function(
            func=grep,
            name="grep",
            description=(
                "Fetch time-series values for an inclusive global index range [start, end] within this segment. "
                "Returns JSON object mapping index string to value. Use it to zoom into suspicious intervals (supported by localization or tool results), "
                "confirm exact anomalous points, and refine interval boundaries against nearby context. "
                "Prefer evidence-backed short calls: keep each interval length <= 200, use a slightly broader "
                "window when visual localization may be misaligned, and avoid merging obviously separate anomaly "
                "regions into one much larger grep call. Do not grep with intervals that has been called before"
            ),
        )
    )

    def value_zscore(thresholds: Any) -> str:
        value_series = data[value_col].astype(float)
        if len(value_series) == 0:
            return json.dumps({"error": "empty segment"}, ensure_ascii=False, separators=(",", ":"))
        if segment_folder_path is None:
            return json.dumps(
                {"error": "segment_folder_path not provided"},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        global_stats, err = _load_global_statistics_json(segment_folder_path)
        if err or global_stats is None:
            return json.dumps(
                {"error": err or "missing global statistics"},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        if "mean" not in global_stats or "std" not in global_stats:
            return json.dumps(
                {"error": "global statistics JSON does not contain mean or std"},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        ref_mean = float(global_stats["mean"])
        ref_std = float(global_stats["std"])
        if ref_std <= 0:
            ref_std = 1e-12
        result: Dict[str, Any] = {
            "global_mean": round_value(ref_mean, 3),
            "global_std": round_value(ref_std, 3),
        }
        if timestamp_col in data.columns:
            indices = data[timestamp_col].values.tolist()
        else:
            indices = list(range(len(value_series)))
        zscores = ((value_series - ref_mean) / ref_std).astype(float).tolist()
        result.update(
            _bucket_points_by_thresholds([int(i) for i in indices], [float(z) for z in zscores], thresholds)
        )
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))

    tools.append(
        StructuredTool.from_function(
            func=value_zscore,
            name="value_zscore",
            description=(
                "Value |z| for the segment vs **full-series** mean/std. "
                "REQUIRED: thresholds (float list), e.g. [2.0,2.5,3.0]. "
                "Returns JSON with thresholds, bins, global_mean, global_std, where bins group segment points by "
                "absolute z-score range against the full-series baseline. Use mainly for global-level outliers or "
                "unusually large absolute values."
            ),
        )
    )

    def diff_zscore(thresholds: Any) -> str:
        value_series = data[value_col].astype(float)
        if len(value_series) < 2:
            return json.dumps(
                {
                    "error": "segment too short for diff",
                    "global_diff_mean": 0.0,
                    "global_diff_std": 0.0,
                    "thresholds": [],
                    "bins": [],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        if segment_folder_path is None:
            return json.dumps(
                {"error": "segment_folder_path not provided"},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        global_stats, err = _load_global_statistics_json(segment_folder_path)
        if err or global_stats is None:
            return json.dumps(
                {"error": err or "missing global statistics"},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        if "diff_mean" not in global_stats or "diff_std" not in global_stats:
            return json.dumps(
                {"error": "global statistics JSON does not contain diff_mean or diff_std"},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        dmean = float(global_stats["diff_mean"])
        dstd = float(global_stats["diff_std"])
        if dstd <= 0:
            dstd = 1e-12
        result: Dict[str, Any] = {
            "global_diff_mean": round_value(dmean, 3),
            "global_diff_std": round_value(dstd, 3),
        }
        diff_values = np.diff(value_series.values)
        diff_z = (diff_values - dmean) / dstd
        if timestamp_col in data.columns:
            diff_indices = data[timestamp_col].values[1:].tolist()
        else:
            diff_indices = list(range(1, len(diff_z) + 1))
        result.update(
            _bucket_points_by_thresholds([int(i) for i in diff_indices], [float(z) for z in diff_z], thresholds)
        )
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))

    tools.append(
        StructuredTool.from_function(
            func=diff_zscore,
            name="diff_zscore",
            description=(
                "First-difference |z| on this segment vs **full-series** diff_mean/diff_std. "
                "REQUIRED: thresholds (float list), e.g. [3.0,3.5]. "
                "Returns JSON with thresholds, bins, global_diff_mean, global_diff_std, where bins group segment "
                "differences by absolute z-score range against the full-series diff baseline. Use mainly for sharp "
                "spikes, dips, and context anomalies that are obvious in change magnitude."
            ),
        )
    )

    return tools
