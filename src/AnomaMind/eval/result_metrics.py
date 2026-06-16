from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

_SERIES_ID_RE = re.compile(r"^(\d+)_")
_SEG_RE = re.compile(r"_seg(\d+)\.json$", re.IGNORECASE)
_SEG_DIR_RE = re.compile(r"_seg(\d+)$", re.IGNORECASE)
_KPI_SERIES_RE = re.compile(
    r"^(series_\d+_[0-9a-fA-F-]+)_seg(\d+)\.json$",
    re.IGNORECASE,
)
_SERIES_PREFIX_BEFORE_SEG_RE = re.compile(r"^(series_.+?)_seg(\d+)\.json$", re.IGNORECASE)
_SKIP_RESULT_JSON_NAMES = frozenset({"meta_info.json"})


def parse_series_and_seg(
    filename: str,
    segment_dir: Optional[str] = None,
) -> Tuple[str, int]:
    base = Path(filename).name

    gm = _SEG_RE.search(base)
    seg = int(gm.group(1)) if gm else 0

    sm = _SERIES_ID_RE.match(base)
    if sm:
        return sm.group(1), seg

    km = _KPI_SERIES_RE.match(base)
    if km:
        return km.group(1), int(km.group(2))

    pm = _SERIES_PREFIX_BEFORE_SEG_RE.match(base)
    if pm:
        return pm.group(1), int(pm.group(2))

    if segment_dir:
        leaf = Path(str(segment_dir)).name
        dm = _SEG_DIR_RE.search(leaf)
        dseg = int(dm.group(1)) if dm else seg
        if leaf.startswith("series_"):
            return _SEG_DIR_RE.sub("", leaf), dseg
        sm2 = _SERIES_ID_RE.match(leaf)
        if sm2:
            return sm2.group(1), dseg

    return "__no_series_id__", seg


def series_sort_key(series: str) -> Tuple[int, str]:
    if series == "__no_series_id__":
        return (1, series)
    if series.isdigit():
        return (0, f"{int(series):012d}")
    return (0, series)


def import_basic_metricor():
    from AnomaMind.eval.backend.basic_metrics import basic_metricor

    return basic_metricor


def pred_vector_from_intervals(segment_df, pred_indices: set) -> np.ndarray:
    idx = segment_df["Index"].astype(int).values
    return np.array([1 if int(i) in pred_indices else 0 for i in idx], dtype=np.int32)


def label_vector(segment_df) -> np.ndarray:
    if "Label" not in segment_df.columns:
        raise ValueError("segment_data missing Label column")
    return segment_df["Label"].astype(int).values


def load_result_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return json.loads(text, strict=False)
    except Exception as exc:
        print(f"[skip] failed to read JSON {path}: {exc}", file=sys.stderr)
        return None


def auc_pr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_pred, dtype=np.float64)
    if int(np.sum(y_true == 1)) == 0:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_score, pos_label=1))
    except ValueError:
        return float("nan")


def metrics_one(
    grader,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, float, float, float, float]:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    score = y_pred.astype(np.float64)
    pre = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    rf1 = float(grader.metric_RF1(y_true, score, preds=y_pred))
    pa = float(grader.metric_PointF1PA(y_true, score, preds=y_pred))
    ap = auc_pr(y_true, y_pred)
    return pre, rec, f1, rf1, pa, ap


def fmt_metric(x: float, width: int = 9) -> str:
    if x != x:
        return f"{'nan':>{width}}"
    return f"{x:>{width}.6f}"


def render_metrics_table(
    *,
    title: str,
    overall: Tuple[float, float, float, float, float, float],
    series_rows: List[Tuple[str, int, float, float, float, float, float, float]],
    overall_note: str = "",
) -> str:
    lines = [title]
    hdr = (
        f"{'scope':<18}{'Pre':>9}{'Rec':>9}{'F1':>9}"
        f"{'RANGE-F1':>9}{'PA-F1':>9}{'AUC-PR':>9}"
    )
    lines.append(hdr)
    overall_label = "Overall (series mean)" + (f"[{overall_note}]" if overall_note else "")
    lines.append(
        f"{overall_label:<18}"
        f"{fmt_metric(overall[0])}{fmt_metric(overall[1])}{fmt_metric(overall[2])}"
        f"{fmt_metric(overall[3])}{fmt_metric(overall[4])}{fmt_metric(overall[5])}"
    )
    for series, nseg, pre, rec, f1, rf1, pa, auc_pr in series_rows:
        label = f"{series} (n={nseg})"
        lines.append(
            f"{label:<18}"
            f"{fmt_metric(pre)}{fmt_metric(rec)}{fmt_metric(f1)}"
            f"{fmt_metric(rf1)}{fmt_metric(pa)}{fmt_metric(auc_pr)}"
        )
    return "\n".join(lines)


def collect_result_json_files(res_dir: Path, *, recursive: bool) -> List[Path]:
    iterator = res_dir.rglob("*.json") if recursive else res_dir.glob("*.json")
    return sorted(
        p
        for p in iterator
        if p.is_file() and not p.name.startswith("_") and p.name not in _SKIP_RESULT_JSON_NAMES
    )


DEFAULT_CONFIDENCE_THRESHOLD = 1.0


def _normalize_interval(interval: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(interval, (list, tuple)) or len(interval) == 0:
        return None
    try:
        if len(interval) == 1:
            start = end = int(interval[0])
        else:
            start = int(interval[0])
            end = int(interval[1])
    except (TypeError, ValueError, IndexError):
        return None
    return min(start, end), max(start, end)


def pred_score_vector_from_confidence(
    segment_df,
    intervals: Sequence[Any],
    confidences: Optional[Sequence[Any]] = None,
) -> np.ndarray:
    idx = segment_df["Index"].astype(int).values
    score_by_index: dict[int, float] = {}
    conf_list = list(confidences or [])
    for i, interval in enumerate(intervals):
        bounds = _normalize_interval(interval)
        if bounds is None:
            continue
        start, end = bounds
        try:
            confidence = int(conf_list[i]) if i < len(conf_list) else 2
        except (TypeError, ValueError):
            confidence = 2
        confidence = 2 if confidence >= 2 else 1
        for point in range(start, end + 1):
            score_by_index[point] = max(score_by_index.get(point, 0.0), float(confidence))
    return np.array([score_by_index.get(int(i), 0.0) for i in idx], dtype=np.float64)


def pred_vector_at_threshold(y_score: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(y_score, dtype=np.float64) >= float(threshold)).astype(np.int32)


def best_f1_at_thresholds(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    if int(np.sum(y_true == 1)) == 0:
        return float("nan"), float("nan")

    unique_scores = sorted(set(float(x) for x in y_score.tolist() if float(x) > 0))
    candidates = [0.5] + unique_scores
    best_f1 = -1.0
    best_threshold = 0.5
    for threshold in candidates:
        y_pred = pred_vector_at_threshold(y_score, threshold)
        f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    if best_f1 < 0:
        return float("nan"), float("nan")
    return best_f1, best_threshold


def metrics_one_with_confidence(
    grader: Any,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Tuple[float, float, float, float, float, float, float, float]:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = pred_vector_at_threshold(y_score, confidence_threshold)

    pre = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    rf1 = float(grader.metric_RF1(y_true, y_score, preds=y_pred))
    pa = float(grader.metric_PointF1PA(y_true, y_score, preds=y_pred))
    best_f1, best_threshold = best_f1_at_thresholds(y_true, y_score)
    ap = auc_pr(y_true, y_score)
    return pre, rec, f1, rf1, pa, best_f1, best_threshold, ap


def render_confidence_metrics_table(
    *,
    title: str,
    overall: Tuple[float, float, float, float, float, float, float, float],
    series_rows: List[Tuple[str, int, float, float, float, float, float, float, float, float]],
    overall_note: str = "",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> str:
    lines = [title]
    hdr = (
        f"{'scope':<18}{'Pre':>9}{'Rec':>9}{'F1':>9}"
        f"{'RANGE-F1':>9}{'PA-F1':>9}{'Best-F1':>9}{'AUC-PR':>9}"
    )
    lines.append(hdr)
    overall_label = "Overall (series mean)" + (f"[{overall_note}]" if overall_note else "")
    lines.append(
        f"{overall_label:<18}"
        f"{fmt_metric(overall[0])}{fmt_metric(overall[1])}{fmt_metric(overall[2])}"
        f"{fmt_metric(overall[3])}{fmt_metric(overall[4])}{fmt_metric(overall[5])}{fmt_metric(overall[7])}"
    )
    lines.append(
        f"{'Best-F1 threshold':<18}"
        f"{'':>9}{'':>9}{'':>9}{'':>9}{'':>9}"
        f"{fmt_metric(overall[6])}{'':>9}"
    )
    lines.append(f"{'Fixed threshold':<18}{confidence_threshold:>9.1f}")
    for series, nseg, pre, rec, f1, rf1, pa, best_f1, _best_thr, auc_pr_val in series_rows:
        label = f"{series} (n={nseg})"
        lines.append(
            f"{label:<18}"
            f"{fmt_metric(pre)}{fmt_metric(rec)}{fmt_metric(f1)}"
            f"{fmt_metric(rf1)}{fmt_metric(pa)}{fmt_metric(best_f1)}{fmt_metric(auc_pr_val)}"
        )
    return "\n".join(lines)
