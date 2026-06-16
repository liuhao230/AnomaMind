"""
Compute point-level F1, Range-F1, and PA-F1 on smoke/detection result dirs (e.g. process/IOPS/res3).

- F1: sklearn ``f1_score`` on **anomaly positives (Label==1)`` (``pos_label=1``).
- Range-F1: ``metric_RF1`` (alpha=0.2)
- PA-F1: ``metric_PointF1PA``
- AUC-PR: sklearn ``average_precision_score`` (positive=anomaly; **0/1 scores from expanded interval preds**).

**Aggregation**: for names like ``273_IOPS_..._seg042.json``, the leading number (``273``) is the **series**;
within a series, sort by ``seg``, concatenate, then compute **Pre / Rec / F1 / RANGE-F1 / PA-F1 / AUC-PR**.
**Overall** = arithmetic mean over series (not weighted by segment count).
``--both-preds`` prints **final** then **localization-only**.
``-r`` / ``--recursive``: recurse under a **parent dir** (e.g. ``wyc/result/TODS``) and evaluate all ``*.json``.
``--remap-segment-dir FROM TO``: remap stale ``segment_dir`` paths in result JSON before ``load_segment_data``.

CLI quickstart::

  python3 -m AnomaMind.eval.evaluate_results /path/data/KPI/result/test_all/API
  python3 -m AnomaMind.eval.evaluate_results /path/data/WSD/result/test_all -r
  python3 -m AnomaMind.eval.evaluate_results /path/data/KPI/result/test_all/API --both-preds
  python3 -m AnomaMind.eval.evaluate_results /path/data/KPI/result/test_all/API --pred-source localization

Defaults live in ``DEFAULT_CONFIG`` at the top; CLI overrides them.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from AnomaMind.core.reward import intervals_to_set  # noqa: E402
from AnomaMind.core.utils import load_segment_data  # noqa: E402
from AnomaMind.eval.result_metrics import (  # noqa: E402
    DEFAULT_CONFIDENCE_THRESHOLD,
    collect_result_json_files,
    fmt_metric,
    import_basic_metricor,
    label_vector,
    load_result_json,
    metrics_one,
    metrics_one_with_confidence,
    parse_series_and_seg,
    pred_score_vector_from_confidence,
    pred_vector_from_intervals,
    render_confidence_metrics_table,
    render_metrics_table,
    series_sort_key,
)


@dataclass(frozen=True)
class EvalConfig:
    recursive: bool = False
    pred_source: str = "final"
    both_preds: bool = False
    per_file: bool = False
    per_series: bool = False
    skip_missing_segment_dir: bool = False
    # Default behavior: exclude series whose GT has no positive points (npos==0) from the
    # overall "mean over series" aggregation, to avoid penalizing recall/F1 on pure-normal series.
    exclude_no_anomaly_series: bool = True
    confidence_metrics: bool = True
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD


DEFAULT_CONFIG = EvalConfig()


def build_parser(defaults: EvalConfig = DEFAULT_CONFIG) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute F1 / Range-F1 / PA-F1 / AUC-PR (average_precision) on a result directory"
    )
    parser.add_argument(
        "res_dir",
        type=Path,
        help="Result dir with *seg*.json; with -r, pass a parent dir (e.g. data/result/TODS) to recurse",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        default=defaults.recursive,
        help="Recursively collect *.json under subdirectories.",
    )
    parser.add_argument(
        "--pred-source",
        choices=("final", "localization"),
        default=defaults.pred_source,
        help="Prediction intervals: final=anomaly_intervals; localization=visual_localization",
    )
    parser.add_argument(
        "--both-preds",
        action="store_true",
        default=defaults.both_preds,
        help="Print metrics for final then localization.",
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        default=defaults.per_file,
        help="Print per-json metrics.",
    )
    parser.add_argument(
        "--per-series",
        action="store_true",
        default=defaults.per_series,
        help="Print per-series metrics and segment counts.",
    )
    parser.add_argument(
        "--skip-missing-segment-dir",
        action="store_true",
        default=defaults.skip_missing_segment_dir,
        help="Skip when segment_dir is missing/invalid (quieter logs).",
    )
    # Prefer BooleanOptionalAction when available so users can do:
    #   --exclude-no-anomaly-series / --no-exclude-no-anomaly-series
    _bool_opt = getattr(argparse, "BooleanOptionalAction", None)
    if _bool_opt is not None:
        parser.add_argument(
            "--exclude-no-anomaly-series",
            action=_bool_opt,
            default=defaults.exclude_no_anomaly_series,
            help=(
                "Exclude series with no GT anomaly points (npos==0) from overall (series-mean) aggregation. "
                "Enabled by default; use --no-exclude-no-anomaly-series to include pure-normal series."
            ),
        )
    else:
        # Fallback for older Python: provide an explicit include flag to disable exclusion.
        parser.add_argument(
            "--exclude-no-anomaly-series",
            action="store_true",
            default=defaults.exclude_no_anomaly_series,
            help="(legacy) Same as --exclude-no-anomaly-series; on by default.",
        )
        parser.add_argument(
            "--include-no-anomaly-series",
            action="store_true",
            default=False,
            help="Include series with no GT anomalies in overall mean (lowers F1/Rec).",
        )
    parser.add_argument(
        "--remap-segment-dir",
        nargs=2,
        metavar=("FROM_PREFIX", "TO_PREFIX"),
        default=None,
        help=(
            "Replace segment_dir prefix FROM_PREFIX with TO_PREFIX before loading GT. "
            "KPI example: FROM=/path/data/processed TO=/path/data/KPI/processed"
        ),
    )
    if _bool_opt is not None:
        parser.add_argument(
            "--confidence-metrics",
            action=_bool_opt,
            default=defaults.confidence_metrics,
            help="When pred_source=final, also print Best-F1/AUC-PR using anomaly_confidences.",
        )
    else:
        parser.add_argument(
            "--confidence-metrics",
            action="store_true",
            default=defaults.confidence_metrics,
        )
        parser.add_argument("--no-confidence-metrics", action="store_true", default=False)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=defaults.confidence_threshold,
        help="Fixed score threshold for confidence-aware Pre/Rec/F1 table.",
    )
    return parser


def parse_args(
    argv: Optional[Sequence[str]] = None,
    defaults: EvalConfig = DEFAULT_CONFIG,
) -> argparse.Namespace:
    return build_parser(defaults).parse_args(argv)


def config_from_args(args: argparse.Namespace) -> EvalConfig:
    # Back-compat: if fallback flag is present, it overrides the default exclusion.
    exclude_no_anomaly_series = bool(getattr(args, "exclude_no_anomaly_series", False))
    if bool(getattr(args, "include_no_anomaly_series", False)):
        exclude_no_anomaly_series = False
    confidence_metrics = bool(getattr(args, "confidence_metrics", DEFAULT_CONFIG.confidence_metrics))
    if bool(getattr(args, "no_confidence_metrics", False)):
        confidence_metrics = False
    return EvalConfig(
        default_res_dir=Path(args.res_dir),
        recursive=bool(args.recursive),
        pred_source=str(args.pred_source),
        both_preds=bool(args.both_preds),
        per_file=bool(args.per_file),
        per_series=bool(args.per_series),
        skip_missing_segment_dir=bool(args.skip_missing_segment_dir),
        exclude_no_anomaly_series=exclude_no_anomaly_series,
        confidence_metrics=confidence_metrics,
        confidence_threshold=float(args.confidence_threshold),
    )


def _intervals_from_result(
    data: Dict[str, Any], pred_source: str
) -> Optional[List[Any]]:
    """Return interval list for intervals_to_set; None to skip file when invalid."""
    if pred_source == "final":
        fo = data.get("final_output") or {}
        if fo.get("error"):
            return None
        intervals = fo.get("anomaly_intervals")
        # Important: if the model predicts "no anomalies", many pipelines store it as
        # anomaly_intervals=[] or even omit the field. We treat missing/None as empty
        # prediction so normal segments participate in overall metrics.
        if intervals is None:
            return []
        if isinstance(intervals, list):
            return intervals
        # Best-effort: sometimes it's a JSON string.
        if isinstance(intervals, str):
            try:
                obj = json.loads(intervals)
                if isinstance(obj, list):
                    return obj
            except Exception:
                pass
        # If it's some other malformed type, treat as empty prediction (don't skip).
        return []
    if pred_source == "localization":
        loc = data.get("visual_localization")
        # Same rationale: allow missing localization to be treated as "predict none".
        if loc is None:
            return []
        if isinstance(loc, list):
            return loc
        return []
    raise ValueError(f"unknown pred_source: {pred_source!r}")


def _print_metrics_table(
    *,
    title: str,
    overall: Tuple[float, float, float, float, float, float],
    series_rows: List[Tuple[str, int, float, float, float, float, float, float]],
    overall_note: str = "",
) -> None:
    print(
        render_metrics_table(
            title=title,
            overall=overall,
            series_rows=series_rows,
            overall_note=overall_note,
        )
    )


def _remap_segment_dir(
    seg_dir: str, remap: Optional[Tuple[str, str]]
) -> str:
    """If ``seg_dir`` starts with ``remap[0]``, replace prefix with ``remap[1]`` (data path migration)."""
    if not remap:
        return seg_dir
    old_p, new_p = remap[0], remap[1]
    s = str(seg_dir)
    if s.startswith(old_p):
        return new_p + s[len(old_p) :]
    return s


def _confidence_payload_from_result(data: Dict[str, Any]) -> Optional[Tuple[List[Any], List[Any]]]:
    fo = data.get("final_output") or {}
    if fo.get("error"):
        return None
    intervals = fo.get("anomaly_intervals")
    if intervals is None:
        intervals = []
    if not isinstance(intervals, list):
        return None
    confidences = fo.get("anomaly_confidences")
    if confidences is None:
        confidences = [2] * len(intervals)
    if not isinstance(confidences, list):
        confidences = [2] * len(intervals)
    return intervals, confidences


def _print_confidence_metrics_table(
    *,
    title: str,
    overall: Tuple[float, float, float, float, float, float, float, float],
    series_rows: List[Tuple[str, int, float, float, float, float, float, float, float, float]],
    overall_note: str = "",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> None:
    print(
        render_confidence_metrics_table(
            title=title,
            overall=overall,
            series_rows=series_rows,
            overall_note=overall_note,
            confidence_threshold=confidence_threshold,
        )
    )


def _run_confidence_eval_for_source(
    *,
    res_dir: Path,
    grader: Any,
    recursive: bool,
    skip_missing_segment_dir: bool,
    exclude_no_anomaly_series: bool,
    confidence_threshold: float,
    segment_dir_remap: Optional[Tuple[str, str]] = None,
) -> int:
    json_files = collect_result_json_files(res_dir, recursive=recursive)
    by_series: DefaultDict[str, List[Tuple[int, np.ndarray, np.ndarray, str]]] = defaultdict(list)

    for jf in json_files:
        data = load_result_json(jf)
        if not data:
            continue
        seg_dir = data.get("segment_dir")
        if not seg_dir:
            continue
        seg_resolved = _remap_segment_dir(str(seg_dir), segment_dir_remap)
        payload = _confidence_payload_from_result(data)
        if payload is None:
            continue
        intervals, confidences = payload
        try:
            pack = load_segment_data(Path(seg_resolved))
        except Exception:
            continue
        sdf = pack["segment_data"]
        try:
            y_true = label_vector(sdf)
        except Exception:
            continue
        y_score = pred_score_vector_from_confidence(sdf, intervals, confidences)
        if len(y_true) != len(y_score):
            continue
        series, seg = parse_series_and_seg(jf.name, seg_resolved)
        by_series[series].append((seg, y_true.copy(), y_score.copy(), jf.name))

    if not by_series:
        print("No samples for confidence-aware metrics.", file=sys.stderr)
        return 1

    series_metrics: List[Tuple[str, int, float, float, float, float, float, float, float, float]] = []
    series_pos_counts: Dict[str, int] = {}
    for series in sorted(by_series.keys(), key=series_sort_key):
        blocks = sorted(by_series[series], key=lambda t: t[0])
        if not blocks:
            continue
        y_s = np.concatenate([t[1] for t in blocks])
        score_s = np.concatenate([t[2] for t in blocks])
        npos = int(np.sum(y_s == 1))
        series_pos_counts[series] = npos
        spre, srec, sf1, srf, spa, sbest, sbest_thr, sap = metrics_one_with_confidence(
            grader,
            y_s,
            score_s,
            confidence_threshold=confidence_threshold,
        )
        series_metrics.append((series, len(blocks), spre, srec, sf1, srf, spa, sbest, sbest_thr, sap))

    if exclude_no_anomaly_series:
        kept = [r for r in series_metrics if series_pos_counts.get(r[0], 0) > 0]
        dropped = len(series_metrics) - len(kept)
        if not kept:
            print("No series with GT positives for confidence metrics.", file=sys.stderr)
            return 1
        overall_note = f"exclude npos=0 ({dropped} dropped)"
    else:
        kept = list(series_metrics)
        overall_note = ""

    mpre = float(np.mean([r[2] for r in kept]))
    mrec = float(np.mean([r[3] for r in kept]))
    mf1 = float(np.mean([r[4] for r in kept]))
    mrf1 = float(np.mean([r[5] for r in kept]))
    mpa = float(np.mean([r[6] for r in kept]))
    mbest = float(np.mean([r[7] for r in kept]))
    mbest_thr = float(np.mean([r[8] for r in kept]))
    mapr_finite = [r[9] for r in kept if r[9] == r[9]]
    mmapr = float(np.mean(mapr_finite)) if mapr_finite else float("nan")

    print()
    _print_confidence_metrics_table(
        title="=== final (confidence-aware: Best-F1 / AUC-PR) ===",
        overall=(mpre, mrec, mf1, mrf1, mpa, mbest, mbest_thr, mmapr),
        series_rows=series_metrics,
        overall_note=overall_note,
        confidence_threshold=confidence_threshold,
    )
    return 0


def _run_eval_for_source(
    *,
    res_dir: Path,
    pred_source: str,
    grader: Any,
    per_file: bool,
    per_series: bool,
    recursive: bool,
    skip_missing_segment_dir: bool,
    exclude_no_anomaly_series: bool,
    segment_dir_remap: Optional[Tuple[str, str]] = None,
) -> int:
    json_files = collect_result_json_files(res_dir, recursive=recursive)
    if not json_files:
        mode = "recursive" if recursive else "current directory"
        print(
            f"No evaluable *.json found ({mode}, skipping _*.json and meta_info.json): {res_dir}",
            file=sys.stderr,
        )
        print(
            "Hint: this script expects detector *seg*.json outputs;"
            "data/.../processed/ usually has slice meta only—use wyc/result/... with -r on a parent dir.",
            file=sys.stderr,
        )
        return 1

    by_series: DefaultDict[str, List[Tuple[int, np.ndarray, np.ndarray, str]]] = (
        defaultdict(list)
    )
    per_rows: List[Tuple[str, float, float, float, float, float, float, int]] = []

    for jf in json_files:
        data = load_result_json(jf)
        if not data:
            continue
        seg_dir = data.get("segment_dir")
        if not seg_dir:
            if not skip_missing_segment_dir:
                print(f"[skip] no segment_dir: {jf.name}", file=sys.stderr)
            continue
        seg_resolved = _remap_segment_dir(str(seg_dir), segment_dir_remap)
        intervals = _intervals_from_result(data, pred_source)
        if intervals is None:
            if pred_source == "final":
                fo = data.get("final_output") or {}
                if fo.get("error"):
                    print(f"[skip] final_output.error: {jf.name}", file=sys.stderr)
                else:
                    print(f"[skip] no anomaly_intervals: {jf.name}", file=sys.stderr)
            else:
                print(f"[skip] no visual_localization: {jf.name}", file=sys.stderr)
            continue
        try:
            pack = load_segment_data(Path(seg_resolved))
        except Exception as e:
            if not skip_missing_segment_dir:
                print(f"[skip] load_segment_data failed {jf.name}: {e}", file=sys.stderr)
            continue
        sdf = pack["segment_data"]
        try:
            y_true = label_vector(sdf)
        except Exception as e:
            print(f"[skip] Label column issue {jf.name}: {e}", file=sys.stderr)
            continue
        pred_set = intervals_to_set(intervals)
        y_pred = pred_vector_from_intervals(sdf, pred_set)
        if len(y_true) != len(y_pred):
            print(f"[skip] length mismatch {jf.name}", file=sys.stderr)
            continue

        series, seg = parse_series_and_seg(jf.name, seg_resolved)
        pre, rec, f1, rf1, pa, auc_pr = metrics_one(grader, y_true, y_pred)
        per_rows.append(
            (jf.name, pre, rec, f1, rf1, pa, auc_pr, int(y_true.sum()))
        )
        by_series[series].append((seg, y_true.copy(), y_pred.copy(), jf.name))

    if not per_rows:
        print(
            f"No samples loaded successfully (pred_source={pred_source}).",
            file=sys.stderr,
        )
        if json_files:
            print(
                "Hint: directory has *.json but none could be evaluated."
                "If many [skip] load_segment_data with paths like .../data/processed/KPI_x/...,"
                "but data lives at .../data/KPI/processed/KPI_x/..., add:"
                "--remap-segment-dir /path/data/processed /path/data/KPI/processed",
                file=sys.stderr,
            )
        return 1

    series_metrics: List[
        Tuple[str, int, float, float, float, float, float, float]
    ] = []
    series_pos_counts: Dict[str, int] = {}
    for series in sorted(by_series.keys(), key=series_sort_key):
        blocks = sorted(by_series[series], key=lambda t: t[0])
        if not blocks:
            continue
        ys = [t[1] for t in blocks]
        ps = [t[2] for t in blocks]
        y_s = np.concatenate(ys)
        p_s = np.concatenate(ps)
        npos = int(np.sum(y_s == 1))
        series_pos_counts[series] = npos
        spre, srec, sf1, srf, spa, sap = metrics_one(grader, y_s, p_s)
        series_metrics.append((series, len(blocks), spre, srec, sf1, srf, spa, sap))

    if not series_metrics:
        print("No series aggregation blocks available.", file=sys.stderr)
        return 1

    # Overall aggregation is a mean over series (unweighted by segment count).
    # Optionally exclude series with no GT positives (npos==0).
    if exclude_no_anomaly_series:
        kept = [r for r in series_metrics if series_pos_counts.get(r[0], 0) > 0]
        dropped = len(series_metrics) - len(kept)
        if not kept:
            print(
                "All series have no GT anomaly points (npos==0); cannot compute overall metrics with exclude_no_anomaly_series.",
                file=sys.stderr,
            )
            return 1
        overall_note = f"exclude npos=0 ({dropped} dropped)"
    else:
        kept = list(series_metrics)
        overall_note = ""

    mpre = float(np.mean([r[2] for r in kept]))
    mrec = float(np.mean([r[3] for r in kept]))
    mf1 = float(np.mean([r[4] for r in kept]))
    mrf1 = float(np.mean([r[5] for r in kept]))
    mpa = float(np.mean([r[6] for r in kept]))
    mapr_vals = [r[7] for r in kept]
    mapr_finite = [x for x in mapr_vals if x == x]
    mmapr = float(np.mean(mapr_finite)) if mapr_finite else float("nan")

    rec_note = "recursive " if recursive else ""
    print(
        f"Directory: {res_dir}  |  {rec_note}valid json: {len(per_rows)}  |  pred_source={pred_source!r}"
    )
    if pred_source == "final":
        title = "=== final (anomaly_intervals; F1 ignores confidence) ==="
    else:
        title = "=== localization-only (visual_localization) ==="

    _print_metrics_table(
        title=title,
        overall=(mpre, mrec, mf1, mrf1, mpa, mmapr),
        series_rows=series_metrics,
        overall_note=overall_note,
    )

    if per_series:
        print("--- per series (brief) ---")
        for s, n, pre, rec, f1, rf1, pa, auc_pr in series_metrics:
            ap_s = f"{auc_pr:.4f}" if auc_pr == auc_pr else "nan"
            print(
                f"  series={s}\tsegments={n}\t"
                f"Pre={pre:.4f}\tRec={rec:.4f}\tF1={f1:.4f}\t"
                f"RANGE-F1={rf1:.4f}\tPA-F1={pa:.4f}\tAUC-PR={ap_s}"
            )

    if per_file:
        print("--- per file ---")
        for name, pre, rec, f1, rf1, pa, auc_pr, npos in per_rows:
            ap_s = f"{auc_pr:.4f}" if auc_pr == auc_pr else "nan"
            print(
                f"  {name}\tPre={pre:.4f}\tRec={rec:.4f}\tF1={f1:.4f}\t"
                f"R-F1={rf1:.4f}\tPA-F1={pa:.4f}\tAUC-PR={ap_s}\tGT_pos={npos}"
            )

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    res_dir: Path = args.res_dir.resolve()
    if not res_dir.is_dir():
        print(f"Directory not found: {res_dir}", file=sys.stderr)
        return 2

    print("[config]")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")

    basic_metricor = import_basic_metricor()
    grader = basic_metricor()

    remap: Optional[Tuple[str, str]] = None
    if args.remap_segment_dir is not None:
        remap = (str(args.remap_segment_dir[0]), str(args.remap_segment_dir[1]))

    sources = ("final", "localization") if args.both_preds else (args.pred_source,)
    rc = 0
    for i, src in enumerate(sources):
        if i:
            print()
        r = _run_eval_for_source(
            res_dir=res_dir,
            pred_source=src,
            grader=grader,
            per_file=args.per_file,
            per_series=args.per_series,
            recursive=args.recursive,
            skip_missing_segment_dir=args.skip_missing_segment_dir,
            exclude_no_anomaly_series=bool(getattr(args, "exclude_no_anomaly_series", False)),
            segment_dir_remap=remap,
        )
        rc = max(rc, r)
        if src == "final" and config.confidence_metrics:
            cr = _run_confidence_eval_for_source(
                res_dir=res_dir,
                grader=grader,
                recursive=args.recursive,
                skip_missing_segment_dir=args.skip_missing_segment_dir,
                exclude_no_anomaly_series=bool(getattr(args, "exclude_no_anomaly_series", False)),
                confidence_threshold=float(config.confidence_threshold),
                segment_dir_remap=remap,
            )
            rc = max(rc, cr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
