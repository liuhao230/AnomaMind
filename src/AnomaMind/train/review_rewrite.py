#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Review low-F1 detector result JSONs with an external LLM and optionally rewrite
the bad ones conservatively.

Workflow:
1. Load result JSONs from a directory.
2. Recover GT from each JSON's segment_dir.
3. Score current detector output against GT with point-level F1.
4. Select poor samples (e.g. low F1 / parse errors).
5. Ask the detector.env model to review whether the response is actually wrong.
6. If the model provides a better answer and it improves F1, save it as a
   candidate rewrite and optionally apply it back to the source JSON.

The review model is instructed to be conservative:
- If evidence is insufficient, keep the original.
- If GT may be wrong, do not force a rewrite.

Example::

  python3 -m AnomaMind.train.review_rewrite \
      --result-dir /path/data/KPI/result/train/8B-in \
      --output-dir /path/data/KPI/rewrite \
      --env-file /path/to/repo/env/detector.env \
      --workers 8 \
      --top-k 100 \
      --max-f1 0.6
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from dotenv import dotenv_values
except Exception:  # pragma: no cover - fallback if python-dotenv is absent
    dotenv_values = None

from AnomaMind.core.finalize import extract_anomaly_points, parse_detector_answer_text
from AnomaMind.core.reward import intervals_to_set
from AnomaMind.core.utils import load_segment_data
from AnomaMind.eval.result_metrics import (
    collect_result_json_files,
    fmt_metric,
    import_basic_metricor,
    label_vector,
    metrics_one,
    parse_series_and_seg,
    pred_vector_from_intervals,
    render_metrics_table,
    series_sort_key,
)


def _default_env_file() -> Path:
    return Path(__file__).resolve().parents[3] / "env" / "detector.env"


DEFAULT_ENV_FILE = _default_env_file()
_THREAD_LOCAL_REVIEW_CLIENT = threading.local()


@dataclass
class SampleMetrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    n_pred_points: int
    n_gt_points: int
    parse_error: bool = False
    parse_error_message: str = ""


@dataclass
class Candidate:
    result_path: str
    file_name: str
    segment_dir: str
    prompt: str
    response: str
    ground_truth: List[List[int]]
    original_intervals: List[List[int]]
    original_metrics: SampleMetrics
    original_final_output: Dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Review low-F1 detector result JSONs and conservatively rewrite them with detector.env."
    )
    p.add_argument(
        "--result-dir",
        "--input-dir",
        dest="result_dir",
        type=Path,
        required=True,
        help="Input result directory containing detector JSONs.",
    )
    p.add_argument("--recursive", action="store_true", help="Recursively collect JSONs from subdirectories.")
    p.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="dotenv file with LLM_API_KEY / LLM_BASE_URL / AUX_LLM_MODEL",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save review outputs. Default: sibling dir with timestamp.",
    )
    p.add_argument("--top-k", type=int, default=50, help="Review at most this many worst samples. 0=all selected.")
    p.add_argument(
        "--max-f1",
        type=float,
        default=0.6,
        help="Select samples whose original point-F1 is <= this threshold.",
    )
    p.add_argument(
        "--include-parse-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Always include samples whose original detector response cannot be parsed.",
    )
    p.add_argument(
        "--min-improvement",
        type=float,
        default=0.0,
        help="Only mark rewrite as replaceable when rewritten F1 improves by at least this margin.",
    )
    p.add_argument(
        "--review-max-tokens",
        type=int,
        default=4096,
        help="Max tokens for the review model completion.",
    )
    p.add_argument("--temperature", type=float, default=0.1, help="Review model temperature.")
    p.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP timeout in seconds for the review API call.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry review API / invalid-JSON response this many times per sample.",
    )
    p.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=2.0,
        help="Sleep between retries.",
    )
    p.add_argument(
        "--apply-rewrite",
        action="store_true",
        help="Apply improved rewrites back into source JSON files.",
    )
    p.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Backup directory used when --apply-rewrite is enabled.",
    )
    p.add_argument(
        "--remap-segment-dir",
        nargs=2,
        metavar=("FROM_PREFIX", "TO_PREFIX"),
        default=None,
        help="Rewrite JSON segment_dir prefix before loading GT.",
    )
    p.add_argument(
        "--point-only",
        action="store_true",
        help="Only process samples whose ground-truth anomalies are all single-point intervals.",
    )
    p.add_argument(
        "--skip-reviewed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip files already present in output_dir/review.jsonl so reruns only process unfinished samples.",
    )
    p.add_argument(
        "--replace-result-dir-from-rewritten",
        action="store_true",
        help=(
            "After review, copy output_dir/rewritten/*.json back into result-dir by file name, "
            "backing up original result-dir files first."
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent review workers. 1=serial.",
    )
    return p


def _load_env(path: Path) -> Dict[str, str]:
    if dotenv_values is not None:
        raw = dotenv_values(path) or {}
        return {str(k): str(v) for k, v in raw.items() if k and v is not None}

    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, value = s.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _make_client(env: Dict[str, str], *, timeout: float):
    try:
        import httpx
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("need packages: openai httpx python-dotenv") from exc

    api_key = env.get("LLM_API_KEY") or env.get("OPENAI_API_KEY")
    base_url = env.get("LLM_BASE_URL") or env.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    model = env.get("AUX_LLM_MODEL") or env.get("LLM_MODEL") or "gpt-4o-mini"
    if not api_key:
        raise SystemExit(f"missing LLM_API_KEY in env file: {DEFAULT_ENV_FILE}")

    http_client = httpx.Client(trust_env=False, timeout=timeout)
    client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"), http_client=http_client)
    return client, model, base_url


def _get_thread_local_review_client(env: Dict[str, str], *, timeout: float):
    cache_key = (
        env.get("LLM_API_KEY") or env.get("OPENAI_API_KEY") or "",
        env.get("LLM_BASE_URL") or env.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
        env.get("AUX_LLM_MODEL") or env.get("LLM_MODEL") or "gpt-4o-mini",
        float(timeout),
    )
    cached = getattr(_THREAD_LOCAL_REVIEW_CLIENT, "cache", None)
    if isinstance(cached, dict) and cached.get("key") == cache_key:
        return cached["value"]
    value = _make_client(env, timeout=timeout)
    _THREAD_LOCAL_REVIEW_CLIENT.cache = {"key": cache_key, "value": value}
    return value


def _iter_result_jsons(result_dir: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*.json" if recursive else "*.json"
    for path in sorted(result_dir.glob(pattern)):
        if not path.is_file():
            continue
        if path.name == "meta_info.json" or path.name.startswith("_"):
            continue
        yield path


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"record is not a JSON object: {path}")
    return obj


def _extract_detector_prompt(record: Dict[str, Any]) -> str:
    detector_messages = record.get("detector_messages") or []
    if isinstance(detector_messages, list):
        for msg in detector_messages:
            if not isinstance(msg, dict):
                continue
            if str(msg.get("role", "")).lower() == "user":
                content = msg.get("content", "")
                return content if isinstance(content, str) else str(content or "")

    messages = record.get("messages") or []
    if isinstance(messages, list):
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if str(msg.get("role", "")).lower() == "user":
                content = msg.get("content", "")
                return content if isinstance(content, str) else str(content or "")
    return ""


def _extract_detector_response(record: Dict[str, Any]) -> str:
    detector_messages = record.get("detector_messages") or []
    if isinstance(detector_messages, list):
        for msg in reversed(detector_messages):
            if not isinstance(msg, dict):
                continue
            if str(msg.get("role", "")).lower() == "assistant":
                content = msg.get("content", "")
                return content if isinstance(content, str) else str(content or "")

    messages = record.get("messages") or []
    if isinstance(messages, list):
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if str(msg.get("role", "")).lower() == "assistant":
                content = msg.get("content", "")
                return content if isinstance(content, str) else str(content or "")
    return ""


def _normalize_intervals(value: Any) -> List[List[int]]:
    out: List[List[int]] = []
    if not isinstance(value, list):
        return out
    for item in value:
        if not isinstance(item, (list, tuple)) or not item:
            continue
        try:
            start = int(item[0])
            end = int(item[1]) if len(item) > 1 else start
        except Exception:
            continue
        out.append([min(start, end), max(start, end)])
    return out


def _metrics_from_intervals(pred: List[List[int]], gt: List[List[int]], *, parse_error: bool = False, parse_error_message: str = "") -> SampleMetrics:
    pred_points = intervals_to_set(pred)
    gt_points = intervals_to_set(gt)
    tp = len(pred_points & gt_points)
    fp = len(pred_points - gt_points)
    fn = len(gt_points - pred_points)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    if not gt_points and not pred_points:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    elif not gt_points and pred_points:
        precision = 0.0
        recall = 0.0
        f1 = 0.0

    return SampleMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        n_pred_points=len(pred_points),
        n_gt_points=len(gt_points),
        parse_error=bool(parse_error),
        parse_error_message=str(parse_error_message or ""),
    )


def _resolve_segment_dir(record: Dict[str, Any], remap: Optional[Tuple[str, str]]) -> Path:
    raw = str(record.get("segment_dir") or "").strip()
    if not raw:
        raise FileNotFoundError("missing segment_dir in result JSON")
    if remap:
        from_prefix, to_prefix = remap
        if raw.startswith(from_prefix):
            raw = to_prefix + raw[len(from_prefix) :]
    return Path(raw).resolve()


def _all_ground_truth_intervals_are_single_points(gt: List[List[int]]) -> bool:
    if not gt:
        return False
    for item in gt:
        if not isinstance(item, list) or len(item) < 2:
            return False
        if int(item[0]) != int(item[1]):
            return False
    return True


def _original_intervals_and_metrics(record: Dict[str, Any], gt: List[List[int]]) -> Tuple[List[List[int]], SampleMetrics, Dict[str, Any]]:
    final_output = record.get("final_output") or {}
    intervals = _normalize_intervals(final_output.get("anomaly_intervals"))
    parse_error = False
    parse_error_message = ""

    if not intervals:
        response_text = _extract_detector_response(record)
        parsed = parse_detector_answer_text(response_text)
        if parsed.get("has_error"):
            parse_error = True
            parse_error_message = str(parsed.get("error_message") or "parse failed")
            intervals = []
        else:
            intervals = _normalize_intervals(parsed.get("anomaly_intervals"))

    metrics = _metrics_from_intervals(
        intervals,
        gt,
        parse_error=parse_error,
        parse_error_message=parse_error_message,
    )
    return intervals, metrics, final_output if isinstance(final_output, dict) else {}


def _select_candidates(
    *,
    result_dir: Path,
    recursive: bool,
    max_f1: float,
    top_k: int,
    include_parse_errors: bool,
    remap_segment_dir: Optional[Tuple[str, str]],
) -> List[Candidate]:
    selected: List[Candidate] = []
    for path in _iter_result_jsons(result_dir, recursive):
        try:
            record = _read_json(path)
            segment_dir = _resolve_segment_dir(record, remap_segment_dir)
            seg = load_segment_data(segment_dir)
            gt = _normalize_intervals(seg.get("ground_truth") or [])
            prompt = _extract_detector_prompt(record)
            response = _extract_detector_response(record)
            intervals, metrics, final_output = _original_intervals_and_metrics(record, gt)
        except Exception as exc:
            print(f"[skip] {path}: {exc}", file=sys.stderr)
            continue

        should_select = metrics.f1 <= max_f1
        if include_parse_errors and metrics.parse_error:
            should_select = True
        if not should_select:
            continue

        selected.append(
            Candidate(
                result_path=str(path.resolve()),
                file_name=path.name,
                segment_dir=str(segment_dir),
                prompt=prompt,
                response=response,
                ground_truth=gt,
                original_intervals=intervals,
                original_metrics=metrics,
                original_final_output=final_output,
            )
        )

    selected.sort(key=lambda x: (0 if x.original_metrics.parse_error else 1, x.original_metrics.f1, x.file_name))
    if top_k > 0:
        selected = selected[:top_k]
    return selected


def _load_reviewed_file_names(path: Path) -> set[str]:
    reviewed: set[str] = set()
    if not path.is_file():
        return reviewed
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("file_name") or "").strip()
        if name:
            reviewed.add(name)
    return reviewed


def _load_completed_file_names(output_dir: Path) -> set[str]:
    completed = set(_load_reviewed_file_names(output_dir / "review.jsonl"))
    rewritten_dir = output_dir / "rewritten"
    if rewritten_dir.is_dir():
        for path in rewritten_dir.glob("*.json"):
            if path.is_file():
                completed.add(path.name)
    return completed


def _load_review_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _merge_review_rows(existing_rows: Sequence[Dict[str, Any]], new_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    by_name: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for row in list(existing_rows) + list(new_rows):
        name = str(row.get("file_name") or "").strip()
        if not name:
            continue
        if name not in by_name:
            order.append(name)
        by_name[name] = row

    for name in order:
        merged.append(by_name[name])
    return merged


def _print_effective_eval_summary(
    *,
    result_dir: Path,
    recursive: bool,
    review_rows: Sequence[Dict[str, Any]],
    remap_segment_dir: Optional[Tuple[str, str]],
) -> Dict[str, Any]:
    review_by_name = {str(r.get("file_name") or "").strip(): r for r in review_rows if str(r.get("file_name") or "").strip()}
    basic_metricor = import_basic_metricor()
    grader = basic_metricor()
    json_files = collect_result_json_files(result_dir, recursive=recursive)

    from collections import defaultdict
    import numpy as np

    by_series = defaultdict(list)
    used_rewrites = 0
    per_rows = 0

    for jf in json_files:
        try:
            record = _read_json(jf)
            segment_dir = _resolve_segment_dir(record, remap_segment_dir)
            pack = load_segment_data(segment_dir)
        except Exception as exc:
            print(f"[effective-eval.skip] {jf.name}: {exc}", file=sys.stderr)
            continue

        row = review_by_name.get(jf.name)
        if row and bool(row.get("should_replace")) and isinstance(row.get("rewritten_intervals"), list):
            intervals = _normalize_intervals(row.get("rewritten_intervals"))
            if intervals:
                used_rewrites += 1
            else:
                intervals = _normalize_intervals((record.get("final_output") or {}).get("anomaly_intervals"))
        else:
            intervals = _normalize_intervals((record.get("final_output") or {}).get("anomaly_intervals"))
            if not intervals:
                parsed = parse_detector_answer_text(_extract_detector_response(record))
                if not parsed.get("has_error"):
                    intervals = _normalize_intervals(parsed.get("anomaly_intervals"))

        sdf = pack["segment_data"]
        try:
            y_true = label_vector(sdf)
        except Exception as exc:
            print(f"[effective-eval.skip] {jf.name}: {exc}", file=sys.stderr)
            continue
        y_pred = pred_vector_from_intervals(sdf, intervals_to_set(intervals))
        series, seg = parse_series_and_seg(jf.name, str(segment_dir))
        by_series[series].append((seg, y_true.copy(), y_pred.copy(), jf.name))
        per_rows += 1

    if not by_series:
        print("[effective-eval] no usable series", file=sys.stderr)
        return {"usable_files": 0, "used_rewrites": used_rewrites}

    series_metrics: List[Tuple[str, int, float, float, float, float, float, float]] = []
    series_pos_counts: Dict[str, int] = {}
    for series in sorted(by_series.keys(), key=series_sort_key):
        blocks = sorted(by_series[series], key=lambda t: t[0])
        ys = [t[1] for t in blocks]
        ps = [t[2] for t in blocks]
        y_s = np.concatenate(ys)
        p_s = np.concatenate(ps)
        npos = int(np.sum(y_s == 1))
        series_pos_counts[series] = npos
        series_metrics.append((series, len(blocks), *metrics_one(grader, y_s, p_s)))

    kept = [r for r in series_metrics if series_pos_counts.get(r[0], 0) > 0]
    dropped = len(series_metrics) - len(kept)
    if not kept:
        print("[effective-eval] all series have npos=0 after filtering", file=sys.stderr)
        return {"usable_files": per_rows, "used_rewrites": used_rewrites}

    import numpy as np

    overall = (
        float(np.mean([r[2] for r in kept])),
        float(np.mean([r[3] for r in kept])),
        float(np.mean([r[4] for r in kept])),
        float(np.mean([r[5] for r in kept])),
        float(np.mean([r[6] for r in kept])),
        float(np.mean([r[7] for r in kept if r[7] == r[7]])) if any(r[7] == r[7] for r in kept) else float("nan"),
    )

    print()
    print(f"Directory: {result_dir}  |  valid json: {per_rows}  |  effective pred='final-with-rewrites'")
    print(
        render_metrics_table(
            title="=== effective final（replaceable rewrites applied virtually）===",
            overall=overall,
            series_rows=series_metrics,
            overall_note=f"exclude npos=0 ({dropped} dropped)",
        )
    )
    return {
        "usable_files": per_rows,
        "used_rewrites": used_rewrites,
        "overall": {
            "pre": overall[0],
            "rec": overall[1],
            "f1": overall[2],
            "range_f1": overall[3],
            "pa_f1": overall[4],
            "auc_pr": overall[5],
        },
        "series_count": len(series_metrics),
        "dropped_npos0_series": dropped,
    }


def _json_object_from_text(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        fenced = fence_match.group(1).strip()
        try:
            obj = json.loads(fenced)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(obj, dict):
                return obj

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found in model response")


def _normalize_review_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("review payload must be a JSON object")

    decision = str(payload.get("decision") or "").strip()
    reason = str(payload.get("reason") or payload.get("reasoning") or "").strip()
    revised_response = str(payload.get("candidate_response") or payload.get("revised_response") or "").strip()
    issues_raw = payload.get("issues")
    issues = [str(x).strip() for x in issues_raw if str(x).strip()] if isinstance(issues_raw, list) else []

    normalized = dict(payload)
    normalized["decision"] = decision
    normalized["reason"] = reason
    normalized["issues"] = issues
    normalized["revised_response"] = revised_response
    return normalized


def _png_to_data_url(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _write_review_gt_pred_png(
    plot_path: Path,
    seg_name: str,
    pack: Dict[str, Any],
    *,
    gt_intervals: List[List[int]],
    pred_intervals: List[List[int]],
    pred_metrics: Dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from AnomaMind.core.utils import get_column_names

    df = pack["segment_data"]
    tcol, vcol = get_column_names(df)
    xs = df[tcol].astype(float).to_numpy()
    ys = df[vcol].astype(float).to_numpy()
    dpi = 165
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.8), dpi=dpi, sharex=True)

    panels = [
        (axes[0], pred_intervals, "tab:red", "Prediction", pred_metrics),
        (axes[1], gt_intervals, "tab:green", "Ground Truth", None),
    ]
    for ax, intervals, shade_color, title, metrics in panels:
        ax.plot(xs, ys, color="0.2", linewidth=0.85)
        for item in intervals:
            if not isinstance(item, (list, tuple)) or not item:
                continue
            try:
                aa = int(item[0])
                bb = int(item[1]) if len(item) > 1 else aa
            except (TypeError, ValueError):
                continue
            if aa > bb:
                aa, bb = bb, aa
            ax.axvspan(float(aa), float(bb), color=shade_color, alpha=0.28)

        panel_title = title
        if isinstance(metrics, dict):
            panel_title = (
                f"{title}  TP={int(metrics.get('tp', 0))} "
                f"FP={int(metrics.get('fp', 0))} FN={int(metrics.get('fn', 0))}"
            )
        ax.set_title(panel_title)
        ax.set_ylabel(str(vcol))
        ax.grid(True, alpha=0.25)

    axes[1].set_xlabel("global index")
    fig.suptitle(seg_name)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, bbox_inches="tight", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)

KPI='''
Avoid false positives caused by low thresholds, while also avoiding misses caused by relying only on high-threshold tools.
For single-point anomalies, a relatively high value_zscore is required; lower values may just be noise. For plateau-type anomalies, a lower value_zscore can still be acceptable, but the boundaries must be supported by significant diff_zscore signals.
Many reported anomaly intervals are incomplete. You should include more points until the series has clearly returned to baseline, rather than stopping merely when the change becomes less significant.

'''

YAHOO='''
YAHOO-specific rules:
- For YAHOO, primary supervision for point-level localization is whether the selected point is the true anomalous extremum rather than a nearby recovery point.
- If nearby peaks/dips occur on a smooth baseline, output the true anomalous points separately, not one merged segment.
- Be conservative about adding neighbors to a one-point anomaly unless the prompt evidence clearly supports a short anomalous span.
'''

TODS='''
- For spike/dip anomalies, keep predictions at point level; do not include neighboring points unless they are clearly anomalous themselves.
- If nearby peaks/dips occur on a smooth baseline, output the true anomalous points separately, not one merged segment.
- If the current answer selected a recovery point, adjacent extremum, or off-by-one neighbor, correct it to the GT-aligned point.
- You need to clarify which is anomaly, which is recovery(thought have large diff but near to normal neighbor). There might be trend anomaly, 
- clean those false postive point with low threshold.
'''

def _build_review_messages(
    candidate: Candidate,
    *,
    extra_image_paths: Optional[Sequence[Path]] = None,
) -> List[Dict[str, str]]:
    system_prompt = (
        "You are generating a time-series anomaly detector result"
        "When you generate a replacement, it must read like a natural first-pass detector answer written directly "
        "from the prompt evidence, not like a review or second look. "
        "Stay faithful to the detector prompt, tool evidence, and domain knowledge. "
    )
    user_prompt = f"""
MOST IMPORTANT:
*****
If you think ground truth is apperantly wrong, just keep original, if there are evidence(localization, zscore) to support gt, rewrite.
*****

Return exactly one JSON object, no markdown fences, with fields:
- "reasoning": your analysis, about whether need to rewrite(no localization/zscore evidence, do not add anomaly), and how to improve the answer.
- "decision": one of "keep_original", "rewrite"
- "candidate_response": full candidate detector answer string, or "" if not rewriting

Rules for "candidate_response":
- It must be a full detector answer in this exact style: <think>...</think>[...]
- Immediately after </think>, output exactly one JSON array.
- If there is no anomaly, use [].
- It must sound like a first-pass detector answer, not a correction or second-pass review.

Specific rules:
{YAHOO}

Ground truth intervals: (Do not copy it, use evidence from tool result to judge, GT may wrong!)
{json.dumps(candidate.ground_truth, ensure_ascii=False)}


Original final_output:
{json.dumps(candidate.original_final_output, ensure_ascii=False, indent=2)}

Detector prompt:
<<<DETECTOR_PROMPT
{candidate.prompt}
DETECTOR_PROMPT

Original detector response:
<<<ORIGINAL_RESPONSE
{candidate.response}
ORIGINAL_RESPONSE
""".strip()
    image_paths: List[Path] = []
    for path in extra_image_paths or []:
        p = Path(path)
        if p.is_file():
            image_paths.append(p)
    if not image_paths:
        segment_image_path = Path(candidate.segment_dir) / "segment_final.png"
        if segment_image_path.is_file():
            image_paths.append(segment_image_path)

    user_content: Any
    if image_paths:
        image_labels = "\n".join(f"- Attached image: {p.name}" for p in image_paths)
        user_content = [{"type": "text", "text": user_prompt + "\n\n" + image_labels}]
        for image_path in image_paths:
            user_content.append({"type": "image_url", "image_url": {"url": _png_to_data_url(image_path)}})
    else:
        user_content = user_prompt
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _call_review_model(
    client: Any,
    model: str,
    candidate: Candidate,
    *,
    extra_image_paths: Optional[Sequence[Path]] = None,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    messages = _build_review_messages(candidate, extra_image_paths=extra_image_paths)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("review model returned empty content")
    parsed = _normalize_review_payload(_json_object_from_text(content))
    return content, parsed


def _call_review_model_with_retry(
    client: Any,
    model: str,
    candidate: Candidate,
    *,
    extra_image_paths: Optional[Sequence[Path]] = None,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_sleep_seconds: float,
) -> Tuple[str, Dict[str, Any], int]:
    last_exc: Exception | None = None
    attempts = max(1, int(max_retries))
    valid_decisions = {"keep_original", "rewrite", "evidence_insufficient", "gt_maybe_wrong"}

    for attempt in range(1, attempts + 1):
        try:
            content, parsed = _call_review_model(
                client,
                model,
                candidate,
                extra_image_paths=extra_image_paths,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            decision = str(parsed.get("decision") or "").strip()
            if decision not in valid_decisions:
                raise ValueError(f"invalid decision: {decision!r}")
            return content, parsed, attempt
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            print(
                f"[review.retry] attempt={attempt}/{attempts} file={candidate.file_name} error={exc}",
                file=sys.stderr,
            )
            time.sleep(max(0.0, float(retry_sleep_seconds)))

    assert last_exc is not None
    raise last_exc


def _materialize_final_output(segment_dir: Path, revised_response: str) -> Tuple[Dict[str, Any], List[List[int]], SampleMetrics]:
    seg = load_segment_data(segment_dir)
    segment_df = seg["segment_data"]
    gt = _normalize_intervals(seg.get("ground_truth") or [])

    parsed = parse_detector_answer_text(revised_response)
    if parsed.get("has_error"):
        metrics = _metrics_from_intervals([], gt, parse_error=True, parse_error_message=str(parsed.get("error_message") or "parse failed"))
        final_output = {
            "error": True,
            "error_message": str(parsed.get("error_message") or "parse failed"),
            "anomaly_intervals": [],
            "anomaly_types": [],
            "explanations": [],
            "anomaly_confidences": [],
            "anomaly_points": [],
        }
        return final_output, [], metrics

    intervals = _normalize_intervals(parsed.get("anomaly_intervals"))
    final_output = {
        "error": False,
        "error_message": "",
        "anomaly_intervals": intervals,
        "anomaly_types": list(parsed.get("anomaly_types") or []),
        "explanations": list(parsed.get("explanations") or []),
        "anomaly_confidences": list(parsed.get("anomaly_confidences") or []),
        "anomaly_points": extract_anomaly_points(segment_df, intervals),
    }
    metrics = _metrics_from_intervals(intervals, gt)
    return final_output, intervals, metrics


def _set_last_assistant_content(messages: Any, new_content: str) -> bool:
    if not isinstance(messages, list):
        return False
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role", "")).lower() == "assistant":
            msg["content"] = new_content
            return True
    return False


def _apply_rewrite_to_record(record: Dict[str, Any], revised_response: str, final_output: Dict[str, Any]) -> Dict[str, Any]:
    _set_last_assistant_content(record.get("detector_messages"), revised_response)
    _set_last_assistant_content(record.get("messages"), revised_response)
    record["final_output"] = final_output
    return record


def _default_output_dir(result_dir: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return result_dir.parent / f"{result_dir.name}_review_rewrite_{stamp}"


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _persist_review_progress(
    review_jsonl_path: Path,
    *,
    existing_rows: Sequence[Dict[str, Any]],
    new_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    all_rows = _merge_review_rows(existing_rows, new_rows)
    _write_jsonl(review_jsonl_path, all_rows)
    return all_rows


def _run_candidates_with_workers(
    candidates: Sequence[Candidate],
    worker_fn,
    *,
    workers: int,
):
    max_workers = max(1, int(workers))
    if max_workers == 1 or len(candidates) <= 1:
        for candidate in candidates:
            yield worker_fn(candidate)
        return

    max_workers = min(max_workers, len(candidates))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_fn, candidate) for candidate in candidates]
        for future in concurrent.futures.as_completed(futures):
            yield future.result()


def _review_single_candidate(
    candidate: Candidate,
    *,
    env: Dict[str, str],
    timeout: float,
    model: str,
    temperature: float,
    review_max_tokens: int,
    max_retries: int,
    retry_sleep_seconds: float,
    min_improvement: float,
    plots_dir: Path,
) -> Dict[str, Any]:
    client, _model, _base_url = _get_thread_local_review_client(env, timeout=timeout)

    plot_path = plots_dir / f"{Path(candidate.file_name).stem}_review_gt_pred.png"
    review_image_paths: List[Path] = []
    try:
        seg = load_segment_data(Path(candidate.segment_dir))
        _write_review_gt_pred_png(
            plot_path,
            Path(candidate.file_name).stem,
            seg,
            gt_intervals=candidate.ground_truth,
            pred_intervals=candidate.original_intervals,
            pred_metrics=asdict(candidate.original_metrics),
        )
        if plot_path.is_file():
            review_image_paths.append(plot_path)
    except Exception as exc:
        print(f"[review.plot] file={candidate.file_name} error={exc}", file=sys.stderr)
        plot_path = Path("")

    try:
        raw_model_text, review, review_attempts = _call_review_model_with_retry(
            client,
            model,
            candidate,
            extra_image_paths=review_image_paths,
            temperature=temperature,
            max_tokens=review_max_tokens,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
    except Exception as exc:
        row = {
            "file_name": candidate.file_name,
            "result_path": candidate.result_path,
            "segment_dir": candidate.segment_dir,
            "original_metrics": asdict(candidate.original_metrics),
            "review_error": str(exc),
            "review_attempts": max_retries,
        }
        return {
            "candidate": candidate,
            "row": row,
            "review_error": True,
            "rewrite_attempted": False,
            "replaceable": False,
            "applied_record": None,
        }

    decision = str(review.get("decision") or "").strip()
    reason = str(review.get("reason") or "").strip()
    issues = review.get("issues") if isinstance(review.get("issues"), list) else []
    revised_response = str(review.get("revised_response") or "")

    rewritten_final_output: Dict[str, Any] | None = None
    rewritten_intervals: List[List[int]] = []
    rewritten_metrics: SampleMetrics | None = None
    should_replace = False
    rewritten_record: Dict[str, Any] | None = None

    if decision == "rewrite" and revised_response.strip():
        rewritten_final_output, rewritten_intervals, rewritten_metrics = _materialize_final_output(
            Path(candidate.segment_dir),
            revised_response,
        )
        improvement = rewritten_metrics.f1 - candidate.original_metrics.f1
        should_replace = bool(
            not rewritten_metrics.parse_error
            and improvement >= float(min_improvement)
            and rewritten_metrics.f1 > candidate.original_metrics.f1
        )
        if should_replace:
            rewritten_record = _apply_rewrite_to_record(
                _read_json(Path(candidate.result_path)),
                revised_response,
                rewritten_final_output,
            )

    row: Dict[str, Any] = {
        "file_name": candidate.file_name,
        "result_path": candidate.result_path,
        "segment_dir": candidate.segment_dir,
        "decision": decision,
        "reason": reason,
        "issues": issues,
        "should_replace": should_replace,
        "original_metrics": asdict(candidate.original_metrics),
        "original_intervals": candidate.original_intervals,
        "ground_truth": candidate.ground_truth,
        "raw_model_text": raw_model_text,
        "review_json": review,
        "review_attempts": review_attempts,
    }
    if str(plot_path):
        row["review_plot_path"] = str(plot_path.resolve())
    if rewritten_metrics is not None:
        row["rewritten_metrics"] = asdict(rewritten_metrics)
        row["rewritten_intervals"] = rewritten_intervals
        row["rewritten_response"] = revised_response

    return {
        "candidate": candidate,
        "row": row,
        "review_error": False,
        "rewrite_attempted": bool(decision == "rewrite" and revised_response.strip()),
        "replaceable": should_replace,
        "applied_record": rewritten_record,
    }


def _replace_result_dir_from_rewritten(
    *,
    result_dir: Path,
    rewritten_dir: Path,
    backup_dir: Path,
) -> int:
    if not rewritten_dir.is_dir():
        return 0

    applied = 0
    replace_backup_dir = backup_dir / "replace_result_dir"
    for rewritten_path in sorted(rewritten_dir.glob("*.json")):
        if not rewritten_path.is_file():
            continue
        target_path = result_dir / rewritten_path.name
        if not target_path.is_file():
            continue
        backup_path = replace_backup_dir / rewritten_path.name
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target_path, backup_path)
        shutil.copy2(rewritten_path, target_path)
        applied += 1
    return applied


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result_dir = args.result_dir.resolve()
    output_dir = (args.output_dir.resolve() if args.output_dir else _default_output_dir(result_dir))
    rewritten_dir = output_dir / "rewritten"
    backup_dir = (args.backup_dir.resolve() if args.backup_dir else output_dir / "backups")

    env = _load_env(args.env_file.resolve())
    client, model, base_url = _make_client(env, timeout=float(args.timeout))

    remap: Optional[Tuple[str, str]] = None
    if args.remap_segment_dir:
        remap = (str(args.remap_segment_dir[0]), str(args.remap_segment_dir[1]))

    candidates = _select_candidates(
        result_dir=result_dir,
        recursive=bool(args.recursive),
        max_f1=float(args.max_f1),
        top_k=int(args.top_k),
        include_parse_errors=bool(args.include_parse_errors),
        remap_segment_dir=remap,
    )
    if bool(args.point_only):
        candidates = [c for c in candidates if _all_ground_truth_intervals_are_single_points(c.ground_truth)]

    reviewed_file_names: set[str] = set()
    existing_review_rows: List[Dict[str, Any]] = []
    if bool(args.skip_reviewed):
        existing_review_rows = _load_review_rows(output_dir / "review.jsonl")
        reviewed_file_names = _load_completed_file_names(output_dir)
        if reviewed_file_names:
            candidates = [c for c in candidates if c.file_name not in reviewed_file_names]

    rows: List[Dict[str, Any]] = []
    applied = 0
    rewrite_count = 0
    replaceable_count = 0
    review_error_count = 0
    plots_dir = output_dir / "plots"

    if candidates:
        print(
            f"[review] selected={len(candidates)} result_dir={result_dir} model={model} base_url={base_url} workers={max(1, int(args.workers))}",
            file=sys.stderr,
        )

        def _worker(one_candidate: Candidate) -> Dict[str, Any]:
            return _review_single_candidate(
                one_candidate,
                env=env,
                timeout=float(args.timeout),
                model=model,
                temperature=float(args.temperature),
                review_max_tokens=int(args.review_max_tokens),
                max_retries=int(args.max_retries),
                retry_sleep_seconds=float(args.retry_sleep_seconds),
                min_improvement=float(args.min_improvement),
                plots_dir=plots_dir,
            )

        for idx, result in enumerate(
            _run_candidates_with_workers(candidates, _worker, workers=int(args.workers)),
            start=1,
        ):
            candidate = result["candidate"]
            print(
                f"[review] {idx}/{len(candidates)} f1={candidate.original_metrics.f1:.6f} parse_error={candidate.original_metrics.parse_error} file={candidate.file_name}",
                file=sys.stderr,
            )

            row = result["row"]
            if bool(result["review_error"]):
                review_error_count += 1
                rows.append(row)
                _persist_review_progress(
                    output_dir / "review.jsonl",
                    existing_rows=existing_review_rows,
                    new_rows=rows,
                )
                continue

            if bool(result["rewrite_attempted"]):
                rewrite_count += 1
            if bool(result["replaceable"]):
                replaceable_count += 1
                rewritten_record = result["applied_record"]
                if rewritten_record is not None:
                    _write_json(rewritten_dir / candidate.file_name, rewritten_record)

                    if args.apply_rewrite:
                        src = Path(candidate.result_path)
                        backup_path = backup_dir / candidate.file_name
                        backup_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, backup_path)
                        _write_json(src, rewritten_record)
                        applied += 1

            rows.append(row)
            _persist_review_progress(
                output_dir / "review.jsonl",
                existing_rows=existing_review_rows,
                new_rows=rows,
            )
    else:
        print("no candidates selected", file=sys.stderr)

    summary = {
        "result_dir": str(result_dir),
        "output_dir": str(output_dir),
        "env_file": str(args.env_file.resolve()),
        "model": model,
        "selected_candidates": len(candidates),
        "rewrite_count": rewrite_count,
        "replaceable_count": replaceable_count,
        "applied_count": applied,
        "review_error_count": review_error_count,
        "top_k": int(args.top_k),
        "max_f1": float(args.max_f1),
        "min_improvement": float(args.min_improvement),
        "max_retries": int(args.max_retries),
        "retry_sleep_seconds": float(args.retry_sleep_seconds),
        "apply_rewrite": bool(args.apply_rewrite),
        "point_only": bool(args.point_only),
        "skip_reviewed": bool(args.skip_reviewed),
        "already_reviewed_count": len(reviewed_file_names),
        "workers": max(1, int(args.workers)),
    }

    all_rows = _persist_review_progress(
        output_dir / "review.jsonl",
        existing_rows=existing_review_rows,
        new_rows=rows,
    )
    replaced_from_rewritten_count = 0
    if bool(args.replace_result_dir_from_rewritten):
        replaced_from_rewritten_count = _replace_result_dir_from_rewritten(
            result_dir=result_dir,
            rewritten_dir=rewritten_dir,
            backup_dir=backup_dir,
        )
    summary["replaced_from_rewritten_count"] = replaced_from_rewritten_count
    effective_eval = _print_effective_eval_summary(
        result_dir=result_dir,
        recursive=bool(args.recursive),
        review_rows=all_rows,
        remap_segment_dir=remap,
    )
    summary["effective_eval_after_rewrite"] = effective_eval
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
