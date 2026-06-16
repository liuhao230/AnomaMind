from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple


def intervals_to_set(intervals: List[List[int]]) -> Set[int]:
    point_set: set[int] = set()
    for interval in intervals:
        if not isinstance(interval, (list, tuple)) or len(interval) == 0:
            continue
        if len(interval) == 1:
            point_set.add(int(interval[0]))
            continue
        start = int(interval[0])
        end = int(interval[1])
        if start > end:
            start, end = end, start
        point_set.update(range(start, end + 1))
    return point_set


def f1_reward(anomaly_intervals: List[List[int]], ground_truth: List[List[int]]) -> float:
    if not isinstance(anomaly_intervals, list):
        anomaly_intervals = []
    if not isinstance(ground_truth, list):
        ground_truth = []

    predicted_points = intervals_to_set(anomaly_intervals)
    true_points = intervals_to_set(ground_truth)

    if len(true_points) == 0:
        if len(predicted_points) == 0:
            return 1.0
        return float(1.0 / (len(predicted_points) + 1.0))

    tp = len(predicted_points & true_points)
    fp = len(predicted_points - true_points)
    fn = len(true_points - predicted_points)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * (precision * recall) / (precision + recall))


def reward(anomaly_intervals: List[List[int]], ground_truth: List[List[int]]) -> float:
    """Backward-compatible alias for point-level F1 reward."""
    return f1_reward(anomaly_intervals, ground_truth)


def _safe_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_strict_post_think_json_list(text: str) -> Tuple[Optional[str], str]:
    if not isinstance(text, str) or not text.strip():
        return None, "empty_output"

    m = re.search(
        r"<(think|redacted_reasoning|redacted_thinking)>(.*?)</\1>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return None, "missing_think_block"

    tail = text[m.end() :]
    if not tail.strip():
        return None, "missing_json_after_think"

    stripped = tail.strip()
    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(stripped)
    except Exception:
        return None, "invalid_json_after_think"

    if not isinstance(obj, list):
        return None, "json_after_think_not_list"

    trailing = stripped[end:].strip()
    if trailing:
        return None, "trailing_text_after_json"

    return stripped[:end], "ok"


def _parse_predicted_intervals(solution_str: str) -> Tuple[Optional[List[List[int]]], str]:
    json_list_str, parse_status = _extract_strict_post_think_json_list(solution_str)
    if not json_list_str:
        return None, parse_status
    obj = _safe_json_loads(json_list_str)
    if not isinstance(obj, list):
        return None, "decoded_json_not_list"

    intervals: List[List[int]] = []
    for item in obj:
        if not isinstance(item, dict):
            continue
        raw = item.get("interval", [])
        if not isinstance(raw, list) or len(raw) == 0:
            continue
        try:
            a = int(raw[0])
            b = int(raw[1]) if len(raw) >= 2 else a
            if a > b:
                a, b = b, a
            intervals.append([a, b])
        except Exception:
            continue
    return intervals, "ok"


def _parse_ground_truth(ground_truth: Any) -> List[List[int]]:
    if isinstance(ground_truth, list):
        return ground_truth
    if isinstance(ground_truth, str):
        gt = _safe_json_loads(ground_truth)
        if isinstance(gt, list):
            return gt
    return []


def _maybe_log_one(
    data_source: str,
    solution_str: str,
    gt_intervals: List[List[int]],
    pred_intervals: Optional[List[List[int]]],
    score: float,
    extra_info: Optional[Dict[str, Any]],
    *,
    parse_status: str = "unknown",
) -> None:
    log_dir = os.getenv("VERL_REWARD_LOG_DIR", "").strip()
    if not log_dir:
        return

    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        return

    pid = os.getpid()
    path = os.path.join(log_dir, f"reward-trace-{pid}.jsonl")

    max_chars = int(os.getenv("VERL_REWARD_LOG_MAX_CHARS", "8000"))
    sol = solution_str if isinstance(solution_str, str) else str(solution_str)
    if max_chars > 0 and len(sol) > max_chars:
        sol = sol[:max_chars] + "\n...[truncated]..."

    uid = None
    cache_json_path = None
    segment_folder = None
    prompt_str = None
    response_str = None
    if isinstance(extra_info, dict):
        uid = extra_info.get("uid")
        cache_json_path = extra_info.get("cache_json_path")
        segment_folder = extra_info.get("segment_folder")
        prompt_str = extra_info.get("prompt_str")
        response_str = extra_info.get("response_str")

    if isinstance(prompt_str, str) and max_chars > 0 and len(prompt_str) > max_chars:
        prompt_str = prompt_str[:max_chars] + "\n...[truncated]..."
    if isinstance(response_str, str) and max_chars > 0 and len(response_str) > max_chars:
        response_str = response_str[:max_chars] + "\n...[truncated]..."

    rec = {
        "ts": time.time(),
        "pid": pid,
        "data_source": data_source,
        "uid": uid,
        "cache_json_path": cache_json_path,
        "segment_folder": segment_folder,
        "score": float(score),
        "parse_status": parse_status,
        "ground_truth": gt_intervals,
        "pred_intervals": pred_intervals,
        "prompt_str": prompt_str,
        "response_str": response_str,
        "solution_str": sol,
    }

    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        return


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """VERL custom reward entrypoint."""
    del kwargs
    gt_intervals = _parse_ground_truth(ground_truth)
    pred_intervals, parse_status = _parse_predicted_intervals(solution_str)

    if parse_status == "trailing_text_after_json":
        score_out = {
            "score": 0.0,
            "parse_status": parse_status,
            "format_ok": False,
        }
        _maybe_log_one(
            data_source,
            solution_str,
            gt_intervals,
            pred_intervals,
            score_out["score"],
            extra_info,
            parse_status=parse_status,
        )
        return score_out

    if pred_intervals is None:
        score_out = {
            "score": -1.0,
            "parse_status": parse_status,
            "format_ok": False,
        }
        _maybe_log_one(
            data_source,
            solution_str,
            gt_intervals,
            pred_intervals,
            score_out["score"],
            extra_info,
            parse_status=parse_status,
        )
        return score_out

    score = float(f1_reward(pred_intervals, gt_intervals))
    score_out = {
        "score": score,
        "parse_status": parse_status,
        "format_ok": True,
    }
    _maybe_log_one(
        data_source,
        solution_str,
        gt_intervals,
        pred_intervals,
        score,
        extra_info,
        parse_status=parse_status,
    )
    return score_out
