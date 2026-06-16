from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage

from AnomaMind.core.utils import get_column_names
from AnomaMind.core.workflowstate import WorkflowState


def parse_detector_answer_text(final_content: str) -> dict[str, Any]:
    empty = {
        "has_error": False,
        "error_message": "",
        "anomaly_intervals": [],
        "anomaly_types": [],
        "explanations": [],
        "anomaly_confidences": [],
    }
    if not final_content or not str(final_content).strip():
        return {
            **empty,
            "has_error": True,
            "error_message": "Detector parse failed: empty response",
        }
    try:
        reasoning_match = re.search(
            r"<(think|redacted_reasoning|redacted_thinking)>(.*?)</\1>",
            final_content,
            re.DOTALL,
        )
        search_start_pos = reasoning_match.end() if reasoning_match else 0
        tail = final_content[search_start_pos:]
        decoder = json.JSONDecoder()
        json_text = None
        for source in (tail, final_content):
            start = source.find("[")
            while start != -1:
                try:
                    parsed_obj, end = decoder.raw_decode(source[start:])
                except json.JSONDecodeError:
                    start = source.find("[", start + 1)
                    continue
                if isinstance(parsed_obj, list):
                    json_text = source[start : start + end]
                    break
                start = source.find("[", start + 1)
            if json_text is not None:
                break
        if json_text is None:
            raise ValueError("no JSON list")
        anomalies_list = json.loads(json_text)
        if not isinstance(anomalies_list, list):
            raise ValueError("JSON is not list type")
        intervals: list[list[int]] = []
        types: list[str] = []
        explanations: list[str] = []
        confidences: list[int] = []
        for item in anomalies_list:
            if not isinstance(item, dict):
                continue
            raw_interval = item.get("interval", [])
            if isinstance(raw_interval, list) and len(raw_interval) >= 1:
                try:
                    start = int(raw_interval[0])
                    end = int(raw_interval[1]) if len(raw_interval) >= 2 else start
                except (ValueError, TypeError, IndexError):
                    continue
                intervals.append([min(start, end), max(start, end)])
                types.append(str(item.get("type", "Unknown")))
                explanations.append(str(item.get("explanation", "")))
                raw_conf = item.get("confidence")
                if raw_conf is None or (isinstance(raw_conf, str) and not str(raw_conf).strip()):
                    confidences.append(2)
                else:
                    try:
                        v = int(raw_conf)
                        confidences.append(2 if v >= 2 else 1)
                    except (TypeError, ValueError):
                        confidences.append(2)
        return {
            "has_error": False,
            "error_message": "",
            "anomaly_intervals": intervals,
            "anomaly_types": types,
            "explanations": explanations,
            "anomaly_confidences": confidences,
        }
    except Exception as exc:
        print(f"[finalize.parse_detector_answer_text] parsing error: {exc}")
        return {
            **empty,
            "has_error": True,
            "error_message": str(exc),
        }


def last_detector_answer_text(messages: list[Any]) -> str:
    if not messages:
        return ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls:
                content = msg.content
                return content if isinstance(content, str) else str(content or "")
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content or "")
    return ""


def extract_anomaly_points(data: Any, intervals: list[Any]) -> list[Any]:
    timestamp_col, value_col = get_column_names(data)
    anomaly_points: list[Any] = []
    for interval in intervals:
        if not isinstance(interval, (list, tuple)) or len(interval) == 0:
            continue
        if len(interval) == 1:
            start = end = interval[0]
        else:
            start, end = interval[0], interval[1]

        if timestamp_col in data.columns:
            start_rows = data[data[timestamp_col] == start]
            end_rows = data[data[timestamp_col] == end]
            if len(start_rows) > 0 and len(end_rows) > 0:
                start_row = start_rows.index[0]
                end_row = end_rows.index[0]
                if start_row > end_row:
                    start_row, end_row = end_row, start_row
                interval_data = data.iloc[start_row : end_row + 1]
                anomaly_points.extend(interval_data[value_col].tolist())
        else:
            if 0 <= start < len(data) and 0 <= end < len(data):
                if start > end:
                    start, end = end, start
                interval_data = data.iloc[start : end + 1]
                anomaly_points.extend(interval_data[value_col].tolist())
    return anomaly_points


def finalize_detector_output(state: WorkflowState) -> WorkflowState:
    empty_out = {
        "error": True,
        "error_message": "",
        "anomaly_intervals": [],
        "anomaly_types": [],
        "explanations": [],
        "anomaly_confidences": [],
        "anomaly_points": [],
    }
    if state.get("has_error"):
        empty_out["error_message"] = str(state.get("error_message", "") or "Unknown error")
        state["final_output"] = empty_out
        return state

    final_text = last_detector_answer_text(list(state.get("detector_messages") or []))
    parsed = parse_detector_answer_text(final_text)
    if parsed.get("has_error"):
        state["final_output"] = {
            "error": True,
            "error_message": str(parsed.get("error_message", "") or "parse failed"),
            "anomaly_intervals": [],
            "anomaly_types": [],
            "explanations": [],
            "anomaly_confidences": [],
            "anomaly_points": [],
        }
        return state

    intervals = list(parsed.get("anomaly_intervals") or [])
    state["final_output"] = {
        "error": False,
        "error_message": "",
        "anomaly_intervals": intervals,
        "anomaly_types": list(parsed.get("anomaly_types") or []),
        "explanations": list(parsed.get("explanations") or []),
        "anomaly_confidences": list(parsed.get("anomaly_confidences") or []),
        "anomaly_points": extract_anomaly_points(state["data"], intervals),
    }
    return state
