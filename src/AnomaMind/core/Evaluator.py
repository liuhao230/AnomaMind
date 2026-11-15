# -*- coding: utf-8 -*-
"""Evaluator: compress Actor/Detector context, then decide whether to redo the Actor."""

from __future__ import annotations

import json
import re
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from AnomaMind.core.detector_prompt import build_actor_replay_text
from AnomaMind.core.workflowstate import WorkflowState

EVALUATOR_ROLE = (
    "You are the **Evaluator** stage. You receive short summaries of the Detector "
    "answer, Actor tool transcript, and any prior retry suggestion.\n"
    "Set ``redo`` to true **only for particularly serious errors**, for example:\n"
    "- clearly wrong anomaly intervals when evidence is strong;\n"
    "- an obvious anomaly was missed;\n"
    "- Actor tool usage contradicts the evidence or ignores critical intervals.\n"
    "Do **not** redo for minor wording issues, small format nits, borderline judgment "
    "calls, or low-confidence disagreements. When in doubt, prefer ``redo: false``."
).strip()

EVALUATOR_OUTPUT = """
## Output format
Reply with a single JSON object only (no markdown fences), for example:
{"redo": false, "suggestion": ""}
- ``redo``: true only for particularly serious errors (see above).
- ``suggestion``: short, actionable guidance for the Actor when ``redo`` is true; otherwise "".
""".strip()

_COMPRESS_INSTRUCTIONS = {
    "detector": (
        "Summarize the following Detector final answer in **one short plain sentence**. "
        "Keep anomaly intervals and the main conclusion. Reply with only that sentence."
    ),
    "actor": (
        "Summarize the following Actor tool transcript in **one short plain sentence**. "
        "Keep which tools ran and the main evidence finding. Reply with only that sentence."
    ),
    "suggestion": (
        "Summarize the following prior retry suggestion in **one short plain sentence**. "
        "Reply with only that sentence."
    ),
}


def _last_ai_text(messages: list[Any]) -> str:
    for msg in reversed(messages or []):
        if isinstance(msg, AIMessage):
            return str(getattr(msg, "content", "") or "")
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return str(msg.get("content") or "")
    return ""


def _detector_source_text(messages: list[Any]) -> str:
    text = _last_ai_text(messages)
    if text.strip():
        return text.strip()
    parts: list[str] = []
    for msg in messages or []:
        if isinstance(msg, HumanMessage):
            parts.append(str(getattr(msg, "content", "") or ""))
        elif isinstance(msg, AIMessage):
            parts.append(str(getattr(msg, "content", "") or ""))
        elif isinstance(msg, dict):
            parts.append(str(msg.get("content") or ""))
    return "\n".join(p for p in parts if p.strip()).strip()


def _actor_source_text(messages: list[Any]) -> str:
    replay = build_actor_replay_text(list(messages or []))
    if replay.strip():
        return replay.strip()
    return _last_ai_text(messages)


def _response_text(resp: Any) -> str:
    if isinstance(resp, AIMessage):
        return str(resp.content or "")
    return str(getattr(resp, "content", resp) or "")


def _one_line_summary(text: str, *, max_len: int = 500) -> str:
    line = " ".join((text or "").split()).strip()
    if len(line) > max_len:
        return line[: max_len - 3].rstrip() + "..."
    return line


def _parse_evaluator_json(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {"redo": False, "suggestion": ""}
    decoder = json.JSONDecoder()
    start = raw.find("{")
    while start != -1:
        try:
            obj, _end = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            start = raw.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            redo = bool(obj.get("redo", False))
            suggestion = str(obj.get("suggestion") or "").strip()
            return {"redo": redo, "suggestion": suggestion}
        start = raw.find("{", start + 1)
    redo_match = re.search(r'"redo"\s*:\s*(true|false)', raw, re.IGNORECASE)
    if redo_match:
        redo = redo_match.group(1).lower() == "true"
        sug_match = re.search(r'"suggestion"\s*:\s*"([^"]*)"', raw)
        suggestion = sug_match.group(1) if sug_match else ""
        return {"redo": redo, "suggestion": suggestion}
    return {"redo": False, "suggestion": ""}


class Evaluator:
    def __init__(self, llm=None, compressor_llm=None):
        self.llm = llm
        self.compressor_llm = compressor_llm if compressor_llm is not None else llm

    def _compress_one(self, kind: str, source_text: str) -> str:
        text = (source_text or "").strip()
        if not text:
            return "(none)"
        llm = self.compressor_llm
        if llm is None:
            return _one_line_summary(text)
        instruction = _COMPRESS_INSTRUCTIONS[kind]
        prompt = f"{instruction}\n\n---\n{text}"
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            summary = _one_line_summary(_response_text(resp))
            return summary or _one_line_summary(text)
        except Exception:
            return _one_line_summary(text)

    def _compress_parallel(
        self,
        *,
        detector_text: str,
        actor_text: str,
        suggestion_text: str,
    ) -> tuple[str, str, str]:
        llm = self.compressor_llm
        if llm is None:
            return (
                _one_line_summary(detector_text) if detector_text.strip() else "(none)",
                _one_line_summary(actor_text) if actor_text.strip() else "(none)",
                _one_line_summary(suggestion_text) if suggestion_text.strip() else "(none)",
            )

        jobs: list[tuple[str, Callable[[], str]]] = [
            ("detector", lambda: self._compress_one("detector", detector_text)),
            ("actor", lambda: self._compress_one("actor", actor_text)),
        ]
        if suggestion_text.strip():
            jobs.append(
                ("suggestion", lambda: self._compress_one("suggestion", suggestion_text))
            )

        results: dict[str, str] = {}
        max_workers = min(3, len(jobs))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            # Preserve LangChain's callback context in compression workers. Each
            # job needs its own context; a Context cannot run concurrently.
            future_map = {pool.submit(copy_context().run, fn): key for key, fn in jobs}
            for fut in future_map:
                key = future_map[fut]
                try:
                    results[key] = fut.result()
                except Exception:
                    if key == "detector":
                        results[key] = _one_line_summary(detector_text)
                    elif key == "actor":
                        results[key] = _one_line_summary(actor_text)
                    else:
                        results[key] = _one_line_summary(suggestion_text)

        suggestion_summary = results.get("suggestion", "(none)")
        return (
            results.get("detector", "(none)"),
            results.get("actor", "(none)"),
            suggestion_summary,
        )

    def _build_evaluator_prompt(
        self,
        state: WorkflowState,
        *,
        detector_summary: str,
        actor_summary: str,
        suggestion_summary: str,
    ) -> str:
        iteration = int(state.get("iteration_count") or 0)
        parts = [
            EVALUATOR_ROLE,
            f"Detector: {detector_summary}",
            f"Actor: {actor_summary}",
        ]
        prior = str(state.get("suggestion") or "").strip()
        if prior:
            parts.append(f"Previous suggestion: {suggestion_summary}")
        parts.append(EVALUATOR_OUTPUT)
        return "\n\n".join(parts)

    def evaluate(self, state: WorkflowState) -> dict[str, Any]:
        if state.get("has_error", False):
            return {"redo": False, "suggestion": ""}
        if self.llm is None:
            return {
                "has_error": True,
                "error_message": "evaluator llm is not configured",
                "redo": False,
                "suggestion": "",
            }

        detector_text = _detector_source_text(list(state.get("detector_messages") or []))
        actor_text = _actor_source_text(list(state.get("actor_messages") or []))
        suggestion_text = str(state.get("suggestion") or "").strip()

        try:
            detector_summary, actor_summary, suggestion_summary = self._compress_parallel(
                detector_text=detector_text,
                actor_text=actor_text,
                suggestion_text=suggestion_text,
            )
            prompt = self._build_evaluator_prompt(
                state,
                detector_summary=detector_summary,
                actor_summary=actor_summary,
                suggestion_summary=suggestion_summary,
            )
            messages: list[BaseMessage] = [HumanMessage(content=prompt)]
            resp = self.llm.invoke(messages)
            text = _response_text(resp)
        except Exception as exc:
            return {
                "has_error": True,
                "error_message": str(exc),
                "redo": False,
                "suggestion": "",
            }

        parsed = _parse_evaluator_json(text)
        return {
            "redo": bool(parsed.get("redo", False)),
            "suggestion": str(parsed.get("suggestion") or "").strip(),
        }
