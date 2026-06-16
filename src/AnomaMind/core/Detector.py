# -*- coding: utf-8 -*-
"""
Detector: final-answer-only stage.

It consumes Locator hints plus the Actor's tool transcript, then emits the
final anomaly JSON answer for downstream finalize parsing.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from AnomaMind.core.detector_prompt import (
    DETECTOR_FINAL_OUTPUT,
    DETECTOR_ROLE,
    build_actor_replay_text,
)
from AnomaMind.core.knowledge_base import KnowledgeBase
from AnomaMind.core.utils import get_column_names
from AnomaMind.core.workflowstate import WorkflowState


def _include_visual_in_prompt() -> bool:
    import os

    raw = str(os.getenv("DETECTOR_INCLUDE_VISUAL", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _strip_locator_visual_block(text: str) -> str:
    import re

    prompt = (text or "").strip()
    if not prompt:
        return ""
    prompt = re.sub(
        r"\n*Locator \(vision, coarse index ranges\).*?(?=\n## |\n\n## |\Z)",
        "",
        prompt,
        flags=re.DOTALL,
    )
    prompt = re.sub(r"\n*VisualDescription:.*?(?=\n[A-Z]|\n## |\Z)", "", prompt, flags=re.DOTALL)
    prompt = re.sub(r"\n*VisualLocalization \(index pairs\):.*?(?=\n|\Z)", "", prompt)
    prompt = re.sub(
        r"\n*VisualLocalizationRationale \(interval \+ rationale from Locator JSON\):.*?(?=\n|\Z)",
        "",
        prompt,
    )
    return prompt.strip()


def _strip_visual_from_assistant_thinking(content: str) -> str:
    import re

    text = str(content or "")
    if not text:
        return text
    # Drop sentences that cite visual description/localization in redacted_thinking.
    text = re.sub(
        r"\s*The visual description[^.]*\.\s*",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*Given the lack of statistical evidence for anomalies and the visual confirmation[^.]*\.\s*",
        " Given the lack of statistical evidence for anomalies, there are no anomalies to report. ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _strip_visual_from_domain_knowledge(text: str) -> str:
    import re

    body = (text or "").strip()
    if not body:
        return body
    body = re.sub(
        r"\n*## [^\n]*second-stage evidence priority[^\n]*\n(?:- [^\n]+\n?)+",
        "",
        body,
        flags=re.IGNORECASE,
    )
    lines = []
    for line in body.splitlines():
        low = line.lower()
        if "visual localization" in low or "visual description" in low or line.strip().startswith("VisualDescription:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _state_context_block(state: WorkflowState, knowledge_base: KnowledgeBase) -> str:
    description = str(state.get("visual_description") or "").strip()
    dataset_key = str(state.get("type") or "UNKNOWN").strip().upper()
    localization = state.get("visual_localization") or []
    localization_json = json.dumps(localization if isinstance(localization, list) else [], ensure_ascii=False)
    localization_items = state.get("visual_localization_items") or []
    if not isinstance(localization_items, list):
        localization_items = []
    localization_items_json = json.dumps(localization_items, ensure_ascii=False)
    kb_raw = knowledge_base.get_agent_knowledge("detector", dataset_key)
    domain_knowledge = str(kb_raw).strip() if kb_raw else "(none)"
    return (
        "Here is domain knowledge:\n"
        f"{domain_knowledge}\n\n"
        "Locator (vision, coarse index ranges)\n"
        "IMPORTANT: Treat the Locator's visual localization as the **main candidate evidence** for where anomalies are. "
        "Use actual tool evidence to confirm, reject, or refine those localized intervals before concluding.\n"
        f"VisualDescription: {description or '(none)'}\n"
        f"VisualLocalization (index pairs): {localization_json}\n"
        f"VisualLocalizationRationale (interval + rationale from Locator JSON): {localization_items_json}"
    )


def _series_values_from_state(state: WorkflowState) -> dict[str, Any]:
    data = state.get("data")
    if data is None:
        return {}
    try:
        timestamp_col, value_col = get_column_names(data)
    except Exception:
        return {}

    values = data[value_col].tolist()
    indices = data[timestamp_col].tolist() if timestamp_col in data.columns else list(range(len(values)))
    out: dict[str, Any] = {}
    for idx, value in zip(indices, values):
        key = str(int(idx)) if isinstance(idx, (int, float)) else str(idx)
        try:
            out[key] = round(float(value), 3)
        except Exception:
            out[key] = value
    return out


def flatten_actor_messages_to_text(messages: list[Any]) -> str:
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            continue
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                lines.append("## Actor Tool Calls")
                for index, one in enumerate(tool_calls):
                    if not isinstance(one, dict):
                        continue
                    try:
                        args_text = json.dumps(one.get("args", {}), ensure_ascii=False)
                    except Exception:
                        args_text = str(one.get("args", {}))
                    lines.append(
                        f"- call[{index}] name={one.get('name', '')} args={args_text}"
                    )
                continue
            content = str(getattr(msg, "content", "") or "").strip()
            if content:
                lines.append("## Actor Note")
                lines.append(content)
            continue
        if isinstance(msg, ToolMessage):
            lines.append("## Tool Result")
            lines.append(str(getattr(msg, "content", "") or ""))
    return "\n".join(lines).strip()

def _legacy_style_replay_from_actor_messages(messages: list[Any]) -> str:
    return build_actor_replay_text(messages)

def _legacy_style_messages_from_actor_messages(messages: list[Any]) -> list[BaseMessage]:

    if not messages:
        return []

    # Reuse the replay parser to locate and sanitize the first user prompt,
    # but emit real ToolMessages instead of embedding everything in one string.
    user_prompt: str | None = None
    for m in messages:
        if isinstance(m, HumanMessage):
            content = str(getattr(m, "content", "") or "").strip()
            if content:
                user_prompt = content
                break
        if isinstance(m, dict) and (m.get("role") == "user"):
            content = str(m.get("content") or "").strip()
            if content:
                user_prompt = content
                break

    def _sanitize_user_prompt(text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""
        for cut in (
            "\n\n## Your job in this stage",
            "\n## Your job in this stage",
            "\n\n## Output format (after you stop calling tools)",
            "\n## Output format (after you stop calling tools)",
        ):
            i = t.find(cut)
            if i != -1:
                t = t[:i].rstrip()
        # Also drop any leading "User:" label if it was already embedded.
        if t.startswith("User:\n"):
            t = t[len("User:\n") :].lstrip()
        return t

    out: list[BaseMessage] = []
    if user_prompt:
        out.append(HumanMessage(content=_sanitize_user_prompt(user_prompt)))

    # Map tool_call_id -> tool name so ToolMessages include `name` (required by some providers).
    id_to_name: dict[str, str] = {}
    for m in messages:
        tool_calls = None
        if isinstance(m, AIMessage):
            tool_calls = getattr(m, "tool_calls", None) or []
        elif isinstance(m, dict) and (m.get("role") == "assistant"):
            tool_calls = m.get("tool_calls") or []
        if not tool_calls or not isinstance(tool_calls, list):
            continue
        for one in tool_calls:
            if not isinstance(one, dict):
                continue
            tid = str(one.get("id") or "").strip()
            nm = str(one.get("name") or "").strip()
            if tid and nm:
                id_to_name[tid] = nm

    # Add a lightweight tool-call list as plain text (NOT a summary), so tool outputs
    # are grounded as responses to explicit calls (closer to legacy AgentLoop shape).
    tool_call_lines: list[str] = []
    for m in messages:
        if isinstance(m, AIMessage):
            tool_calls = getattr(m, "tool_calls", None) or []
        elif isinstance(m, dict) and (m.get("role") == "assistant"):
            tool_calls = m.get("tool_calls") or []
        else:
            continue
        if not isinstance(tool_calls, list) or not tool_calls:
            continue
        for one in tool_calls:
            if not isinstance(one, dict):
                continue
            name = str(one.get("name") or "")
            try:
                args_text = json.dumps(one.get("args", {}), ensure_ascii=False)
            except Exception:
                args_text = str(one.get("args", {}))
            tool_call_lines.append(f"- name={name} args={args_text}")
        # only include the first tool-call batch (legacy: one assistant tool-call turn)
        break
    if tool_call_lines:
        out.append(HumanMessage(content="Assistant:\n(tool calls)\n" + "\n".join(tool_call_lines)))

    # Append tool outputs as ToolMessages in original order.
    for m in messages:
        if isinstance(m, ToolMessage):
            tid = str(getattr(m, "tool_call_id", "") or "")
            out.append(
                ToolMessage(
                    content=str(getattr(m, "content", "") or ""),
                    tool_call_id=tid,
                    name=id_to_name.get(tid),
                )
            )
        elif isinstance(m, dict) and (m.get("role") == "tool"):
            tid = str(m.get("tool_call_id") or "")
            out.append(
                ToolMessage(
                    content=str(m.get("content") or ""),
                    tool_call_id=tid,
                    name=id_to_name.get(tid),
                )
            )

    return out


def _coerce_ai_message(resp: Any) -> AIMessage:
    if isinstance(resp, AIMessage):
        return resp
    content = getattr(resp, "content", resp)
    if not isinstance(content, str):
        content = str(content or "")
    return AIMessage(content=content)


def _append_redo_suggestion(prompt: str, state: WorkflowState) -> str:
    if not state.get("redo"):
        return prompt
    suggestion = str(state.get("suggestion") or "").strip()
    if not suggestion:
        return prompt
    return f"{prompt.rstrip()}\n\nsuggestion: {suggestion}"


def _detector_final_knowledge(knowledge_base: KnowledgeBase, state: WorkflowState) -> str:
    dataset_key = str(state.get("type") or "").strip().upper()
    if not dataset_key:
        return ""
    knowledge = getattr(knowledge_base, "prior_knowledge", {}).get(f"detector_final_{dataset_key}", "")
    return str(knowledge).strip() if knowledge else ""


class Detector:
    def __init__(self, knowledge_base: KnowledgeBase, llm=None):
        self.knowledge_base = knowledge_base
        self.llm = llm

    def _build_messages(self, state: WorkflowState) -> list[BaseMessage]:
        # NOTE: We intentionally keep Detector input as plain text (single HumanMessage).
        # Some providers reject ToolMessage/function_response structures in non-tool sessions.
        actor_messages = list(state.get("actor_messages") or [])
        legacy_replay = _legacy_style_replay_from_actor_messages(actor_messages)
        context_fallback = _state_context_block(state, self.knowledge_base).strip()

        parts: list[str] = []
        if legacy_replay:
            parts.append(legacy_replay)
        else:
            parts.append(DETECTOR_ROLE)
            if context_fallback:
                parts.append(context_fallback)

        final_knowledge = _detector_final_knowledge(self.knowledge_base, state)
        if final_knowledge:
            parts.append(final_knowledge)
        parts.append(DETECTOR_FINAL_OUTPUT)
        parts.append("Produce the final output now. Reply using only the required final output format.")
        prompt = "\n\n".join([p for p in parts if p])
        prompt = _append_redo_suggestion(prompt, state)
        return [HumanMessage(content=prompt)]

    def detect(self, state: WorkflowState) -> dict[str, Any]:
        if state.get("has_error", False):
            return {}
        if self.llm is None:
            return {
                "has_error": True,
                "error_message": "detector llm is not configured",
            }

        detector_messages = list(state.get("detector_messages") or [])

        try:
            prompt_messages = self._build_messages(state)
            detector_messages = list(prompt_messages)
            ai = _coerce_ai_message(self.llm.invoke(prompt_messages))
        except Exception as exc:
            err = str(exc)
            return {
                "detector_messages": detector_messages,
                "has_error": True,
                "error_message": err,
            }

        detector_messages.append(ai)
        return {
            "detector_messages": detector_messages,
        }
