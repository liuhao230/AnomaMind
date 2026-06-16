from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def export_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for msg in messages or []:
        if isinstance(msg, dict):
            role = str(msg.get("role") or "").lower()
            if role == "system":
                out.append({"role": "system", "content": msg.get("content", "")})
            elif role == "user":
                out.append({"role": "user", "content": msg.get("content", "")})
            elif role == "assistant":
                item = {"role": "assistant", "content": msg.get("content", "")}
                if msg.get("tool_calls"):
                    item["tool_calls"] = msg.get("tool_calls")
                out.append(item)
            elif role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "content": msg.get("content", ""),
                        "tool_call_id": msg.get("tool_call_id"),
                    }
                )
            else:
                out.append({"role": role or "unknown", "content": msg.get("content", "")})
            continue
        if isinstance(msg, SystemMessage):
            out.append({"role": "system", "content": msg.content})
            continue
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            out.append({"role": "user", "content": content})
            continue
        if isinstance(msg, ToolMessage):
            out.append(
                {
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": getattr(msg, "tool_call_id", None),
                }
            )
            continue
        if isinstance(msg, AIMessage):
            item: Dict[str, Any] = {"role": "assistant", "content": msg.content}
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                simplified = []
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    raw_args = tc.get("args", tc.get("arguments", {}))
                    simplified.append(
                        {
                            "name": tc.get("name", ""),
                            "args": raw_args if isinstance(raw_args, dict) else raw_args,
                            "arguments": raw_args if isinstance(raw_args, dict) else raw_args,
                            "id": tc.get("id", ""),
                        }
                    )
                if simplified:
                    item["tool_calls"] = simplified
            out.append(item)
            continue

        out.append({"role": "unknown", "content": str(getattr(msg, "content", msg))})
    return out


def _ai_message_with_tool_calls(content: str, tool_calls: Optional[List[Dict[str, Any]]]) -> AIMessage:
    if tool_calls:
        try:
            return AIMessage(content=content, tool_calls=tool_calls)  # type: ignore[arg-type]
        except Exception:
            pass
    return AIMessage(content=content)


def import_messages(records: List[Dict[str, Any]]) -> List[Any]:
    msgs: List[Any] = []
    for r in records or []:
        role = (r.get("role") or "").lower()
        if role == "system":
            msgs.append(SystemMessage(content=r.get("content", "")))
        elif role == "user":
            msgs.append(HumanMessage(content=r.get("content", "")))
        elif role == "assistant":
            msgs.append(_ai_message_with_tool_calls(r.get("content", ""), r.get("tool_calls")))
        elif role == "tool":
            msgs.append(ToolMessage(content=r.get("content", ""), tool_call_id=r.get("tool_call_id") or ""))
        else:
            msgs.append(HumanMessage(content=str(r.get("content", ""))))
    return msgs


def exported_messages_tail_for_detector(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(records, list) or len(records) < 5:
        return []
    r0, r1, r2, r3, r4 = records[0], records[1], records[2], records[3], records[4]
    if (
        isinstance(r0, dict)
        and r0.get("role") == "system"
        and isinstance(r1, dict)
        and r1.get("role") == "user"
        and isinstance(r2, dict)
        and r2.get("role") == "assistant"
        and isinstance(r3, dict)
        and r3.get("role") == "system"
        and isinstance(r4, dict)
        and r4.get("role") == "user"
    ):
        return records[5:]
    return []


def extract_used_tool_names_from_records(records: List[Dict[str, Any]]) -> set[str]:
    used: set[str] = set()
    for msg in records or []:
        if not isinstance(msg, dict):
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            name = tc.get("name")
            if isinstance(name, str) and name:
                used.add(name)
    return used
