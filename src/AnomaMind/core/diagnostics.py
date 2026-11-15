"""Per-invocation diagnostics; no labels, prompts, or credentials in call accounting."""
from __future__ import annotations

import copy
import threading
import time
import uuid
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class DiagnosticRecorder(BaseCallbackHandler):
    """Account for nested LLM calls, including Evaluator compression workers.

    Create one recorder per graph invocation, and pass ``runnable_config()`` to
    ``invoke``. Internal SDK retries are not separately observable here.
    """

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.run_id = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._calls: dict[str, dict[str, Any]] = {}

    def runnable_config(self) -> dict[str, Any]:
        return {"callbacks": [self], "configurable": {"diagnostic_recorder": self}}

    def _start(self, serialized: Any, run_id: Any, metadata: Any) -> None:
        kwargs = (serialized or {}).get("kwargs") or {}
        with self._lock:
            self._calls[str(run_id)] = {
                "role": (metadata or {}).get("langgraph_node", "unknown"),
                "model": kwargs.get("model_name", kwargs.get("model", "unknown")),
                "started_seconds": time.perf_counter() - self.started,
                "status": "running",
                "input_tokens": None,
                "output_tokens": None,
            }

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        self._start(serialized, run_id, metadata)

    def on_llm_start(self, serialized, prompts, *, run_id, metadata=None, **kwargs):
        self._start(serialized, run_id, metadata)

    def on_llm_end(self, response, *, run_id, **kwargs):
        usage = None
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if usage is not None:
                    break
            if usage is not None:
                break
        if usage is None:
            raw = (response.llm_output or {}).get("token_usage") or {}
            if "prompt_tokens" in raw and "completion_tokens" in raw:
                usage = {"input_tokens": raw["prompt_tokens"], "output_tokens": raw["completion_tokens"]}
        with self._lock:
            call = self._calls.get(str(run_id))
            if call is None:
                return
            call["status"] = "ok"
            call["elapsed_seconds"] = time.perf_counter() - self.started - call["started_seconds"]
            if usage is not None:
                for key in ("input_tokens", "output_tokens"):
                    value = usage.get(key)
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        call[key] = value

    def on_llm_error(self, error, *, run_id, **kwargs):
        with self._lock:
            call = self._calls.get(str(run_id))
            if call is not None:
                call["status"] = "error"
                call["elapsed_seconds"] = time.perf_counter() - self.started - call["started_seconds"]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            calls = copy.deepcopy(list(self._calls.values()))
        known = [c for c in calls if c["input_tokens"] is not None and c["output_tokens"] is not None]
        return {
            "llm_calls": len(calls),
            "failed_llm_calls": sum(c["status"] == "error" for c in calls),
            "input_tokens": sum(c["input_tokens"] or 0 for c in calls),
            "output_tokens": sum(c["output_tokens"] or 0 for c in calls),
            "calls_without_token_usage": len(calls) - len(known),
            "elapsed_seconds": time.perf_counter() - self.started,
            "calls": calls,
        }
