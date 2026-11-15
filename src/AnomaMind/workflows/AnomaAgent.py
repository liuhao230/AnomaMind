from __future__ import annotations

import logging
import math
from typing import Literal, Optional

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from AnomaMind.core.Actor import Actor
from AnomaMind.core.Detector import Detector
from AnomaMind.core.Evaluator import Evaluator
from AnomaMind.core.Locator import Locator
from AnomaMind.core.finalize import (
    finalize_detector_output, last_detector_answer_text, parse_detector_answer_text,
)
from AnomaMind.core.budget_controller import BudgetController, covered_indices
from AnomaMind.core.diagnostics import DiagnosticRecorder
from AnomaMind.core.utils import get_column_names
from AnomaMind.helpers.message_io import export_messages
from AnomaMind.core.knowledge_base import KnowledgeBase
from AnomaMind.core.llm_config import LLMConfig, create_llm, default_locator_llm_config
from AnomaMind.core.workflowstate import WorkflowState

logger = logging.getLogger(__name__)

MAX_EVAL_REDO_ITERATIONS = 3


class AnomaAgent:
    """LangGraph: Locator -> Actor -> Detector -> [Evaluator loop] -> finalize."""

    def __init__(
        self,
        max_turns: int = 8,
        locator_config: LLMConfig | None = None,
        actor_config: LLMConfig | None = None,
        detector_config: LLMConfig | None = None,
        evaluator_config: LLMConfig | None = None,
        knowledgeBase: Optional[KnowledgeBase] = None,
        max_eval_redo_iterations: int = MAX_EVAL_REDO_ITERATIONS,
        enable_reflection: bool = True,
        capture_diagnostics: bool = False,
        controller: BudgetController | None = None,
        controller_cost_weight: float = 0.0,
    ):
        self.max_turns = max_turns
        self.max_eval_redo_iterations = max(0, int(max_eval_redo_iterations))
        self.enable_reflection = bool(enable_reflection)
        if controller is not None and not self.enable_reflection:
            raise ValueError("A redo controller requires reflection")
        if not math.isfinite(controller_cost_weight) or controller_cost_weight < 0:
            raise ValueError("Controller cost weight must be finite and nonnegative")
        self.capture_diagnostics = capture_diagnostics or controller is not None
        self.controller = controller
        self.controller_cost_weight = controller_cost_weight
        self.knowledgeBase = knowledgeBase if knowledgeBase else KnowledgeBase()

        loc_cfg = locator_config or default_locator_llm_config()
        self.locator_llm = create_llm(loc_cfg)
        self.locator = Locator(llm=self.locator_llm, knowledge_base=self.knowledgeBase)

        act_cfg = actor_config or detector_config or LLMConfig()
        self.actor_llm = create_llm(act_cfg)
        self.actor = Actor(knowledge_base=self.knowledgeBase, llm=self.actor_llm)

        det_cfg = detector_config or actor_config or LLMConfig()
        self.detector_llm = create_llm(det_cfg)
        self.detector = Detector(knowledge_base=self.knowledgeBase, llm=self.detector_llm)

        self.evaluator: Evaluator | None = None
        if self.enable_reflection:
            ev_cfg = evaluator_config or actor_config or detector_config or LLMConfig()
            self.evaluator = Evaluator(llm=create_llm(ev_cfg))

    def locator_node(self, state: WorkflowState, config: RunnableConfig) -> dict:
        if self.capture_diagnostics and not isinstance(
            config.get("configurable", {}).get("diagnostic_recorder"), DiagnosticRecorder,
        ):
            raise ValueError("Diagnostic capture requires DiagnosticRecorder.runnable_config()")
        return self.locator.locate(state)

    def actor_node(self, state: WorkflowState) -> dict:
        return self.actor.act(state, max_turns=self.max_turns)

    def detector_node(self, state: WorkflowState) -> dict:
        return self.detector.detect(state)

    def evaluator_node(self, state: WorkflowState) -> dict:
        if self.evaluator is None:
            return {"redo": False, "suggestion": ""}
        return self.evaluator.evaluate(state)

    def prepare_redo_node(self, state: WorkflowState) -> dict:
        return {
            "actor_messages": [],
            "detector_messages": [],
            "final_output": {},
            "iteration_count": int(state.get("iteration_count") or 0) + 1,
        }

    def diagnostic_node(self, state: WorkflowState, config: RunnableConfig) -> dict:
        recorder = config.get("configurable", {}).get("diagnostic_recorder")
        if not isinstance(recorder, DiagnosticRecorder):
            raise ValueError("Diagnostic capture requires DiagnosticRecorder.runnable_config()")
        prediction = parse_detector_answer_text(
            last_detector_answer_text(list(state.get("detector_messages") or []))
        )
        if state.get("has_error"):
            prediction["has_error"] = True
            prediction["error_message"] = str(state.get("error_message") or "")
        usage = recorder.snapshot()
        passes = list(state.get("diagnostic_passes") or [])
        iteration = int(state.get("iteration_count") or 0)
        data = state["data"]
        index_col, value_col = get_column_names(data)
        observed = [int(i) for i in data[index_col]]
        # Deliberately select only the observed values, never the Label column.
        values = data[value_col].astype(float)
        tool_messages = [m for m in state.get("actor_messages", []) if isinstance(m, ToolMessage)]
        confidences = prediction.get("anomaly_confidences") or []
        previous_calls = passes[-1]["usage"]["llm_calls"] if passes else 0
        std = float(values.std(ddof=0))
        step = float(values.diff().abs().max()) if len(values) > 1 else 0.0
        features = {
            "pass_index": iteration,
            "remaining_redos": max(0, self.max_eval_redo_iterations - iteration + 1),
            "window_size": len(observed),
            "candidate_count": len(state.get("visual_localization") or []),
            "interval_count": len(prediction["anomaly_intervals"]),
            "predicted_fraction": len(covered_indices(prediction["anomaly_intervals"], observed)) / max(1, len(observed)),
            "mean_confidence": sum(confidences) / max(1, len(confidences)),
            "parse_error": int(bool(prediction["has_error"])),
            "tool_calls": len(tool_messages),
            "tool_errors": sum('"error"' in str(m.content) for m in tool_messages),
            "suggestion_length": len(str(state.get("suggestion") or "")),
            "spent_llm_calls": usage["llm_calls"],
            "pass_llm_calls": usage["llm_calls"] - previous_calls,
            "value_std": std if math.isfinite(std) else 0.0,
            "largest_step": step if math.isfinite(step) else 0.0,
        }
        should_redo = self.enable_reflection and self._route_after_evaluator(state) == "prepare_redo"
        if state.get("has_error"):
            reason = "error"
        elif not self.enable_reflection:
            reason = "reflection_disabled"
        elif not state.get("redo"):
            reason = "evaluator_accepted"
        elif not should_redo:
            reason = "redo_limit"
        else:
            reason = "evaluator_redo"
        decision = {"continue": bool(should_redo), "reason": reason}
        if should_redo and self.controller is not None:
            decision.update(self.controller.decide(features, self.controller_cost_weight))
            decision["reason"] = "controller_continue" if decision["continue"] else "controller_stop"
        decision["unresolved_review"] = bool(state.get("redo")) and not decision["continue"]
        passes.append({
            "pass_index": iteration,
            "prediction": prediction,
            "review": {"redo": bool(state.get("redo")), "suggestion": str(state.get("suggestion") or "")},
            "features": features,
            "usage": usage,
            "decision": decision,
            "actor_messages": export_messages(list(state.get("actor_messages") or [])),
            "detector_messages": export_messages(list(state.get("detector_messages") or [])),
        })
        return {"diagnostic_passes": passes, "diagnostic_decision": decision}

    def _route_after_diagnostics(self, state: WorkflowState) -> Literal["prepare_redo", "finalize"]:
        return "prepare_redo" if state["diagnostic_decision"]["continue"] else "finalize"

    def finalize(self, state: WorkflowState) -> WorkflowState:
        return self._finalize_output(state)

    def _route_after_evaluator(
        self, state: WorkflowState
    ) -> Literal["prepare_redo", "finalize"]:
        if state.get("has_error"):
            return "finalize"
        if not state.get("redo"):
            return "finalize"
        if int(state.get("iteration_count") or 0) > self.max_eval_redo_iterations:
            return "finalize"
        return "prepare_redo"

    def build_workflow(self) -> StateGraph:
        workflow = StateGraph(WorkflowState)
        workflow.add_node("locator", self.locator_node)  # type: ignore
        workflow.add_node("actor", self.actor_node)  # type: ignore
        workflow.add_node("detector", self.detector_node)  # type: ignore
        workflow.add_node("finalize", self.finalize)  # type: ignore
        workflow.set_entry_point("locator")
        workflow.add_edge("locator", "actor")
        workflow.add_edge("actor", "detector")
        if self.capture_diagnostics:
            workflow.add_node("diagnostics", self.diagnostic_node)
        if not self.enable_reflection:
            if self.capture_diagnostics:
                workflow.add_edge("detector", "diagnostics")
                workflow.add_edge("diagnostics", "finalize")
            else:
                workflow.add_edge("detector", "finalize")
            workflow.add_edge("finalize", END)
            return workflow.compile()

        workflow.add_node("evaluator", self.evaluator_node)  # type: ignore
        workflow.add_node("prepare_redo", self.prepare_redo_node)  # type: ignore
        workflow.add_edge("detector", "evaluator")
        if self.capture_diagnostics:
            workflow.add_edge("evaluator", "diagnostics")
        workflow.add_conditional_edges(
            "diagnostics" if self.capture_diagnostics else "evaluator",
            self._route_after_diagnostics if self.capture_diagnostics else self._route_after_evaluator,
            {"prepare_redo": "prepare_redo", "finalize": "finalize"},
        )
        workflow.add_edge("prepare_redo", "actor")
        workflow.add_edge("finalize", END)
        return workflow.compile()

    def _finalize_output(self, state: WorkflowState) -> WorkflowState:
        return finalize_detector_output(state)
