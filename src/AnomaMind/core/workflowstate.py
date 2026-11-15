from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd


class WorkflowState(TypedDict, total=False):
    """Workflow state shared across locator, actor, detector, evaluator, and finalize."""

    path: str
    data: pd.DataFrame
    type: str
    visual_description: str
    visual_localization: list[list[int]]
    visual_localization_items: list[dict[str, Any]]
    actor_messages: list[Any]
    detector_messages: list[Any]
    iteration_count: int
    redo: bool
    suggestion: str
    has_error: bool
    error_message: str
    final_output: dict
    diagnostic_passes: list[dict[str, Any]]
    diagnostic_decision: dict[str, Any]
