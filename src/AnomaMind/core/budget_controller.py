"""Small, versioned linear predictors for the value and cost of one redo.

    Cost is the number of *additional LLM invocations*, not currency or tokens.
    Only inference-time numeric features are accepted by the predictor.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


FEATURE_NAMES = (
    "pass_index", "remaining_redos", "window_size", "candidate_count",
    "interval_count", "predicted_fraction", "mean_confidence", "parse_error",
    "tool_calls", "tool_errors", "suggestion_length", "spent_llm_calls",
    "pass_llm_calls", "value_std", "largest_step",
)
MODEL_VERSION = 1


def covered_indices(intervals: list, observed: list[int]) -> set[int]:
    """Clip predictions to observed indices without expanding arbitrary intervals."""
    return {i for i in observed if any(int(a) <= i <= int(b) for a, b in intervals)}


def quality(prediction: dict, ground_truth: list, observed: list[int]) -> float:
    if prediction.get("has_error"):
        return -1.0
    pred = covered_indices(prediction.get("anomaly_intervals", []), observed)
    truth = covered_indices(ground_truth, observed)
    if not truth:
        return 1.0 / (len(pred) + 1)
    return 2.0 * len(pred & truth) / (len(pred) + len(truth))


def feature_vector(features: Mapping[str, Any]) -> list[float]:
    if set(features) != set(FEATURE_NAMES):
        raise ValueError("Controller features do not match the versioned feature schema")
    values = [float(features[name]) for name in FEATURE_NAMES]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Controller features must be finite")
    return values


@dataclass(frozen=True)
class BudgetController:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    gain_weights: tuple[float, ...]
    cost_weights: tuple[float, ...]

    def __post_init__(self):
        n = len(FEATURE_NAMES)
        for values, expected in ((self.means, n), (self.scales, n),
                                 (self.gain_weights, n + 1), (self.cost_weights, n + 1)):
            if len(values) != expected or not all(math.isfinite(v) for v in values):
                raise ValueError("Invalid controller dimensions or non-finite parameters")
        if any(v <= 0 for v in self.scales):
            raise ValueError("Controller scales must be positive")

    def predict(self, features: Mapping[str, Any]) -> tuple[float, float]:
        values = feature_vector(features)
        x = [1.0] + [(v - m) / s for v, m, s in zip(values, self.means, self.scales)]
        gain = sum(a * b for a, b in zip(x, self.gain_weights))
        cost = sum(a * b for a, b in zip(x, self.cost_weights))
        return max(-2.0, min(2.0, gain)), max(0.0, cost)

    def decide(self, features: Mapping[str, Any], cost_weight: float) -> dict[str, Any]:
        if not math.isfinite(cost_weight) or cost_weight < 0:
            raise ValueError("Controller cost weight must be finite and nonnegative")
        gain, cost = self.predict(features)
        return {
            "continue": gain > cost_weight * cost,
            "predicted_gain": gain,
            "predicted_extra_llm_calls": cost,
            "cost_weight": cost_weight,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_VERSION, "cost_unit": "llm_calls",
            "feature_names": list(FEATURE_NAMES), "means": list(self.means),
            "scales": list(self.scales), "gain_weights": list(self.gain_weights),
            "cost_weights": list(self.cost_weights),
        }

    @classmethod
    def load(cls, path: str | Path) -> "BudgetController":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if (obj.get("schema_version") != MODEL_VERSION or obj.get("cost_unit") != "llm_calls"
                or obj.get("feature_names") != list(FEATURE_NAMES)):
            raise ValueError("Incompatible controller schema, cost unit, or feature order")
        return cls(*(tuple(float(v) for v in obj[key])
                     for key in ("means", "scales", "gain_weights", "cost_weights")))
