"""Fit two small ridge regressions; no LLM training or GPU required."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from AnomaMind.core.budget_controller import BudgetController, feature_vector


def read_examples(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Controller training/evaluation data is empty")
    seen = set()
    for row in rows:
        if row.get("schema_version") != 1 or row.get("cost_unit") != "llm_calls":
            raise ValueError("Incompatible controller data schema or cost unit")
        feature_vector(row["features"])
        if not row.get("series_id") or not row.get("state_id") or row["state_id"] in seen:
            raise ValueError("Examples require series_id and unique state_id values")
        seen.add(row["state_id"])
        gain = float(row["targets"]["gain"])
        cost = float(row["targets"]["extra_llm_calls"])
        if not math.isfinite(gain) or not -2 <= gain <= 2 or not math.isfinite(cost) or cost < 0:
            raise ValueError("Targets must contain a finite quality gain and nonnegative cost")
    return rows


def fit_controller(rows: list[dict], ridge: float = 1.0) -> BudgetController:
    if not rows or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("Need training examples and a finite positive ridge penalty")
    x = np.asarray([feature_vector(r["features"]) for r in rows], dtype=float)
    y = np.asarray([[r["targets"]["gain"], r["targets"]["extra_llm_calls"]] for r in rows], dtype=float)
    if not np.isfinite(y).all() or (y[:, 1] < 0).any():
        raise ValueError("Training targets must be finite; costs cannot be negative")
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales < 1e-12] = 1.0
    design = np.column_stack([np.ones(len(x)), (x - means) / scales])
    penalty = np.eye(design.shape[1]) * math.sqrt(ridge)
    penalty[0, 0] = 0.0  # Do not regularize the intercept.
    weights, *_ = np.linalg.lstsq(
        np.vstack([design, penalty]), np.vstack([y, np.zeros((len(penalty), 2))]), rcond=None,
    )
    return BudgetController(tuple(means), tuple(scales), tuple(weights[:, 0]), tuple(weights[:, 1]))


def evaluate(model: BudgetController, rows: list[dict]) -> dict[str, float]:
    predictions = np.asarray([model.predict(r["features"]) for r in rows])
    targets = np.asarray([[r["targets"]["gain"], r["targets"]["extra_llm_calls"]] for r in rows])
    errors = ((predictions - targets) ** 2).mean(axis=0)
    return {"gain_mse": float(errors[0]), "extra_llm_calls_mse": float(errors[1])}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_data", type=Path)
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1.0)
    args = parser.parse_args(argv)
    train = read_examples(args.training_data)
    validation = read_examples(args.validation_data) if args.validation_data else None
    if validation is not None:
        overlap = {r["series_id"] for r in train} & {r["series_id"] for r in validation}
        if overlap:
            raise ValueError("Training and validation series overlap; split original series before windowing")
    model = fit_controller(train, args.ridge)
    artifact = model.as_dict()
    artifact["training_examples"] = len(train)
    artifact["ridge"] = args.ridge
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = {"training_examples": len(train), "training_fit": evaluate(model, train)}
    if validation is not None:
        report["validation"] = evaluate(model, validation)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
