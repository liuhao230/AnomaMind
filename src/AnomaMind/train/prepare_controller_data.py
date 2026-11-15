"""Build one-step redo examples from baseline diagnostic traces.

Usage: python -m AnomaMind.train.prepare_controller_data RESULTS --output train.jsonl
Split original series/services before collecting traces; this command never splits
windows at random and never treats an unobserved continuation as a negative label.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from AnomaMind.core.budget_controller import feature_vector, quality


def examples_from_trace(trace: dict, ground_truth: list) -> list[dict[str, Any]]:
    if trace.get("schema_version") != 1 or trace.get("cost_unit") != "llm_calls":
        raise ValueError("Unsupported diagnostic trace schema or cost unit")
    observed = trace["observed_indices"]
    if not observed or not trace.get("series_id") or not trace.get("run_id"):
        raise ValueError("Trace requires observed indices, series_id, and run_id")
    examples = []
    for current, following in zip(trace["passes"], trace["passes"][1:]):
        if not current["review"]["redo"] or not current["decision"]["continue"]:
            continue
        if following["pass_index"] != current["pass_index"] + 1:
            raise ValueError("Cannot pair nonconsecutive diagnostic passes")
        feature_vector(current["features"])
        extra = following["usage"]["llm_calls"] - current["usage"]["llm_calls"]
        if extra < 0:
            raise ValueError("Cumulative LLM call count decreased")
        stop = quality(current["prediction"], ground_truth, observed)
        continued = quality(following["prediction"], ground_truth, observed)
        examples.append({
            "schema_version": 1,
            "cost_unit": "llm_calls",
            "series_id": trace["series_id"],
            "state_id": f"{trace['run_id']}:{trace['segment_id']}:{current['pass_index']}",
            "features": current["features"],
            "targets": {"quality_stop": stop, "quality_continue": continued,
                        "gain": continued - stop, "extra_llm_calls": extra},
        })
    return examples


def build_dataset(input_dir: Path, output: Path) -> dict[str, int]:
    rows = []
    traces = 0
    for path in sorted(input_dir.rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or "diagnostics" not in record:
            continue
        ground_truth = record.get("ground_truth_intervals")
        if ground_truth is None:
            # Labels are loaded only by this offline command, never by the controller.
            from AnomaMind.core.utils import load_segment_data
            ground_truth = load_segment_data(Path(record["segment_dir"]))["ground_truth"]
        if not isinstance(ground_truth, list):
            raise ValueError(f"Ground truth must be an interval list: {path}")
        for trace in [*record.get("previous_diagnostic_attempts", []), record["diagnostics"]]:
            rows.extend(examples_from_trace(trace, ground_truth))
            traces += 1
    if not rows:
        raise ValueError("No observed stop/continue pairs; collect baseline traces with reflection enabled")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    return {"traces": traces, "examples": len(rows), "series": len({r["series_id"] for r in rows})}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_dataset(args.input_dir, args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
