import copy
import json

import pytest

from AnomaMind.core.budget_controller import BudgetController, FEATURE_NAMES, quality
from AnomaMind.infer.infer import parse_args
from AnomaMind.train.prepare_controller_data import examples_from_trace
from AnomaMind.train.train_controller import fit_controller, main, read_examples


def features(**changes):
    return {**dict.fromkeys(FEATURE_NAMES, 0.0), **changes}


def constant_controller(gain=0.2, cost=3.0):
    n = len(FEATURE_NAMES)
    return BudgetController((0.0,) * n, (1.0,) * n,
                            (gain,) + (0.0,) * n, (cost,) + (0.0,) * n)


def trace():
    return {
        "schema_version": 1, "cost_unit": "llm_calls", "run_id": "run-1",
        "series_id": "source-1", "segment_id": "window-1", "observed_indices": [10, 11, 12],
        "passes": [
            {"pass_index": 0, "review": {"redo": True}, "decision": {"continue": True},
             "prediction": {"anomaly_intervals": [[10, 10]]},
             "features": features(), "usage": {"llm_calls": 7}},
            {"pass_index": 1, "review": {"redo": False}, "decision": {"continue": False},
             "prediction": {"anomaly_intervals": [[11, 11]]},
             "features": features(pass_index=1), "usage": {"llm_calls": 13}},
        ],
    }


def test_gain_and_cost_both_influence_decision():
    model = constant_controller()
    assert model.decide(features(), 0.01)["continue"]
    assert not model.decide(features(), 0.1)["continue"]
    assert not constant_controller(gain=-0.1).decide(features(), 0)["continue"]


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_invalid_cost_weight_rejected(weight):
    with pytest.raises(ValueError):
        constant_controller().decide(features(), weight)


def test_model_round_trip_and_validation(tmp_path):
    path = tmp_path / "model.json"
    obj = constant_controller().as_dict()
    path.write_text(json.dumps(obj))
    assert BudgetController.load(path).predict(features()) == (0.2, 3.0)
    obj["feature_names"].reverse()
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="schema"):
        BudgetController.load(path)
    obj = constant_controller().as_dict()
    obj["scales"][0] = 0
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="positive"):
        BudgetController.load(path)


def test_quality_counts_normal_false_positives_and_clips_huge_intervals():
    observed = [10, 11, 12]
    assert quality({"anomaly_intervals": []}, [], observed) == 1
    assert quality({"anomaly_intervals": [[10, 10]]}, [], observed) == 0.5
    assert quality({"anomaly_intervals": [[-10**15, 10**15]]}, [[11, 11]], observed) == 0.5
    assert quality({"has_error": True}, [], observed) == -1


def test_adjacent_pairs_have_only_current_features_and_observed_cost():
    t = trace()
    rows = examples_from_trace(t, [[11, 11]])
    assert len(rows) == 1
    assert rows[0]["targets"] == {"quality_stop": 0, "quality_continue": 1, "gain": 1, "extra_llm_calls": 6}
    other_labels = examples_from_trace(t, [])
    assert other_labels[0]["features"] == rows[0]["features"]
    assert other_labels[0]["targets"]["quality_stop"] == 0.5
    t["passes"][0]["decision"]["continue"] = False
    assert examples_from_trace(t, [[11, 11]]) == []  # Unobserved continuation is not a zero-gain label.


def test_failed_continuation_is_not_silently_filtered():
    t = trace()
    t["passes"][1]["prediction"] = {"has_error": True}
    row = examples_from_trace(t, [[11, 11]])[0]
    assert row["targets"]["quality_continue"] == -1
    assert row["targets"]["gain"] == -1
    t["passes"][1]["pass_index"] = 3
    with pytest.raises(ValueError, match="nonconsecutive"):
        examples_from_trace(t, [])


def test_training_and_validation_are_group_disjoint(tmp_path):
    rows = examples_from_trace(trace(), [[11, 11]])
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    output = tmp_path / "controller.json"
    train.write_text(json.dumps(rows[0]) + "\n")
    validation.write_text(train.read_text())
    with pytest.raises(ValueError, match="series overlap"):
        main([str(train), "--validation-data", str(validation), "--output", str(output)])
    assert not output.exists()
    val = copy.deepcopy(rows[0])
    val["series_id"] = "held-out-source"
    validation.write_text(json.dumps(val) + "\n")
    assert main([str(train), "--validation-data", str(validation), "--output", str(output)]) == 0
    assert BudgetController.load(output).predict(features()) == pytest.approx((1, 6))


def test_training_rejects_label_features_and_nan_targets(tmp_path):
    row = examples_from_trace(trace(), [[11, 11]])[0]
    path = tmp_path / "train.jsonl"
    row["features"]["quality_continue"] = 1
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="feature schema"):
        read_examples(path)
    del row["features"]["quality_continue"]
    row["targets"]["gain"] = float("nan")
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="finite"):
        read_examples(path)
    with pytest.raises(ValueError):
        fit_controller([row])


def test_cli_rejects_controller_without_reflection():
    with pytest.raises(SystemExit):
        parse_args(["data", "-o", "out", "--controller-model", "model.json", "--no-reflection"])
