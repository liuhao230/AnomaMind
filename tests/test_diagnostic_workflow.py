import importlib
import json
import threading
from dataclasses import replace

import pandas as pd
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from AnomaMind.core.diagnostics import DiagnosticRecorder
from AnomaMind.core.llm_config import LLMConfig
from AnomaMind.core.utils import load_segment_data
from AnomaMind.infer.infer import InferConfig, SegmentTask, _execute_tasks, _run_segment_once
from AnomaMind.infer.infer import build_segment_tasks
from AnomaMind.train.prepare_controller_data import build_dataset, examples_from_trace
from test_budget_controller import constant_controller


class ScriptedChat(BaseChatModel):
    role: str
    always_redo: bool = False
    fail_second_detection: bool = False
    _reviews: int = PrivateAttr(default=0)
    _detections: int = PrivateAttr(default=0)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @property
    def _llm_type(self):
        return "scripted-test-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        calls = []
        text = str(messages[0].content)
        if self.role == "locator":
            answer = '<locator_description>Suspicious spike.</locator_description>[{"interval":[0,3]}]'
        elif self.role == "actor":
            if any(isinstance(m, ToolMessage) for m in messages):
                answer = "Evidence acquired."
            else:
                answer = ""
                calls = [{"name": "stats", "args": {}, "id": "stats-1", "type": "tool_call"}]
        elif self.role == "detector":
            self._detections += 1
            if self.fail_second_detection and self._detections == 2:
                raise RuntimeError("simulated service failure")
            point = 0 if self._detections == 1 else 1
            answer = json.dumps([{"interval": [point, point], "confidence": 2, "explanation": "test"}])
        elif text.startswith("Summarize"):
            answer = "Compressed evidence."
        else:
            with self._lock:
                self._reviews += 1
                redo = self.always_redo or self._reviews == 1
            answer = json.dumps({"redo": redo, "suggestion": "Check the boundary." if redo else ""})
        message = AIMessage(content=answer, tool_calls=calls,
                            usage_metadata={"input_tokens": 11, "output_tokens": 3, "total_tokens": 14})
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.fixture
def segment(tmp_path):
    source = tmp_path / "source"
    folder = source / "sample_seg0"
    folder.mkdir(parents=True)
    pd.DataFrame({"Index": range(5), "Value": [0, 5, 0, 0, 0], "Label": [0, 1, 0, 0, 0]}).to_csv(source / "series.csv", index=False)
    (folder / "meta_info.json").write_text(json.dumps({
        "dataset_root": "..", "global_data_file": "series.csv", "series_stem": "series",
        "type": "YAHOO", "index": {"start": 0, "end": 5},
    }))
    (folder / "segment_preview_plain.png").write_bytes(b"image contents are ignored by the test model")
    return folder


def agent_factory(monkeypatch, *, capture=True, controller=None, reflection=True, **model_kwargs):
    module = importlib.import_module("AnomaMind.workflows.AnomaAgent")
    monkeypatch.setattr(module, "create_llm", lambda config: ScriptedChat(role=config.model_name, **model_kwargs))

    def factory():
        agent = module.AnomaAgent(
            locator_config=LLMConfig(model_name="locator"), actor_config=LLMConfig(model_name="actor"),
            detector_config=LLMConfig(model_name="detector"), evaluator_config=LLMConfig(model_name="evaluator"),
            capture_diagnostics=capture, controller=controller,
            enable_reflection=reflection,
        )
        return agent.build_workflow()
    return factory


def initial_state(segment):
    pack = load_segment_data(segment)
    return {"path": pack["path"], "data": pack["segment_data"], "type": "YAHOO"}


def infer_config(tmp_path):
    return InferConfig(output_dir=tmp_path / "results", max_turns=6, enable_reflection=True,
                       no_print_messages=True, no_debug_artifacts=True, workers=1, max_retries=1,
                       retry_delay=0, show_progress=False, force=True, capture_diagnostics=True)


def test_trace_preserves_redos_and_counts_parallel_compression(monkeypatch, segment):
    factory = agent_factory(monkeypatch)
    recorder = DiagnosticRecorder()
    output = factory().invoke(initial_state(segment), recorder.runnable_config())
    assert output["final_output"]["anomaly_intervals"] == [[1, 1]]
    passes = output["diagnostic_passes"]
    assert [p["prediction"]["anomaly_intervals"] for p in passes] == [[[0, 0]], [[1, 1]]]
    assert passes[0]["actor_messages"]  # Still available after prepare_redo clears live history.
    assert passes[0]["usage"]["llm_calls"] == 7
    usage = recorder.snapshot()
    assert usage["llm_calls"] == 14
    assert sum(c["role"] == "evaluator" for c in usage["calls"]) == 7
    assert usage["input_tokens"] == 14 * 11
    assert usage["calls_without_token_usage"] == 0
    assert output["diagnostic_decision"]["reason"] == "evaluator_accepted"


def test_default_workflow_remains_unchanged(monkeypatch, segment):
    output = agent_factory(monkeypatch, capture=False)().invoke(initial_state(segment))
    assert output["final_output"]["anomaly_intervals"] == [[1, 1]]
    assert "diagnostic_passes" not in output


def test_capture_without_reflection_keeps_only_one_pass(monkeypatch, segment):
    recorder = DiagnosticRecorder()
    output = agent_factory(monkeypatch, reflection=False)().invoke(initial_state(segment), recorder.runnable_config())
    assert len(output["diagnostic_passes"]) == 1
    assert output["diagnostic_decision"]["reason"] == "reflection_disabled"
    assert recorder.snapshot()["llm_calls"] == 4


def test_missing_recorder_fails_before_making_api_calls(monkeypatch, segment):
    factory = agent_factory(monkeypatch)
    monkeypatch.setattr(ScriptedChat, "_generate", lambda *args, **kwargs: pytest.fail("No LLM should be invoked"))
    with pytest.raises(ValueError, match="requires DiagnosticRecorder"):
        factory().invoke(initial_state(segment))


def test_controller_stops_without_overwriting_review(monkeypatch, segment):
    recorder = DiagnosticRecorder()
    output = agent_factory(monkeypatch, controller=constant_controller(gain=-0.1))().invoke(
        initial_state(segment), recorder.runnable_config())
    assert len(output["diagnostic_passes"]) == 1
    assert output["redo"] is True
    assert output["diagnostic_decision"]["reason"] == "controller_stop"
    assert output["diagnostic_decision"]["unresolved_review"] is True
    assert output["final_output"]["anomaly_intervals"] == [[0, 0]]


def test_original_hard_limit_still_wins(monkeypatch, segment):
    recorder = DiagnosticRecorder()
    output = agent_factory(monkeypatch, controller=constant_controller(gain=1), always_redo=True)().invoke(
        initial_state(segment), {**recorder.runnable_config(), "recursion_limit": 100})
    assert len(output["diagnostic_passes"]) == 5
    assert output["diagnostic_decision"]["reason"] == "redo_limit"
    assert output["diagnostic_passes"][-1]["features"]["remaining_redos"] == 0


def test_live_features_do_not_depend_on_labels(monkeypatch, segment):
    factory = agent_factory(monkeypatch)
    state = initial_state(segment)
    first = factory().invoke(state, DiagnosticRecorder().runnable_config())
    state["data"]["Label"] = 1 - state["data"]["Label"]
    second = factory().invoke(state, DiagnosticRecorder().runnable_config())
    assert [p["features"] for p in first["diagnostic_passes"]] == [p["features"] for p in second["diagnostic_passes"]]


def test_inference_json_to_training_pairs(monkeypatch, segment, tmp_path):
    config = infer_config(tmp_path)
    task = SegmentTask(0, segment, segment.name, False)
    rc, row = _run_segment_once(task, workflow_factory=agent_factory(monkeypatch), config=config)
    assert rc == 0
    output = tmp_path / "controller-train.jsonl"
    report = build_dataset(config.output_dir, output)
    assert report["examples"] == 1
    example = json.loads(output.read_text())
    assert example["targets"]["gain"] == 1
    assert example["targets"]["extra_llm_calls"] == 7
    assert row["_diagnostics"]["termination"]["unresolved_review"] is False


def test_failed_attempt_traces_survive_retry_writer(monkeypatch, segment, tmp_path):
    task = SegmentTask(0, segment, segment.name, False)
    config = replace(infer_config(tmp_path), max_retries=2)
    result = _execute_tasks([task], workflow_factory=agent_factory(monkeypatch, fail_second_detection=True), config=config)
    assert result[0]["exit_code"] == 2
    record = json.loads((config.output_dir / f"{task.stem}.json").read_text())
    assert record["final_output"]["error"] is True
    assert len(record["previous_diagnostic_attempts"]) == 1
    assert result[0]["diagnostic_usage"]["failed_llm_calls"] == 2
    assert examples_from_trace(record["diagnostics"], [[1, 1]])[0]["targets"]["quality_continue"] == -1


def test_missing_usage_is_explicit_not_invented_zero_tokens():
    from uuid import uuid4
    from langchain_core.outputs import LLMResult
    recorder = DiagnosticRecorder()
    run_id = uuid4()
    recorder.on_chat_model_start({}, [], run_id=run_id)
    recorder.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="ok"))]]), run_id=run_id)
    assert recorder.snapshot()["calls_without_token_usage"] == 1
    assert recorder.snapshot()["calls"][0]["input_tokens"] is None


def test_resume_does_not_mix_baseline_and_controller_results(segment, tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    path = output / f"{segment.name}.json"
    path.write_text(json.dumps({"final_output": {"error": False}}))
    assert build_segment_tasks([segment], output_dir=output, force=False)[0].skip_existing_ok
    assert not build_segment_tasks([segment], output_dir=output, force=False, capture_diagnostics=True)[0].skip_existing_ok
    policy = {"sha256": "model-v1", "cost_weight": 0.1}
    path.write_text(json.dumps({"final_output": {"error": False}, "diagnostics": {"schema_version": 1, "controller": policy}}))
    assert not build_segment_tasks([segment], output_dir=output, force=False)[0].skip_existing_ok
    assert build_segment_tasks([segment], output_dir=output, force=False, capture_diagnostics=True, controller_info=policy)[0].skip_existing_ok
    assert not build_segment_tasks([segment], output_dir=output, force=False, capture_diagnostics=True,
                                   controller_info={**policy, "cost_weight": 0.2})[0].skip_existing_ok


def test_parallel_requests_have_separate_recorders(monkeypatch, segment):
    from concurrent.futures import ThreadPoolExecutor
    factory = agent_factory(monkeypatch)

    def run():
        recorder = DiagnosticRecorder()
        result = factory().invoke(initial_state(segment), recorder.runnable_config())
        return recorder.run_id, result["diagnostic_passes"][-1]["usage"]["llm_calls"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(lambda _: run(), range(2)))
    assert a[0] != b[0]
    assert a[1] == b[1] == 14
