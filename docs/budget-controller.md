# Diagnostic traces and a cost-aware redo controller

This optional experiment learns whether one additional Actor → Detector → Evaluator
pass is worth its extra LLM calls. The Detector and other LLMs remain frozen.
It does not train Qwen or replace the Evaluator's quality assessment.

The original workflow remains the default. With a controller, the Evaluator still
runs first. Only a `redo: true` result that is within the original redo limit is
eligible for the controller. A rejected continuation ends with
`diagnostics.termination.reason = "controller_stop"` and
`unresolved_review = true`; it is not reported as Evaluator acceptance.

## 1. Collect the baseline

Deploy an existing Detector checkpoint with the serving instructions in the README
and configure the four role APIs. Keep their checkpoints, prompts, temperatures,
and tool settings fixed throughout the experiment. First hold out original
series/services for validation and testing, then generate windows. Do not randomly
split adjacent windows or diagnostic passes between datasets.

Run full inference on the training split, with reflection enabled:

```bash
python -m AnomaMind.infer.infer data/controller/processed/train \
  --output-dir data/controller/result/train \
  --capture-diagnostics --workers 1 --max-retries 1
```

Repeat on disjoint validation and test splits with their own result directories.
Do not pass `--no-reflection`: a one-pass result has no observed continuation.
Existing successful outputs without diagnostic traces are rerun when capture is
requested. Outputs produced by a different controller artifact or cost weight are
also rerun. Changing other model/API settings still requires a separate output
directory or `--force`.

Each result's `diagnostics` object contains:

- A schema version, run ID, source-series ID, segment ID, and observed indices.
- Every completed pass before its live messages are cleared: parsed predictions,
  review feedback, numeric controller features, Actor/Detector transcripts, the
  route decision, and cumulative usage. These records can be inspected alongside
  the input segment; they do not embed the input CSV or form a LangGraph checkpoint.
- LLM invocation counts, known token usage, call errors, and elapsed wall time.
  Calls are attributed to graph roles, including the Evaluator's parallel summary
  calls. Unknown token usage is explicit (`calls_without_token_usage` and null
  per-call counts); a total of zero with missing usage is not evidence of free calls.

`previous_diagnostic_attempts` preserves earlier attempts when the batch runner
retries a failed window. The fresh-run summary's per-segment `diagnostic_usage`
sums those attempts. For accounting across resumed runs, sum each result's current
and previous attempt usage. Internal retries inside an API SDK are not individually
observable; API billing remains authoritative. Per-call durations can overlap
during compression; use `usage.elapsed_seconds` for graph wall time.

Inspect the redo rate and the distribution of additional-pass gains before spending
on a larger dataset. If redos are rare or almost always beneficial, this particular
decision point may offer little savings.

## 2. Build one-step supervision offline

```bash
python -m AnomaMind.train.prepare_controller_data \
  data/controller/result/train --output data/controller/train.jsonl
python -m AnomaMind.train.prepare_controller_data \
  data/controller/result/validation --output data/controller/validation.jsonl
```

The builder pairs a reviewed state with its actually observed next pass, never
across retries or gaps. Stopping uses the existing prediction and requires no extra
LLM request. The targets are:

```text
gain = quality(next_prediction) - quality(current_prediction)
extra_llm_calls = next_cumulative_calls - current_cumulative_calls
```

Quality uses point F1 on anomalous windows and `1 / (false_positive_points + 1)`
on normal windows. Only observed indices count, including for out-of-range model
predictions. A failed/unparseable pass scores -1; negative gains are retained.
This uses the paper's detection reward component with the inference parser, not
its stricter RL response-format reward. Labels are read only by the offline builder,
from the saved ground truth or the original segment when debug artifacts were disabled.

Only actually continued states yield examples. Budget-limited or accepted states
without a next pass are not assigned artificial zero-gain targets. Baseline traces
therefore describe the original redo policy's observed states; they are not
exhaustive counterfactual exploration. More repeated continuations can be collected
in independent runs with distinct run IDs. An interrupted graph may have usage but
no completed pair; its cost remains recorded.

## 3. Fit the small model

```bash
python -m AnomaMind.train.train_controller data/controller/train.jsonl \
  --validation-data data/controller/validation.jsonl \
  --output models/redo-controller.json
```

Two standardized ridge regressions predict quality gain and extra LLM calls.
Training requires NumPy and a CPU, not a GPU or an LLM endpoint. The validation
series must be disjoint from training series; overlap is rejected. The command
reports training fit and, when supplied, validation MSE. These diagnostics are not
claims about end-to-end detection performance.

The JSON artifact has a strict version, cost unit, feature order, normalization
statistics, and coefficients. Inference uses only Python arithmetic. Features
include pass depth, remaining redos, candidate/prediction counts, coverage,
confidence, evidence/error counts, observed-value statistics, and calls already
spent. Ground-truth labels and observed future quality/cost are never inputs.

## 4. Run the controller and compare

```bash
python -m AnomaMind.infer.infer data/controller/processed/test \
  --output-dir data/controller/result/test-controller \
  --controller-model models/redo-controller.json \
  --controller-cost-weight 0.01
```

The cost weight is illustrative: choose it on validation data, then hold it fixed
on test data. A continuation is admitted when
`predicted_gain > cost_weight * predicted_extra_llm_calls`, subject to the original
hard redo limit. This is a quality-versus-call-count policy, not a strict monetary,
token, or latency budget. It cannot interrupt an in-flight LLM call or guarantee a
wall-time ceiling. Controller overhead is small but should be included in real
end-to-end measurements. Diagnostics are enabled automatically and record the
controller artifact hash, cost weight, and each decision.

Compare against the original workflow, uniformly reduced redo limits, and simple
rules under identical model, split, concurrency, and hardware settings. Report both
quality at equal cost and cost at similar quality. Include normal-only series,
failed windows, and unresolved reviews rather than dropping them. The existing
evaluator can include normal-only series:

```bash
python -m AnomaMind.eval.evaluate_results data/controller/result/test-controller \
  --no-exclude-no-anomaly-series
```

The unchanged workflow's `max_eval_redo_iterations` constructor argument uses the
original zero-based condition (`iteration_count > limit`): default 3 allows four
redos / five Detector passes. Uniform-limit baselines must account for this.
This first version learns a one-step gain, not the value of an arbitrary sequence
of future tool actions. No pretrained controller or measured savings are shipped.

## Python API and CPU checks

When constructing a graph directly, pass a fresh recorder per invocation:

```python
from AnomaMind.core.diagnostics import DiagnosticRecorder
from AnomaMind.workflows.AnomaAgent import AnomaAgent

recorder = DiagnosticRecorder()
graph = AnomaAgent(capture_diagnostics=True).build_workflow()
result = graph.invoke(state, {**recorder.runnable_config(), "recursion_limit": 200})
```

The batch CLI handles this automatically, including one recorder per parallel
request. Do not share a recorder between graph invocations. Restore the original
workflow by omitting `--controller-model`; omit `--capture-diagnostics` as well to
disable pass snapshots. Diagnostic recording does not make extra model calls.

The tests use real LangGraph execution and deterministic local chat models; they
need neither cloud credentials nor GPU packages:

```bash
python -m pip install -e . pytest numpy pandas langgraph langchain-openai python-dotenv
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```
