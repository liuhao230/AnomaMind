set -euo pipefail
set -x

_TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_TRAIN_DIR}/.." && pwd)"

# ── Required paths (no machine-specific defaults) ─────────────────────────
: "${TRAIN_PARQUET:?set TRAIN_PARQUET to the RL train.parquet path}"
: "${VAL_PARQUET:?set VAL_PARQUET to the RL val.parquet path}"
: "${MODEL_PATH:?set MODEL_PATH to the SFT/HF checkpoint directory}"

# ── Optional paths & runtime ────────────────────────────────────────────────
TRAIN_REWARD_FILE="${TRAIN_REWARD_FILE:-${_REPO_ROOT}/src/AnomaMind/core/reward.py}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
MONITOR_ROOT="${MONITOR_ROOT:-${_REPO_ROOT}/logs/monitor_runs}"
MONITOR_RUN_DIR="${MONITOR_RUN_DIR:-${MONITOR_ROOT}/${RUN_ID}}"
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-${MONITOR_RUN_DIR}/rollout}"
VALIDATION_LOG_DIR="${VALIDATION_LOG_DIR:-${MONITOR_RUN_DIR}/validation}"
REWARD_TRACE_DIR="${REWARD_TRACE_DIR:-${MONITOR_RUN_DIR}/reward_trace}"
LOG_FILE="${LOG_FILE:-${MONITOR_RUN_DIR}/train.log}"
CKPT_DIR="${CKPT_DIR:-${_REPO_ROOT}/checkpoints/${RUN_ID}}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-900}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DISABLE_SERVICE="${WANDB_DISABLE_SERVICE:-true}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::ResourceWarning}"
export PYTHONUNBUFFERED=1
export VERL_REWARD_LOG_DIR="${VERL_REWARD_LOG_DIR:-${REWARD_TRACE_DIR}}"
export VERL_REWARD_LOG_MAX_CHARS="${VERL_REWARD_LOG_MAX_CHARS:-60000}"

for required_path in "$TRAIN_PARQUET" "$VAL_PARQUET" "$MODEL_PATH" "$TRAIN_REWARD_FILE"; do
  if [[ ! -e "$required_path" ]]; then
    echo "missing required path: $required_path" >&2
    exit 1
  fi
done

cleanup() {
  set +e
  pkill -P $$ >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

mkdir -p "$ROLLOUT_LOG_DIR" "$VALIDATION_LOG_DIR" "$REWARD_TRACE_DIR" "$CKPT_DIR"

{
  echo "train_dir=${_TRAIN_DIR}"
  echo "repo_root=${_REPO_ROOT}"
  echo "run_id=$RUN_ID"
  echo "train_parquet=$TRAIN_PARQUET"
  echo "val_parquet=$VAL_PARQUET"
  echo "model_path=$MODEL_PATH"
  echo "ckpt_dir=$CKPT_DIR"
  echo "monitor_run_dir=$MONITOR_RUN_DIR"
  echo "rollout_log_dir=$ROLLOUT_LOG_DIR"
  echo "validation_log_dir=$VALIDATION_LOG_DIR"
  echo "reward_trace_dir=$VERL_REWARD_LOG_DIR"
  echo "train_reward_file=$TRAIN_REWARD_FILE"
  echo "start_time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$MONITOR_RUN_DIR/run_metadata.env"

echo "Monitor run dir: $MONITOR_RUN_DIR"
python3 -m verl.trainer.main_ppo \
  +ray_kwargs.ray_init.address=local \
  algorithm.adv_estimator=grpo \
  trainer.project_name=anoma_detector_rl \
  trainer.experiment_name=run \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  data.train_batch_size=16 \
  data.max_prompt_length=3000 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.top_k=150 \
  actor_rollout_ref.rollout.top_p=0.95 \
  trainer.max_actor_ckpt_to_keep=3 \
  actor_rollout_ref.rollout.enforce_eager=True \
  algorithm.kl_ctrl.kl_coef=0.0 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.rollout.max_model_len=5100 \
  actor_rollout_ref.rollout.max_num_seqs=32 \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.actor.grad_clip=2.0 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8096 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=240 \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.rollout_data_dir="${ROLLOUT_LOG_DIR}" \
  trainer.validation_data_dir="${VALIDATION_LOG_DIR}" \
  trainer.logger='["console"]' \
  trainer.total_epochs=1 \
  custom_reward_function.path="${TRAIN_REWARD_FILE}" \
  actor_rollout_ref.actor.checkpoint.save_contents="['hf_model']" \
  "$@" \
  2>&1 | tee "$LOG_FILE"
