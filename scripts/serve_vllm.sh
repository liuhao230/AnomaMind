#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   MODEL_PATH=/path/to/model bash scripts/serve_vllm.sh

MODEL_PATH="${1:-${MODEL_PATH:-}}"
: "${MODEL_PATH:?set MODEL_PATH to the HF checkpoint directory (or pass as first argument)}"
ASCEND_DEVICES="${ASCEND_DEVICES:-0,1,2,3}"
PORT="${PORT:-8098}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-detector}"
VLLM_BIN="${VLLM_BIN:-vllm}"
CANN_ENV_BASH="${CANN_ENV_BASH:-}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_DEVICES}"

if [[ -n "${CANN_ENV_BASH}" && -f "${CANN_ENV_BASH}" ]]; then
    # Ensure pyACL / acl is visible to vllm-ascend workers.
    # shellcheck source=/dev/null
    set +e
    set +u
    source "${CANN_ENV_BASH}" >/dev/null 2>&1 || true
    set -u
    set -e
fi

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --served-model-name "${SERVED_MODEL_NAME}"
