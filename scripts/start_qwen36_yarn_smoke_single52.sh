#!/usr/bin/env bash
# Qwen3.6 YaRN stage-1 smoke profile. It keeps the four capacity controls equal,
# uses separate outputs, and delegates topology/runtime setup to the proven
# single52 homo synchronous launcher. It does not modify Polar configuration.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

export FEAT_YARN=1
export YARN_ROPE_THETA=10000000
export YARN_PARTIAL_ROTARY_FACTOR=0.25
export YARN_FACTOR=4.0
export YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS=262144
export YARN_BETA_FAST=32.0
export YARN_BETA_SLOW=1.0
export YARN_MSCALE=1.0
export YARN_MSCALE_ALL_DIM=0.0
export YARN_CORRECTION_RANGE_ROUND_TO_INT=1

YARN_SMOKE_CONTEXT_LEN="${YARN_SMOKE_CONTEXT_LEN:-262144}"
case "${YARN_SMOKE_CONTEXT_LEN}" in
  ''|*[!0-9]*)
    echo "[yarn-smoke][FATAL] YARN_SMOKE_CONTEXT_LEN must be a positive integer" >&2
    exit 1
    ;;
esac
if [ "${YARN_SMOKE_CONTEXT_LEN}" -le 0 ]; then
  echo "[yarn-smoke][FATAL] YARN_SMOKE_CONTEXT_LEN must be a positive integer" >&2
  exit 1
fi
if [ "${YARN_SMOKE_CONTEXT_LEN}" -ne 262144 ]; then
  echo "[yarn-smoke][FATAL] homo stage-1 launcher only supports YARN_SMOKE_CONTEXT_LEN=262144" >&2
  echo "[yarn-smoke][FATAL] later capacity stages need a dedicated launcher; the sync baseline is immutable" >&2
  exit 1
fi
export SEQ_LENGTH="${YARN_SMOKE_CONTEXT_LEN}"
export MAX_POSITION_EMBEDDINGS="${YARN_SMOKE_CONTEXT_LEN}"
export ROLLOUT_MAX_CONTEXT_LEN="${YARN_SMOKE_CONTEXT_LEN}"
export VLLM_MAX_MODEL_LEN="${YARN_SMOKE_CONTEXT_LEN}"

export HF_CKPT="${HF_CKPT:-/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16}"
export REF_LOAD="${REF_LOAD:-/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t_torch_dist}"
export RUN_ID="${RUN_ID:-qwen36_yarn_smoke_$(date +%Y%m%d-%H%M%S)}"
export SAVE="${SAVE:-/workspace/Qwen3.6-35B-A3B_yarn_smoke}"
export SAVE_HF="${SAVE_HF:-/workspace/Qwen3.6-35B-A3B_yarn_smoke/rollout_{rollout_id}}"
export POLAR_OUTPUT_DIR="${POLAR_OUTPUT_DIR:-output/polar_bridge_yarn_smoke}"
export VIME_SAVE_TIS_LOGPROBS="${VIME_SAVE_TIS_LOGPROBS:-${POLAR_OUTPUT_DIR}/tis_evidence}"
export LOG_FILE="${LOG_FILE:-/mnt/pipeline-data/train_log/train_${RUN_ID}.log}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-2}"

source "${SCRIPT_DIR}/models/qwen3.5-35B-A3B.sh"
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 python3 "${VIME_ROOT}/tools/qwen36_yarn_preflight.py" \
  --model "${HF_CKPT}" \
  --hf-overrides "${QWEN36_VLLM_HF_OVERRIDES}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}"

case "${YARN_PREFLIGHT_ONLY:-0}" in
  1) exit 0 ;;
  0) ;;
  *)
    echo "[yarn-smoke][FATAL] YARN_PREFLIGHT_ONLY must be 0 or 1" >&2
    exit 1
    ;;
esac

exec bash "${SCRIPT_DIR}/start_sync_homo_single52.sh"
