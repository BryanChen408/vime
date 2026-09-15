#!/usr/bin/env bash
# Stage 8: cross the original 262144-token boundary at a 270336-token capacity.
# This is deliberately separate from the immutable 262144 sync/homo baseline.
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

readonly STAGE8_CONTEXT_LEN=270336
export SEQ_LENGTH=${STAGE8_CONTEXT_LEN}
export MAX_POSITION_EMBEDDINGS=${STAGE8_CONTEXT_LEN}
export ROLLOUT_MAX_CONTEXT_LEN=${STAGE8_CONTEXT_LEN}
export VLLM_MAX_MODEL_LEN=${STAGE8_CONTEXT_LEN}
export MAX_TOKENS_PER_GPU=33792

export HF_CKPT="${HF_CKPT:-/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16}"
export REF_LOAD="${REF_LOAD:-/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t_torch_dist}"
export RUN_ID="${RUN_ID:-qwen36_yarn_step8_270k_$(date +%Y%m%d_%H%M%S)}"
export SAVE="${SAVE:-/workspace/Qwen3.6-35B-A3B_${RUN_ID}}"
export SAVE_HF="${SAVE_HF:-${SAVE}/rollout_{rollout_id}}"
export POLAR_OUTPUT_DIR="${POLAR_OUTPUT_DIR:-output/polar_bridge_${RUN_ID}}"
export VIME_SAVE_TIS_LOGPROBS="${VIME_SAVE_TIS_LOGPROBS:-${POLAR_OUTPUT_DIR}/tis_evidence}"
export LOG_FILE="${LOG_FILE:-/mnt/pipeline-data/train_log/train_${RUN_ID}.log}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_y8_270k}"

source "${SCRIPT_DIR}/models/qwen3.5-35B-A3B.sh"
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 python3 "${VIME_ROOT}/tools/qwen36_yarn_preflight.py" \
  --model "${HF_CKPT}" \
  --hf-overrides "${QWEN36_VLLM_HF_OVERRIDES}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}"

_LB_PORT="${VLLM_ROUTER_PORT:-8001}"
pkill -f "vime\.ray\.lb_proxy .*--port ${_LB_PORT}" 2>/dev/null || true
pkill -f "dp_load_balance_proxy_server.*${_LB_PORT}" 2>/dev/null || true
for _ in 1 2 3 4 5; do
  ss -tln 2>/dev/null | grep -q ":${_LB_PORT} " || break
  sleep 1
done
if ss -tln 2>/dev/null | grep -q ":${_LB_PORT} "; then
  echo "[stage8][FATAL] port ${_LB_PORT} is still occupied" >&2
  ss -tlnp 2>/dev/null | grep ":${_LB_PORT} " >&2
  exit 1
fi

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
CURRENT_IP=80.48.5.52 MASTER_ADDR=80.48.5.52 NNODES=1 NPUS_PER_NODE=16 SOCKET_IFNAME=ens1f3 \
ACTOR_NUM_NODES=1 \
ACTOR_NUM_GPUS_PER_NODE=16 \
TRAIN_ENTRY=train.py \
FEAT_OFFLOAD=1 \
FEAT_SYNC_ROLLOUT=1 \
RESOURCE_LAYOUT="${SCRIPT_DIR}/resource_layout.single52_homo_colocate.yaml" \
ROLLOUT_NODE_IP=80.48.5.52 \
ROLLOUT_NUM_GPUS=12 \
ROLLOUT_NUM_GPUS_PER_ENGINE=2 \
FEAT_PD_DISAGG=0 \
VLLM_SERVED_MODEL_NAME=/home/docker/Qwen3.6-35B-A3B \
VLLM_GPU_MEM_UTIL=0.70 \
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU} \
SEQ_LENGTH=${SEQ_LENGTH} \
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN} \
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN} \
VIME_MEM_PROBE=1 \
RAY_memory_usage_threshold=0.95 \
no_proxy=127.0.0.1,localhost,80.48.5.52,.huawei.com,local,.local \
NO_PROXY=127.0.0.1,localhost,80.48.5.52,.huawei.com,local,.local \
TP=2 PP=1 CP=8 EP=8 \
POLAR_TRAJECTORY_PG_FLOOR=0.05 \
POLAR_ROLLOUT_URL=http://80.48.5.52:8080 \
VLLM_ROUTER_PORT=8001 \
FEAT_TRAIN_EXPANDABLE=1 \
VIME_EMPTY_CACHE_PER_STEP=1 \
TRANSFORMERS_VERBOSITY=error \
HCCL_INTER_HCCS_DISABLE=false \
HCCL_INTRA_ROCE_ENABLE=1 \
HCCL_INTRA_PCIE_ENABLE=0 \
HCCL_BUFFSIZE=512 \
HCCL_HOST_SOCKET_PORT_RANGE=60000-60255 \
HCCL_NPU_SOCKET_PORT_RANGE=61000-61255 \
ROLLOUT_BATCH_SIZE=2 N_SAMPLES_PER_PROMPT=2 GLOBAL_BATCH_SIZE=4 NUM_ROLLOUT=${NUM_ROLLOUT} \
FEAT_DP_EXTERNAL_LB=0 FEAT_BALANCE_SCHED=0 FEAT_LB_PROXY=1 FEAT_CROSS_DP_EP=0 \
FEAT_ROLLOUT_EP=0 FEAT_FLASHCOMM1=0 FEAT_PREFIX_CACHE=1 FEAT_MULTISTREAM_SHARED_EXPERT=1 \
FEAT_STATIC_KERNEL=0 FEAT_HCCL_AIV=1 \
OPERATOR_DATA_ROOT=/home/docker/datasets/op_tasks/op_assets_cudallm_filtered189 \
OPERATOR_TASK_JSONL=/home/docker/datasets/op_tasks/op_assets_cudallm_filtered189/operator_tasks.16.jsonl \
PROFILE_TRAIN=0 \
bash "${SCRIPT_DIR}/run-qwen36-35b-polar-multi-pd.sh"
