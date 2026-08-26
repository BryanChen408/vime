#!/bin/bash
# 单机 NPU 纯评测底座。入口脚本只提供权重、数据和评测规模。

set -euxo pipefail
: "${MODEL_TAG}" "${HF_CKPT}" "${DATASET_TAG}" "${EVAL_DATA}" \
  "${OPERATOR_TASKS_DIR}" "${ROLLOUT_BATCH_SIZE}" "${ROLLOUT_MAX_ACTIVE_SESSIONS}"

BASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${BASE_DIR}/../.." &>/dev/null && pwd)"
cd "${VIME_ROOT}"

[ -d "${HF_CKPT}" ] || { echo "HF checkpoint not found: ${HF_CKPT}" >&2; exit 1; }
[ -f "${HF_CKPT}/config.json" ] || {
  echo "Checkpoint is not an HF directory (missing config.json): ${HF_CKPT}" >&2
  echo "Convert the Megatron iter_* checkpoint to HF format before evaluation." >&2
  exit 1
}
[ -f "${EVAL_DATA}" ] || { echo "Evaluation data not found: ${EVAL_DATA}" >&2; exit 1; }
[ -d "${OPERATOR_TASKS_DIR}" ] || { echo "Operator tasks directory not found: ${OPERATOR_TASKS_DIR}" >&2; exit 1; }

# 第三方环境脚本直接引用未定义变量，只在 source 期间关闭 nounset。
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

# .66 单机拓扑：0-3 留给 Polar，4-15 用于三个 TP4 vLLM engine。
LOCAL_IP=80.48.5.66
SOCKET_IFNAME=enp91s0f3
NPUS_PER_NODE=12
ROLLOUT_NUM_GPUS=12
ROLLOUT_NUM_GPUS_PER_ENGINE=4
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7,8,9,10,11,12,13,14,15

POLAR_URL=http://${LOCAL_IP}:8080
RAY_PORT=6460
RAY_DASHBOARD_PORT=8290
RAY_TEMP_DIR=/home/docker/vime_evaluation/ray
OUTPUT_ROOT=/home/docker/vime_evaluation/output
LOG_ROOT=/home/docker/vime_evaluation/logs
RUN_ID=${MODEL_TAG}_${DATASET_TAG}_$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR=${OUTPUT_ROOT}/${RUN_ID}
LOG_FILE=${LOG_ROOT}/${RUN_ID}.log
mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}"

NUM_ROLLOUT=1
N_SAMPLES_PER_PROMPT=1
N_SAMPLES_PER_EVAL_PROMPT=3
ROLLOUT_MAX_ASYNC_LEVEL=1
ROLLOUT_MAX_RESPONSE_LEN=32768
ROLLOUT_MAX_CONTEXT_LEN=131072
VLLM_MAX_MODEL_LEN=131072
VLLM_MAX_NUM_SEQS=96

export PYTHONUNBUFFERED=1
export PYTHONPATH="/workspace/vllm:/workspace/vllm-ascend:/workspace/Megatron-LM:${VIME_ROOT}:${PYTHONPATH:-}"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH:-}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TASK_QUEUE_ENABLE=0
export TORCHDYNAMO_DISABLE=1
export CPU_AFFINITY_CONF=1
export VLLM_ASCEND_ENABLE_NZ=0
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=1
export RAY_memory_usage_threshold=0.98
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HCCL_CONNECT_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=2400
export HCCL_BUFFSIZE=512
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_SOCKET_FAMILY=AF_INET
export HCCL_WHITELIST_DISABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export POLAR_KEEP_SESSION_DIR=1
export POLAR_ANTHROPIC_DEFAULT_MAX_TOKENS=12288

export no_proxy="127.0.0.1,localhost,${LOCAL_IP}${no_proxy:+,${no_proxy}}"
export NO_PROXY="${no_proxy}"
export HCCL_SOCKET_IFNAME="${SOCKET_IFNAME}"
export GLOO_SOCKET_IFNAME="${SOCKET_IFNAME}"

TOPO_ARGS=(
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --num-gpus-per-node "${NPUS_PER_NODE}"
)

ROLLOUT_ARGS=(
  --rollout-function-path vime_bridge.rollout.generate_rollout_polar_async
  --prompt-data "${EVAL_DATA}"
  --eval-prompt-data "${DATASET_TAG}" "${EVAL_DATA}"
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --reward-key score
  --custom-reward-post-process-path vime_bridge.reward_post_process.post_process_rewards
  --rollout-shuffle
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
  --rollout-temperature 0.7
  --rollout-seed 42
  --save-debug-rollout-data "${OUTPUT_DIR}/debug/{rollout_id}.pt"
)

POLAR_ARGS=(
  --polar-url "${POLAR_URL}"
  --polar-run-id "${RUN_ID}"
  --polar-reward-key score
  --polar-task-id-template '{args.polar_run_id}-polar-op-{rollout_id}-{sample.group_index}'
  --operator-tasks-dir "${OPERATOR_TASKS_DIR}"
  --rollout-max-async-level "${ROLLOUT_MAX_ASYNC_LEVEL}"
  --rollout-request-timeout 8000
  --rollout-scheduler-mode session_pool
  --rollout-max-active-sessions "${ROLLOUT_MAX_ACTIVE_SESSIONS}"
  --rollout-release-on-postrun
  --rollout-min-complete-accept-fraction 0.8
)

VLLM_ARGS=(
  --vllm-tool-call-parser qwen3_coder
  --vllm-enable-auto-tool-choice
  --vllm-reasoning-parser qwen3
  --qwen-gdn-backend npu
  --model-name qwen3_5moeforconditionalgeneration
  --vllm-hf-overrides '{"architectures":["Qwen3_5MoeForConditionalGeneration"]}'
  --vllm-router-port 8001
  --vllm-gpu-memory-utilization 0.8
  --vllm-max-num-seqs "${VLLM_MAX_NUM_SEQS}"
  --vllm-max-model-len "${VLLM_MAX_MODEL_LEN}"
  --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
  --vllm-enable-prefix-caching
  --vllm-enable-chunked-prefill
  --vllm-additional-config '{"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":true}}'
)

# 清理本评测残留；不触碰 :8080 上的 Polar 服务。
pkill -9 -f 'VLL[M]::' 2>/dev/null || true
pkill -9 -f 'vime.ray.lb_proxy' 2>/dev/null || true
ray stop --force
rm -rf "${RAY_TEMP_DIR}"

ray start --head --port "${RAY_PORT}" --dashboard-host=0.0.0.0 \
  --node-ip-address="${LOCAL_IP}" --dashboard-port="${RAY_DASHBOARD_PORT}" \
  --num-gpus="${NPUS_PER_NODE}" --resources='{"NPU": '"${NPUS_PER_NODE}"'}' \
  --temp-dir="${RAY_TEMP_DIR}" --object-store-memory=50000000000 --disable-usage-stats

# Ray 启动后取消设备限制，VLLMEngine actor 再由 Ray 分配逻辑设备。
unset ASCEND_RT_VISIBLE_DEVICES HCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME
export RAY_ADDRESS="${LOCAL_IP}:${RAY_PORT}"

echo "[evaluation] model=${HF_CKPT} data=${EVAL_DATA} output=${OUTPUT_DIR}"
python3 experiments/evaluation/evaluation.py \
  --debug-rollout-only \
  --rollout-lb-proxy \
  "${TOPO_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${POLAR_ARGS[@]}" \
  "${VLLM_ARGS[@]}" --hf-checkpoint "${HF_CKPT}" \
  2>&1 | tee "${LOG_FILE}"
