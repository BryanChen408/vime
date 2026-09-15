#!/usr/bin/env bash
# YaRN validation step 5: real NPU forward/backward at CP=1 with fused RoPE off.
# This is a training-only replay; --load-debug-rollout-data prevents vLLM/Polar startup.
set -e

VIME_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MEGATRON_ROOT=${MEGATRON_ROOT:-/workspace/Megatron-LM}
MIND_SPEED_ROOT=${MIND_SPEED_ROOT:-/workspace/MindSpeed}
VALIDATION_STEP=${VALIDATION_STEP:-step5}
CP_SIZE=${CP_SIZE:-1}
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
EP_SIZE=${EP_SIZE:-8}
ACTOR_NUM_GPUS=${ACTOR_NUM_GPUS:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
RUN_ID=${RUN_ID:-qwen36_yarn_${VALIDATION_STEP}_cp${CP_SIZE}_$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-${VIME_ROOT}/output/${RUN_ID}}
LOG_FILE=${LOG_FILE:-/mnt/pipeline-data/train_log/train_${RUN_ID}.log}
ROLLOUT_DATA=${ROLLOUT_DATA:-${VIME_ROOT}/output/polar_bridge_qwen36_yarn_s1_vllmfix_20260914_0924/vime_debug_rollout_qwen36_yarn_step5_cp1_8samples_1k.pt}

HF_CKPT=${HF_CKPT:-/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16}
REF_LOAD=${REF_LOAD:-/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t_torch_dist}
SEQ_LENGTH=${SEQ_LENGTH:-262144}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-1024}

mkdir -p "${RUN_ROOT}" "$(dirname -- "${LOG_FILE}")"
test -f "${ROLLOUT_DATA}" || { echo "[step5][FATAL] rollout data missing: ${ROLLOUT_DATA}" >&2; exit 1; }
test -d "${HF_CKPT}" || { echo "[step5][FATAL] HF checkpoint missing: ${HF_CKPT}" >&2; exit 1; }
test -d "${REF_LOAD}" || { echo "[step5][FATAL] torch-dist checkpoint missing: ${REF_LOAD}" >&2; exit 1; }

# Match the proven NPU/GDN runtime used by the corrected YaRN smoke run.
ASCEND_ROOT=${ASCEND_ROOT:-/usr/local/Ascend}
CANN_ROOT=${CANN_ROOT:-${ASCEND_ROOT}/cann}
CANN_TOOLKIT_ROOT="${ASCEND_ROOT}/ascend-toolkit/cann-9.0.0"
CANN_PYTHON_SITE_PACKAGES="${CANN_ROOT}/python/site-packages"
CANN_TBE_DIR="${CANN_ROOT}/opp/built-in/op_impl/ai_core/tbe"
# CANN 9.2's set_env.sh enables errexit internally. This host currently has
# an empty /etc/ascend_install.info, so its optional driver-path grep aborts
# the source halfway through. Use it when that field exists; otherwise set
# the same runtime paths explicitly below.
if grep -qi '^driver_install_path_param=' /etc/ascend_install.info 2>/dev/null; then
  source "${CANN_ROOT}/set_env.sh"
else
  echo "[${VALIDATION_STEP}] skip CANN set_env.sh: driver_install_path_param is absent"
fi
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_OPP_PATH="${CANN_ROOT}/opp"
export ASCEND_AICPU_PATH="${CANN_ROOT}"
export ASCEND_HOME_PATH="${CANN_ROOT}"
export ASCEND_TOOLKIT_HOME="${CANN_ROOT}"
export TOOLCHAIN_HOME="${CANN_ROOT}/toolkit"
export PATH="${CANN_ROOT}/bin:${PATH:-}"
export LD_LIBRARY_PATH="${CANN_ROOT}/lib64:${CANN_ROOT}/lib64/plugin/opskernel:${CANN_ROOT}/lib64/plugin/nnengine:${CANN_ROOT}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-/usr/local/Ascend/cann-9.0.0/opp/vendors/fla_npu_transformer}
test -f "${ASCEND_CUSTOM_OPP_PATH}/op_api/lib/libcust_opapi.so" || {
  echo "[step5][FATAL] GDN custom op missing: ${ASCEND_CUSTOM_OPP_PATH}/op_api/lib/libcust_opapi.so" >&2
  exit 1
}

export SLIME_SCRIPT_TRAIN_BACKEND=megatron
export PYTHONPATH="/usr/local/lib/python3.11/site-packages:${MEGATRON_ROOT}:${VIME_ROOT}:${MIND_SPEED_ROOT}:${CANN_PYTHON_SITE_PACKAGES}:${CANN_TBE_DIR}:${CANN_TOOLKIT_ROOT}/python/site-packages:${PYTHONPATH:-}"
# Polar/its inference processes may occupy physical cards 0-3; use the free
# eight-card slice by default. Ray sees these as logical devices 0-7.
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7,8,9,10,11}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=1
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-2400}
export HCCL_SOCKET_FAMILY=${HCCL_SOCKET_FAMILY:-AF_INET}
export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}
export HCCL_INTRA_ROCE_ENABLE=${HCCL_INTRA_ROCE_ENABLE:-1}
export HCCL_INTRA_PCIE_ENABLE=${HCCL_INTRA_PCIE_ENABLE:-0}
export HCCL_INTER_HCCS_DISABLE=${HCCL_INTER_HCCS_DISABLE:-true}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export TORCHDYNAMO_DISABLE=1
export QWEN36_CP_MODE=${QWEN36_CP_MODE:-ulysses}
export QWEN36_CAUSAL_CONV1D_IMPL=${QWEN36_CAUSAL_CONV1D_IMPL:-triton}
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_VERSION=0.23.0
export WANDB_MODE=disabled
NODE_IP="$(hostname -I | awk '{print $1}')"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy="127.0.0.1,localhost,${NODE_IP}"
export NO_PROXY="${no_proxy}"

export VIME_SAVE_TIS_LOGPROBS="${RUN_ROOT}/tis_evidence"
mkdir -p "${VIME_SAVE_TIS_LOGPROBS}"

export FEAT_YARN=${FEAT_YARN:-1}
export YARN_ROPE_THETA=10000000
export YARN_PARTIAL_ROTARY_FACTOR=0.25
export YARN_FACTOR=4.0
export YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS=262144
export YARN_BETA_FAST=32.0
export YARN_BETA_SLOW=1.0
export YARN_MSCALE=1.0
export YARN_MSCALE_ALL_DIM=0.0
export YARN_CORRECTION_RANGE_ROUND_TO_INT=1
source "${VIME_ROOT}/scripts/models/qwen3.5-35B-A3B.sh"

RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/ray_qwen36_yarn_step5_cp1}
RAY_PORT=${RAY_PORT:-6461}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8291}
ray stop --force >/dev/null 2>&1 || true
rm -rf "${RAY_TEMP_DIR}"
ray start --head --port "${RAY_PORT}" --dashboard-port "${RAY_DASHBOARD_PORT}" \
  --node-ip-address "${NODE_IP}" --num-gpus "${ACTOR_NUM_GPUS}" \
  --resources="{\"NPU\": ${ACTOR_NUM_GPUS}}" \
  --temp-dir="${RAY_TEMP_DIR}" --dashboard-host=0.0.0.0 --disable-usage-stats
export RAY_ADDRESS="${NODE_IP}:${RAY_PORT}"

cd "${VIME_ROOT}"
echo "[${VALIDATION_STEP}] run=${RUN_ID} yarn=${FEAT_YARN} cp=${CP_SIZE} tp=${TP_SIZE} pp=${PP_SIZE} ep=${EP_SIZE} fused_rope=off seq=${SEQ_LENGTH}"
echo "[${VALIDATION_STEP}] rollout_data=${ROLLOUT_DATA}"

python3 "${VIME_ROOT}/train.py" \
  --train-backend megatron \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS}" \
  --rollout-num-gpus "${ACTOR_NUM_GPUS}" \
  --rollout-num-gpus-per-engine "${ACTOR_NUM_GPUS}" \
  "${MODEL_ARGS[@]}" \
  --no-rope-fusion \
  --qwen-gdn-backend npu \
  --hf-checkpoint "${HF_CKPT}" \
  --ref-load "${REF_LOAD}" \
  --megatron-to-hf-mode raw \
  --prompt-data /home/docker/datasets/op_tasks/op_assets_cudallm_filtered189/operator_tasks.16.jsonl \
  --input-key prompt \
  --label-key label \
  --metadata-key metadata \
  --reward-key score \
  --custom-reward-post-process-path vime_bridge.reward_post_process.post_process_rewards \
  --rollout-function-path vime_bridge.rollout.generate_rollout_polar_sync \
  --eval-function-path vime_bridge.rollout.generate_rollout_polar_sync \
  --num-rollout 1 \
  --rollout-batch-size 4 \
  --n-samples-per-prompt 2 \
  --rollout-max-response-len 32768 \
  --rollout-max-context-len "${SEQ_LENGTH}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --use-dynamic-global-batch-size \
  --advantage-estimator grpo \
  --entropy-coef 0.001 \
  --eps-clip 0.2 \
  --use-tis \
  --optimizer adam \
  --lr 3e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --optimizer-cpu-offload \
  --overlap-cpu-optimizer-d2h-h2d \
  --use-precision-aware-optimizer \
  --tensor-model-parallel-size "${TP_SIZE}" \
  --pipeline-model-parallel-size "${PP_SIZE}" \
  --context-parallel-size "${CP_SIZE}" \
  --expert-model-parallel-size "${EP_SIZE}" \
  --expert-tensor-parallel-size 1 \
  --sequence-parallel \
  --chunked-lm-head \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
  --log-probs-chunk-size 1024 \
  --seq-length "${SEQ_LENGTH}" \
  --max-position-embeddings "${SEQ_LENGTH}" \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --use-flash-attn \
  --no-gradient-accumulation-fusion \
  --offload-train \
  --no-offload-rollout \
  --polar-url http://127.0.0.1:8080 \
  --polar-run-id "${RUN_ID}" \
  --polar-reward-key score \
  --load-debug-rollout-data "${ROLLOUT_DATA}" \
  --save-debug-train-data "${RUN_ROOT}/vime_debug_train_${RUN_ID}_rollout_{rollout_id}_{rank}.pt" \
  --no-save-optim \
  --optimization-level 0 \
  2>&1 | tee "${LOG_FILE}"

echo "VALIDATION_RUN_ROOT=${RUN_ROOT}"
echo "VALIDATION_LOG_FILE=${LOG_FILE}"
