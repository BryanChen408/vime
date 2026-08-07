#!/bin/bash
# vime + polar 算子 RL 启动(qwen3.6-35B-A3B / NPU) — 训练侧 PD (Prefill/Decode) + Mooncake。
# 卡位:rollout 共占 16 张卡,2P2D — Prefill×2(各 tp4)占 8 卡 + Decode×2(各 tp4)占 8 卡;训练侧不启动。
# Polar 侧请使用 profile.vime-pd.yaml, 将推理入口指向 :8001 (Mooncake PD proxy), 而不是 :18000。
set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
cd "${VIME_ROOT}"

ASCEND_ROOT=${ASCEND_ROOT:-/usr/local/Ascend}
CANN_ROOT=${CANN_ROOT:-${ASCEND_ROOT}/cann}
CANN_TOOLKIT_ROOT="${ASCEND_ROOT}/ascend-toolkit/cann-9.0.0"
CANN_BIN_DIR="${CANN_ROOT}/bin"
CANN_LIB_DIR="${CANN_ROOT}/lib64"
CANN_PYTHON_SITE_PACKAGES="${CANN_ROOT}/python/site-packages"
CANN_TBE_DIR="${CANN_ROOT}/opp/built-in/op_impl/ai_core/tbe"
CANN_CUSTOM_TRANSFORMER_LIB="${CANN_ROOT}/opp/vendors/custom_transformer/op_api/lib"
CANN_HCCL_LIB_DIR="$(find /usr/local/Ascend -path '*/x86_64-linux/lib64/libhccl.so' -print -quit 2>/dev/null | xargs -r dirname)"

source "${CANN_ROOT}/set_env.sh"
source /usr/local/Ascend/nnal/atb/set_env.sh

# Force override ASCEND paths to use CANN 9.2.0 (where we copied ops_legacy)
# Must be AFTER set_env.sh to override any values it sets
export ASCEND_OPP_PATH="${CANN_ROOT}/opp"
export ASCEND_HOME_PATH="${CANN_ROOT}"
export ASCEND_TOOLKIT_HOME="${CANN_ROOT}"

echo "[启动脚本] Forced ASCEND paths to CANN 9.2.0:"
echo "  ASCEND_OPP_PATH=$ASCEND_OPP_PATH"
echo "  ASCEND_HOME_PATH=$ASCEND_HOME_PATH"

export LD_LIBRARY_PATH="${CANN_TOOLKIT_ROOT}/x86_64-linux/lib64:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CANN_TOOLKIT_ROOT}/x86_64-linux/devlib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CANN_TOOLKIT_ROOT}/opp/lib64:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CANN_TOOLKIT_ROOT}/opp/lib64/plugin/opskernel:${LD_LIBRARY_PATH}"
export PYTHONPATH="${CANN_TOOLKIT_ROOT}/python/site-packages:${PYTHONPATH}"

for required_dir in "${CANN_BIN_DIR}" "${CANN_LIB_DIR}" "${CANN_PYTHON_SITE_PACKAGES}"; do
   if [ ! -e "${required_dir}" ]; then
      echo "[FATAL] Required CANN path missing: ${required_dir}" >&2
      exit 1
   fi
done

RUN_ID=${RUN_ID:-qwen36_polar_pd_$(date +%Y%m%d-%H%M%S)}
MASTER_ADDR=${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}
CURRENT_IP=${CURRENT_IP:-}
SOCKET_IFNAME=${SOCKET_IFNAME:-}
RAY_HOST_IP=${RAY_HOST_IP:-}
MOONCAKE_HOST_IP=${MOONCAKE_HOST_IP:-}

get_ipv4_addr() {
   local ifname="$1"
   ip -o -4 addr show dev "$ifname" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1
}

get_ifname_by_ip() {
   local host="$1"
   [ -z "$host" ] && return 0
   ip -o addr show 2>/dev/null | awk -v target="$host" '
      {
         split($4, parts, "/")
         if (parts[1] == target) {
            print $2
            exit
         }
      }'
}
NNODES=${NNODES:-1}
NPUS_PER_NODE=${NPUS_PER_NODE:-16}
RAY_PORT=${RAY_PORT:-26460}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-28290}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/ray_qwen36_vime_polar_pd}

ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-0}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-16}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}

POLAR_OUTPUT_DIR=${POLAR_OUTPUT_DIR:-output/polar_bridge}
OPERATOR_DATA_ROOT=${OPERATOR_DATA_ROOT:-/home/docker/datasets/op_assets_cudallm_filtered189}
OPERATOR_TASK_JSONL=${OPERATOR_TASK_JSONL:-${OPERATOR_DATA_ROOT}/operator_tasks.jsonl}
OPERATOR_TASKS_DIR=${OPERATOR_TASKS_DIR:-${OPERATOR_DATA_ROOT}/op_tasks}
VLLM_ROUTER_IP=${VLLM_ROUTER_IP:-}
VLLM_PD_CONFIG=${VLLM_PD_CONFIG:-${VIME_ROOT}/scripts/vllm_qwen36_35b_polar_single56_pd_mooncake_rollout_only.yaml}
PREFILL_PORT=${PREFILL_PORT:-18100}
DECODE_PORT=${DECODE_PORT:-18200}
PROXY_PORT=${PROXY_PORT:-8001}
VLLM_ROUTER_PORT=${VLLM_ROUTER_PORT:-${PROXY_PORT}}
KV_CONNECTOR=${KV_CONNECTOR:-MooncakeConnectorV1}

export PYTHONBUFFERED=16
export PATH="${CANN_BIN_DIR}:${PATH:-}"
# Prepend /usr/local/lib/python3.11/site-packages for newly compiled mooncake
export PYTHONPATH="/usr/local/lib/python3.11/site-packages:/workspace/vllm:/workspace/vllm-ascend:/workspace/Megatron-LM:${VIME_ROOT}:${CANN_PYTHON_SITE_PACKAGES}:${CANN_TBE_DIR}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64:${CANN_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TASK_QUEUE_ENABLE=0
export TORCHDYNAMO_DISABLE=1
export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}
export QWEN36_CP_MODE=ulysses
export QWEN36_CAUSAL_CONV1D_IMPL=triton
export QWEN36_CHUNK_LMHEAD=${QWEN36_CHUNK_LMHEAD:-0}
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_TOOL_CALL_PARSER=qwen3_coder
export VLLM_REASONING_PARSER=qwen3
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-2400}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-512}
export HCCL_INTRA_ROCE_ENABLE=${HCCL_INTRA_ROCE_ENABLE:-1}
export HCCL_INTRA_PCIE_ENABLE=${HCCL_INTRA_PCIE_ENABLE:-0}
export HCCL_INTER_HCCS_DISABLE=${HCCL_INTER_HCCS_DISABLE:-true}
export HCCL_SOCKET_FAMILY=${HCCL_SOCKET_FAMILY:-AF_INET}
export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export POLAR_KEEP_SESSION_DIR=${POLAR_KEEP_SESSION_DIR:-1}
export POLAR_TRAJECTORY_PG_STRICT=${POLAR_TRAJECTORY_PG_STRICT:-1}
export POLAR_ANTHROPIC_DEFAULT_MAX_TOKENS=${POLAR_ANTHROPIC_DEFAULT_MAX_TOKENS:-12288}

source "${VIME_ROOT}/scripts/models/qwen3.5-35B-A3B.sh"

if [ -n "${SOCKET_IFNAME}" ]; then
   CURRENT_IP=${CURRENT_IP:-$(get_ipv4_addr "${SOCKET_IFNAME}")}
fi
CURRENT_IP=${CURRENT_IP:-$(hostname -I | awk '{print $1}')}
MASTER_ADDR=${MASTER_ADDR:-${CURRENT_IP}}
RAY_HOST_IP=${RAY_HOST_IP:-${CURRENT_IP}}
MOONCAKE_HOST_IP=${MOONCAKE_HOST_IP:-${CURRENT_IP}}
VLLM_ROUTER_IP=${VLLM_ROUTER_IP:-${CURRENT_IP}}
export VIME_HOST_IP=${VIME_HOST_IP:-${MASTER_ADDR}}
if [ -z "${CURRENT_IP}" ] || [ -z "${MASTER_ADDR}" ]; then
   echo "[FATAL] Failed to detect CURRENT_IP/MASTER_ADDR. Set CURRENT_IP, MASTER_ADDR, or SOCKET_IFNAME explicitly." >&2
   exit 1
fi
if [ -z "${SOCKET_IFNAME}" ]; then
   SOCKET_IFNAME=$(get_ifname_by_ip "${CURRENT_IP}")
fi
if [ -z "${SOCKET_IFNAME}" ]; then
   echo "[FATAL] Failed to determine SOCKET_IFNAME for CURRENT_IP=${CURRENT_IP}. Set SOCKET_IFNAME explicitly." >&2
   exit 1
fi
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},${CURRENT_IP}${no_proxy:+,${no_proxy}}"
export NO_PROXY="${no_proxy}"
export HCCL_SOCKET_IFNAME="${SOCKET_IFNAME}"
export GLOO_SOCKET_IFNAME="${SOCKET_IFNAME}"
export TP_SOCKET_IFNAME="${SOCKET_IFNAME}"
export HCCL_IF_IP="${CURRENT_IP}"

POLAR_ROLLOUT_URL=${POLAR_ROLLOUT_URL:-http://${MASTER_ADDR}:8080}
LOG_FILE=${LOG_FILE:-/home/docker/logs/train_${RUN_ID}.log}
USE_WANDB=${USE_WANDB:-1}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_PROJECT=${WANDB_PROJECT:-qwen36-rollout-only-pd}
WANDB_GROUP=${WANDB_GROUP:-single-rollout}
WANDB_DIR=${WANDB_DIR:-/workspace/wandb_logs}
WANDB_HOST=${WANDB_HOST:-http://127.0.0.1:8088}
mkdir -p logs "${POLAR_OUTPUT_DIR}" /home/docker/logs "${WANDB_DIR}"

TRACKING_ARGS=()
if [ "${USE_WANDB}" = "1" ]; then
   source /workspace/vime/scripts/common/wandb_ready.sh
   assert_wandb_ready "${WANDB_HOST}"
   TRACKING_ARGS+=(--use-wandb)
   TRACKING_ARGS+=(--wandb-mode "${WANDB_MODE}")
   TRACKING_ARGS+=(--wandb-dir "${WANDB_DIR}")
   TRACKING_ARGS+=(--wandb-host "${WANDB_HOST}")
   TRACKING_ARGS+=(--wandb-project "${WANDB_PROJECT}")
   TRACKING_ARGS+=(--wandb-group "${WANDB_GROUP}")
   [ -n "${WANDB_KEY:-}" ] && TRACKING_ARGS+=(--wandb-key "${WANDB_KEY}")
   [ "${WANDB_RANDOM_SUFFIX:-1}" = "0" ] && TRACKING_ARGS+=(--disable-wandb-random-suffix)
   [ "${WANDB_ALWAYS_USE_TRAIN_STEP:-0}" = "1" ] && TRACKING_ARGS+=(--wandb-always-use-train-step)
   [ -n "${WANDB_RUN_ID:-}" ] && TRACKING_ARGS+=(--wandb-run-id "${WANDB_RUN_ID}")
   [ "${USE_TENSORBOARD:-0}" = "1" ] && TRACKING_ARGS+=(--use-tensorboard)
   [ -n "${TB_PROJECT_NAME:-}" ] && TRACKING_ARGS+=(--tb-project-name "${TB_PROJECT_NAME}")
   [ -n "${TB_EXPERIMENT_NAME:-}" ] && TRACKING_ARGS+=(--tb-experiment-name "${TB_EXPERIMENT_NAME}")
fi

CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT:-/home/docker/Qwen3.6-35B-A3B}
   --ref-load ${REF_LOAD:-/home/docker/Qwen3.6-35B-A3B_fused_torch_dist}
   --save ${SAVE:-/workspace/Qwen3.6-35B-A3B_vime_polar_pd}/
   --save-interval 10
   --no-save-optim
   --megatron-to-hf-mode raw
   --optimization-level "$([ "${FEAT_OPT2:-0}" = "1" ] && echo 2 || echo 0)"
)

TOPO_ARGS=(
   --actor-num-nodes ${ACTOR_NUM_NODES}
   --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE}
   --rollout-num-gpus ${ROLLOUT_NUM_GPUS}
   --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
)

ROLLOUT_ARGS=(
   --rollout-function-path vime_bridge.rollout.generate_rollout_polar_async
   --eval-function-path vime_bridge.rollout.generate_rollout_polar_async
   --prompt-data "${OPERATOR_TASK_JSONL}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --reward-key score
   --custom-reward-post-process-path vime_bridge.reward_post_process.post_process_rewards
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT:-1}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-4}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-32768}"
   --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-131072}"
   --rollout-temperature 0.7
   --global-batch-size "${GLOBAL_BATCH_SIZE:-32}"
   --save-debug-rollout-data "${POLAR_OUTPUT_DIR}/vime_debug_rollout_${RUN_ID}_{rollout_id}.pt"
   --save-debug-train-data "${POLAR_OUTPUT_DIR}/vime_debug_train_${RUN_ID}_rollout_{rollout_id}_{rank}.pt"
   --use-dynamic-global-batch-size
   --rollout-seed "${ROLLOUT_SEED:-42}"
)

POLAR_ARGS=(
   --polar-url "${POLAR_ROLLOUT_URL}"
   --polar-run-id "${RUN_ID}"
   --polar-reward-key score
   --polar-task-id-template "{args.polar_run_id}-polar-op-{rollout_id}-{sample.group_index}"
   --operator-tasks-dir "${OPERATOR_TASKS_DIR}"
   --rollout-max-async-level "${POLAR_MAX_ASYNC_LEVEL:-1}"
   --rollout-request-timeout "${POLAR_ROLLOUT_REQUEST_TIMEOUT:-8000}"
   --rollout-scheduler-mode session_pool
   --rollout-max-active-sessions "${POLAR_MAX_ACTIVE_SESSIONS:-16}"
   --rollout-release-on-postrun
   --rollout-min-complete-accept-fraction "${POLAR_MIN_COMPLETE_ACCEPT_FRACTION:-0.8}"
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TP:-2}"
   --pipeline-model-parallel-size "${PP:-1}"
   --context-parallel-size "${CP:-4}"
   --expert-model-parallel-size "${EP:-8}"
   --expert-tensor-parallel-size 1
   --sequence-parallel
   --chunked-lm-head
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-32768}"
   --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-64}"
   --seq-length "${SEQ_LENGTH:-131072}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 2e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

VLLM_ARGS=(
   --rollout-backend vllm
   --qwen-gdn-backend npu
   --model-name qwen3_5moeforconditionalgeneration
   --vllm-hf-overrides '{"architectures":["Qwen3_5MoeForConditionalGeneration"]}'
   --vllm-router-ip "${VLLM_ROUTER_IP}"
   --vllm-router-port "${VLLM_ROUTER_PORT}"
   --vllm-weight-sync-mode native
   --no-vllm-weight-sync-packed
   --vllm-enable-expert-parallel
   --vllm-gpu-memory-utilization "${VLLM_GPU_MEM_UTIL:-0.8}"
   --vllm-max-num-seqs "${VLLM_MAX_NUM_SEQS:-96}"
   --vllm-max-model-len "${VLLM_MAX_MODEL_LEN:-131072}"
   --vllm-enable-sleep-mode
   --vllm-speculative-config '{"method":"mtp","num_speculative_tokens":3}'  # Disabled: 0% acceptance rate
   --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
   --vllm-config "${VLLM_PD_CONFIG}"
   --disaggregation-backend mooncake
   --no-offload-train
   --no-offload-rollout
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-flash-attn
   --moe-token-dispatcher-type alltoall
   --no-gradient-accumulation-fusion
   --seed "${SEED:-1234}"
)

[ "${FEAT_PREFIX_CACHE:-1}" = "1" ] && VLLM_ARGS+=(--vllm-enable-prefix-caching --vllm-enable-chunked-prefill)
echo "[feat] pd=1 mooncake=1 internal_vllm_config=1 prefix_cache=${FEAT_PREFIX_CACHE:-1} vllm_gpu_mem_util=${VLLM_GPU_MEM_UTIL:-0.8} proxy_port=${PROXY_PORT}"

if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
   ray stop --force
   rm -rf "${RAY_TEMP_DIR}"
   ray start --head --port "${RAY_PORT}" --dashboard-host=0.0.0.0 --node-ip-address="${CURRENT_IP}" --dashboard-port="${RAY_DASHBOARD_PORT}" --num-gpus="${NPUS_PER_NODE}" --resources='{"NPU": '"${NPUS_PER_NODE}"'}' --temp-dir="${RAY_TEMP_DIR}" --disable-usage-stats
else
   ray stop --force
   rm -rf "${RAY_TEMP_DIR}"
   while true; do
      ray start --address="${MASTER_ADDR}:${RAY_PORT}" --node-ip-address="${CURRENT_IP}" --num-gpus="${NPUS_PER_NODE}" --resources='{"NPU": '"${NPUS_PER_NODE}"'}' --temp-dir="${RAY_TEMP_DIR}" --disable-usage-stats
      ray status && break
      sleep 5
   done
fi

while true; do
   active_node_count=$(ray status | awk '
      /^Active:/ {in_active=1; next}
      /^Pending:/ {in_active=0}
      in_active && $1 == "1" && $2 ~ /^node_/ {count++}
      END {print count + 0}')
   echo "[stage] wait Ray nodes active=${active_node_count}/${NNODES}"
   if [ "$active_node_count" -eq "$NNODES" ]; then
      break
   fi
   sleep 5
done

unset ASCEND_RT_VISIBLE_DEVICES HCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME TP_SOCKET_IFNAME HCCL_IF_IP

# ─── 启动 vLLM metrics 监控面板(旁路,失败不影响训练)───
# engine 端口由 vime 运行时分配(rollout.py:165 从 base_port=15000 起探测空闲端口),
# 启动前不可知且不连续 → 面板自动扫描发现。不要用 PREFILL_PORT/DECODE_PORT:
# 那两个变量没有传给 vllm,上面没有服务在监听。
METRICS_DASHBOARD_PORT=${METRICS_DASHBOARD_PORT:-5000} \
METRICS_HOST_IP="${CURRENT_IP}" \
   source "${VIME_ROOT}/scripts/common/start_metrics_monitor.sh"

python3 train_async.py \
   --debug-rollout-only \
   ${TOPO_ARGS[@]} ${MODEL_ARGS[@]} ${ROLLOUT_ARGS[@]} ${POLAR_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} ${GRPO_ARGS[@]} ${PERF_ARGS[@]} ${VLLM_ARGS[@]} \
   ${MISC_ARGS[@]} ${CKPT_ARGS[@]} ${TRACKING_ARGS[@]} \
   2>&1 | tee "${LOG_FILE}"
