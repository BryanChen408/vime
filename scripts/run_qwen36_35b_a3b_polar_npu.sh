#!/usr/bin/env bash
set -eo pipefail
set -x

# ============================================================================
# vime + polar 算子 RL 启动脚本(全异步分卡,vllm/vllm-ascend rollout)。
#
# 由团队的 run_qwen36_35b_a3b_async_npu4_15.sh 改造:rollout/reward hook 指向
# 本仓 vendored 的 `vime_bridge`(slime_bridge 的 slime->vime 港口,自足、不需
# `slime` 包)。轨迹 loss 用 vime 原生按-rollout 均权(Option A):adapter 给
# 每条 polar 轨迹的所有 trace 同一个 rollout_id,故 **不** 传
# --custom-pg-loss-reducer-function-path。
#
# 前置:vime_bridge/ 必须在 VIME_DIR(=vime 仓根)下、且在 PYTHONPATH 上
# (下方 PYTHONPATH 已含 /workspace/vime)。PYTHONPATH 里的 ProRL-Agent-Server/src
# 现已非必需(vime_bridge 自带 wire.py),保留无害。
# 依赖:httpx / fastapi / uvicorn / pydantic。
# 参考对齐对象:slime-ascend + polar(oracle)。
# ============================================================================

# Ascend runtime env
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
set -u

cleanup_runtime() {
  ray stop 2>/dev/null || true
  for pattern in "vllm serve" "VLLM::" "EngineCore" "from multiprocessing"; do
    pkill -TERM -f "$pattern" 2>/dev/null || true
  done
  sleep 5

  if pgrep -f "raylet|gcs_server|dashboard|ray::|log_monitor.py|monitor.py" >/dev/null 2>&1; then
    ray stop --force 2>/dev/null || true
  fi

  for pattern in "vllm serve" "VLLM::" "EngineCore" "from multiprocessing"; do
    pkill -KILL -f "$pattern" 2>/dev/null || true
  done
  sleep 3
}

wandb_has_netrc_login() {
  local wandb_host="$1"
  python - "$wandb_host" <<'PY'
import netrc
import sys
from urllib.parse import urlparse

host = sys.argv[1].rstrip('/')
parsed = urlparse(host if '://' in host else f'https://{host}')
machine = parsed.netloc or parsed.path
try:
    auth = netrc.netrc().authenticators(machine)
except Exception:
    auth = None
print("yes" if auth and auth[2] else "no")
PY
}

wandb_check_host() {
  local wandb_host="$1"
  python - "$wandb_host" <<'PY'
import sys
import urllib.error
import urllib.request

url = sys.argv[1].rstrip('/') + '/'
request = urllib.request.Request(url, headers={"User-Agent": "wandb-preflight"})
try:
    with urllib.request.urlopen(request, timeout=5) as response:
        print(f"W&B host reachable: HTTP {response.status}")
except urllib.error.HTTPError as exc:
    print(f"W&B host reachable with HTTP error: {exc.code}")
except Exception as exc:
    raise SystemExit(f"W&B host check failed for {url}: {exc}")
PY
}

assert_wandb_ready() {
  local wandb_host="$1"

  case "${WANDB_MODE:-online}" in
    offline|disabled)
      echo "WANDB_MODE=${WANDB_MODE:-online}; skipping W&B reachability/login preflight."
      export WANDB_BASE_URL="$wandb_host"
      return
      ;;
  esac

  if ! command -v wandb >/dev/null 2>&1; then
    echo "wandb CLI not found; refusing to launch without W&B." >&2
    exit 1
  fi

  echo "Checking W&B login against ${wandb_host}"
  wandb --version
  wandb status || true
  wandb_check_host "$wandb_host"

  if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "WANDB_API_KEY detected; enabling W&B logging."
  elif [ "$(wandb_has_netrc_login "$wandb_host")" = "yes" ]; then
    echo "Detected W&B credentials for ${wandb_host} in ~/.netrc; enabling W&B logging."
  else
    echo "No W&B credentials detected for ${wandb_host}. Run 'wandb login --relogin --host=${wandb_host}' first." >&2
    exit 1
  fi

  export WANDB_BASE_URL="$wandb_host"
}

# cleanup_runtime  # disabled

export SLIME_SCRIPT_TRAIN_BACKEND=megatron
export PYTHONPATH="/workspace/vllm:/workspace/vllm-ascend:/workspace/Megatron-Bridge-slime/src:/workspace/Megatron-LM:/workspace/vime:/home/l00830933/polar_debug/ProRL-Agent-Server/src:${PYTHONPATH:-}"
cd /tmp

# 12-card layout: actor gets physical 4-11, rollout gets physical 12-15.
# The 4-11 actor span crosses planes on this host, so enable intra-node RoCE
# per the HCCL error guidance captured in the prior debugging notes.
export ASCEND_RT_VISIBLE_DEVICES=${NPU_CARDS:?set NPU_CARDS}
export HCCL_INTRA_ROCE_ENABLE=1
# Megatron FSDP requires CUDA_DEVICE_MAX_CONNECTIONS to be unset / >1.
unset CUDA_DEVICE_MAX_CONNECTIONS
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)
export DISABLE_L2_CACHE=1
export VLLM_ASCEND_ENABLE_NZ=0
export QWEN36_CAUSAL_CONV1D_IMPL=triton
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1
export TASK_QUEUE_ENABLE=0

NODE_IP="$(hostname -I | awk '{print $1}')"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy="127.0.0.1,localhost,${NODE_IP}"
export NO_PROXY="${no_proxy}"

VIME_DIR=/workspace/vime
HF_CKPT="${HF_CKPT:-/home/docker/Qwen3.6-35B-A3B}"
REF_LOAD="${REF_LOAD:-/home/docker/Qwen3.6-35B-A3B_fused_torch_dist}"
# Operator-RL data: each JSONL row MUST carry metadata.op_name; OPERATOR_TASKS_DIR
# holds the per-op task assets that polar loads. (Was dapo-math; operator_samples
# submit mode requires op_name or every group is dropped.)
PROMPT_DATA="${PROMPT_DATA:-/workspace/datasets/op_assets_cudallm_filtered189/operator_tasks.jsonl}"
OPERATOR_TASKS_DIR="${OPERATOR_TASKS_DIR:-/workspace/datasets/op_assets_cudallm_filtered189/op_tasks}"
WANDB_HOST="${WANDB_HOST:-http://127.0.0.1:8088}"
WANDB_PROJECT="${WANDB_PROJECT:-vime-qwen36-35b-a3b}"
WANDB_GROUP="${WANDB_GROUP:-qwen36-35b-a3b-async-npu4-15}"
POLAR_ROLLOUT_URL="${POLAR_ROLLOUT_URL:-http://127.0.0.1:8080}"
POLAR_GATEWAY_URL="${POLAR_GATEWAY_URL:-http://127.0.0.1:8100}"
POLAR_CALLBACK_HOST="${POLAR_CALLBACK_HOST:-${NODE_IP}}"
POLAR_INFERENCE_ENGINE="${POLAR_INFERENCE_ENGINE:-vllm}"
POLAR_ROUTER_PORT="${POLAR_ROUTER_PORT:-8001}"
# session_pool + accept-fraction 0.6 + async-level 1 to match the live slime operator-RL run.
POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-1}"
POLAR_REQUEST_TIMEOUT="${POLAR_REQUEST_TIMEOUT:-8000}"
POLAR_MIN_COMPLETE_ACCEPT_FRACTION="${POLAR_MIN_COMPLETE_ACCEPT_FRACTION:-0.6}"
POLAR_SCHEDULER_MODE="${POLAR_SCHEDULER_MODE:-session_pool}"
POLAR_MAX_ACTIVE_SESSIONS="${POLAR_MAX_ACTIVE_SESSIONS:-16}"
POLAR_RUN_ID="${POLAR_RUN_ID:-${WANDB_GROUP}}"
POLAR_CONFIG_PATH="${POLAR_CONFIG_PATH:-${VIME_DIR}/runs/polar_config_qwen36_35b_a3b_${STAMP:-manual}.yaml}"

source "${VIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh"
assert_wandb_ready "${WANDB_HOST}"

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="${VIME_DIR}/runs/qwen36_35b_a3b_async_npu4_15_${STAMP}"
CKPT_DIR="${RUN_ROOT}/actor_ckpt"
WANDB_DIR="${RUN_ROOT}/wandb"
POLAR_CONFIG_PATH="${RUN_ROOT}/polar_bridge_config.yaml"
mkdir -p "${RUN_ROOT}/debug" "${CKPT_DIR}" "${WANDB_DIR}"
echo "RUN_ROOT=${RUN_ROOT}"

cat > "${POLAR_CONFIG_PATH}" <<EOF
polar_rollout_url: "${POLAR_ROLLOUT_URL}"
polar_gateway_url: "${POLAR_GATEWAY_URL}"
polar_callback_host: "${POLAR_CALLBACK_HOST}"
polar_reward_key: "score"
polar_max_async_level: ${POLAR_MAX_ASYNC_LEVEL}
polar_request_timeout: ${POLAR_REQUEST_TIMEOUT}
polar_min_complete_accept_fraction: ${POLAR_MIN_COMPLETE_ACCEPT_FRACTION}
polar_scheduler_mode: "${POLAR_SCHEDULER_MODE}"
polar_inference_engine: "${POLAR_INFERENCE_ENGINE}"
sglang_router_ip: "${NODE_IP}"
sglang_router_port: ${POLAR_ROUTER_PORT}
EOF

echo "Using HF checkpoint: ${HF_CKPT}"
echo "Using ref checkpoint: ${REF_LOAD}"
echo "Using prompt data: ${PROMPT_DATA}"
echo "Using W&B host: ${WANDB_HOST}"
echo "Using Polar rollout URL: ${POLAR_ROLLOUT_URL}"
echo "Using Polar gateway URL: ${POLAR_GATEWAY_URL}"
echo "Using Polar callback host: ${POLAR_CALLBACK_HOST}"
echo "Using actor/rollout split: 8 / 4"
echo "Using rollout engine topology: 1 engine x 4 NPUs"
echo "Using rollout max response len: 16384"
echo "Using Polar config: ${POLAR_CONFIG_PATH}"

python "${VIME_DIR}/train_async.py" \
  --train-backend megatron \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --rollout-num-gpus 4 \
  --rollout-num-gpus-per-engine 4 \
  ${MODEL_ARGS[@]} \
  --qwen-gdn-backend npu \
  --hf-checkpoint "${HF_CKPT}" \
  --model-name qwen3_5moeforconditionalgeneration \
  --ref-load "${REF_LOAD}" \
  --save "${CKPT_DIR}" \
  --save-interval 20 \
  --megatron-to-hf-mode raw \
  --optimization-level 0 \
  --prompt-data "${PROMPT_DATA}" \
  --input-key prompt \
  --label-key label \
  --metadata-key metadata \
  --apply-chat-template \
  --rollout-shuffle \
  --save-debug-rollout-data "${RUN_ROOT}/debug/rollout_{rollout_id}.pt" \
  --save-debug-train-data "${RUN_ROOT}/debug/train_{rollout_id}_{rank}.pt" \
  --rm-type math \
  --reward-key score \
  --custom-reward-post-process-path vime_bridge.reward_post_process.post_process_rewards \
  --rollout-backend vllm \
  --rollout-function-path vime_bridge.rollout.generate_rollout_polar_async \
  --eval-function-path vime_bridge.rollout.generate_rollout_polar_async \
  --custom-config-path "${POLAR_CONFIG_PATH}" \
  --polar-url "${POLAR_ROLLOUT_URL}" \
  --polar-run-id "${POLAR_RUN_ID}" \
  --polar-reward-key score \
  --polar-task-id-template '{args.polar_run_id}-polar-op-{rollout_id}-{sample.group_index}' \
  --operator-tasks-dir "${OPERATOR_TASKS_DIR}" \
  --rollout-scheduler-mode "${POLAR_SCHEDULER_MODE}" \
  --rollout-max-active-sessions "${POLAR_MAX_ACTIVE_SESSIONS}" \
  --rollout-release-on-postrun \
  --rollout-min-complete-accept-fraction "${POLAR_MIN_COMPLETE_ACCEPT_FRACTION}" \
  --rollout-max-async-level "${POLAR_MAX_ASYNC_LEVEL}" \
  --rollout-request-timeout "${POLAR_REQUEST_TIMEOUT}" \
  --use-tis \
  --vllm-hf-overrides '{"architectures":["Qwen3_5MoeForConditionalGeneration"]}' \
  --vllm-weight-sync-mode native \
  --no-vllm-weight-sync-packed \
  --vllm-gpu-memory-utilization 0.45 \
  --vllm-max-num-seqs 8 \
  --vllm-enable-sleep-mode \
  --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --vllm-router-port ${POLAR_ROUTER_PORT} \
  --no-offload-train \
  --no-offload-rollout \
  --vllm-max-model-len 8192 \
  --num-rollout ${NUM_ROLLOUT:-20} \
  --rollout-batch-size 1 \
  --n-samples-per-prompt 8 \
  --num-steps-per-rollout 1 \
  --rollout-max-response-len 2048 \
  --rollout-temperature 1.0 \
  --global-batch-size 8 \
  --balance-data \
  --advantage-estimator grpo \
  --kl-loss-coef 0.0 \
  --kl-loss-type low_var_kl \
  --entropy-coef 0.0 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --optimizer-cpu-offload \
  --overlap-cpu-optimizer-d2h-h2d \
  --use-precision-aware-optimizer \
  --tensor-model-parallel-size 1 \
  --sequence-parallel \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 8 \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 512 \
  --log-probs-chunk-size 1024 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --use-flash-attn \
  --no-gradient-accumulation-fusion \
  --use-wandb \
  --wandb-mode "${WANDB_MODE:-online}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-dir "${WANDB_DIR}" \
  --wandb-host "${WANDB_HOST}" \
  --train-memory-margin-bytes 2147483648 \
  --distributed-timeout-minutes 60 \
  2>&1 | tee "${RUN_ROOT}/run.log"
