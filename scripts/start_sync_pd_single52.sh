#!/usr/bin/env bash
# Single-node synchronous train + PD rollout E2E launcher.
#
# Physical placement on the 16-NPU host:
#   actor:   NPU 0-15 (16 NPU training)
#   Polar:   NPU 0-3   (managed by the Polar host service, outside Ray)
#   rollout: NPU 4-15  (12 NPU, 2P4D/TP2 PD; engines share actor NPU 4-15)
#
# This is intentionally separate from start_sync_homo_single52.sh. That legacy
# entry keeps FEAT_PD_DISAGG=0 for the non-PD colocated baseline.
# Polar must already be serving the rollout API (normally :8180). The :8001
# Mooncake PD proxy is created by the VIME driver; it must not be pre-started.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

CURRENT_IP=${CURRENT_IP:-80.48.5.52}
MASTER_ADDR=${MASTER_ADDR:-${CURRENT_IP}}
SOCKET_IFNAME=${SOCKET_IFNAME:-ens1f3}
POLAR_ROLLOUT_URL=${POLAR_ROLLOUT_URL:-http://${CURRENT_IP}:8180}
VLLM_ROUTER_IP=${VLLM_ROUTER_IP:-${CURRENT_IP}}
VLLM_ROUTER_PORT=${VLLM_ROUTER_PORT:-8001}
RUN_ID=${RUN_ID:-qwen36_polar_pd_sync_${PD_E2E_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}}
LOG_FILE=${LOG_FILE:-/home/docker/logs/train_${RUN_ID}.log}

# The proxy is a plain subprocess and can outlive Ray after an aborted run.
# Clean both the ordinary LB proxy and the Mooncake PD proxy on this launcher
# port. The latter is not a Ray actor and therefore survives `ray stop`.
for proxy_pid in $(pgrep -f "pd_mooncake_proxy_server.py .*--port ${VLLM_ROUTER_PORT}" 2>/dev/null); do
   echo "[cleanup] stopping stale Mooncake PD proxy pid=${proxy_pid} port=${VLLM_ROUTER_PORT}"
   kill "${proxy_pid}" 2>/dev/null || true
done
pkill -f "vime\.ray\.lb_proxy .*--port ${VLLM_ROUTER_PORT}" 2>/dev/null || true
pkill -f "dp_load_balance_proxy_server.*${VLLM_ROUTER_PORT}" 2>/dev/null || true
for _ in 1 2 3 4 5; do
   ss -tln 2>/dev/null | grep -q ":${VLLM_ROUTER_PORT} " || break
   sleep 1
done
if ss -tln 2>/dev/null | grep -q ":${VLLM_ROUTER_PORT} "; then
   echo "[start_sync_pd][FATAL] port ${VLLM_ROUTER_PORT} is still occupied" >&2
   ss -tlnp 2>/dev/null | grep ":${VLLM_ROUTER_PORT} " >&2 || true
   exit 1
fi

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export CURRENT_IP MASTER_ADDR SOCKET_IFNAME
export NNODES=${NNODES:-1}
export NPUS_PER_NODE=16
export ACTOR_NUM_NODES=1
export ACTOR_NUM_GPUS_PER_NODE=16
export TRAIN_ENTRY=train.py
export FEAT_OFFLOAD=1
export FEAT_SYNC_ROLLOUT=1
export RESOURCE_LAYOUT=${RESOURCE_LAYOUT:-${SCRIPT_DIR}/resource_layout.single52_homo_colocate.yaml}
export ROLLOUT_NODE_IP=${ROLLOUT_NODE_IP:-${CURRENT_IP}}
export ROLLOUT_NUM_GPUS=12
export ROLLOUT_NUM_GPUS_PER_ENGINE=2

# This is the 12-card PD topology: prefill 4 NPU (2 TP2 engines) + decode 8
# NPU (4 TP2 engines). The 16-card 1P3D YAML must not be used here.
export FEAT_PD_DISAGG=1
export VLLM_PD_CONFIG=${VLLM_PD_CONFIG:-${SCRIPT_DIR}/vllm_qwen36_35b_polar_dual140_pd_12card.yaml}
export FEAT_ROLLOUT_EP=0
export FEAT_PREFIX_CACHE=0

[ "${NNODES}" = "1" ] || { echo "[start_sync_pd][FATAL] NNODES must be 1" >&2; exit 1; }
[ "${ASCEND_RT_VISIBLE_DEVICES}" = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" ] \
   || { echo "[start_sync_pd][FATAL] ASCEND_RT_VISIBLE_DEVICES must expose 0-15" >&2; exit 1; }
[ -f "${RESOURCE_LAYOUT}" ] \
   || { echo "[start_sync_pd][FATAL] layout not found: ${RESOURCE_LAYOUT}" >&2; exit 1; }
[ -f "${VLLM_PD_CONFIG}" ] \
   || { echo "[start_sync_pd][FATAL] PD config not found: ${VLLM_PD_CONFIG}" >&2; exit 1; }

export POLAR_ROLLOUT_URL VLLM_ROUTER_IP VLLM_ROUTER_PORT RUN_ID LOG_FILE
export VLLM_SERVED_MODEL_NAME=${VLLM_SERVED_MODEL_NAME:-/home/docker/Qwen3.6-35B-A3B}
export VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.70}
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-32768}
export SEQ_LENGTH=${SEQ_LENGTH:-262144}
export ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-262144}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-262144}
export VIME_MEM_PROBE=${VIME_MEM_PROBE:-1}
export VIME_EMPTY_CACHE_PER_STEP=${VIME_EMPTY_CACHE_PER_STEP:-1}
# Sample every IPC weight chunk on both producer and all TP workers. The final
# summary is collected before KV-cache wake; set to 0 only for overhead A/B.
export VIME_WSYNC_WEIGHT_PROBE=${VIME_WSYNC_WEIGHT_PROBE:-1}
# After every KV wake, exercise every prefill/decode Mooncake pair while Polar
# admission is still closed. Set to 0 only when intentionally skipping this check.
export VIME_PD_POST_WAKE_PROBE=${VIME_PD_POST_WAKE_PROBE:-1}
export FEAT_TRAIN_EXPANDABLE=${FEAT_TRAIN_EXPANDABLE:-1}
export FEAT_LB_PROXY=${FEAT_LB_PROXY:-1}
export FEAT_DP_EXTERNAL_LB=${FEAT_DP_EXTERNAL_LB:-0}
export FEAT_BALANCE_SCHED=${FEAT_BALANCE_SCHED:-0}
export FEAT_CROSS_DP_EP=${FEAT_CROSS_DP_EP:-0}
export FEAT_FLASHCOMM1=${FEAT_FLASHCOMM1:-0}
export FEAT_MULTISTREAM_SHARED_EXPERT=${FEAT_MULTISTREAM_SHARED_EXPERT:-1}
export FEAT_STATIC_KERNEL=${FEAT_STATIC_KERNEL:-0}
export FEAT_HCCL_AIV=${FEAT_HCCL_AIV:-1}
export PROFILE_TRAIN=${PROFILE_TRAIN:-0}
export PROFILE_OP=${PROFILE_OP:-0}

export TP=${TP:-2}
export PP=${PP:-1}
export CP=${CP:-8}
export EP=${EP:-8}
export POLAR_TRAJECTORY_PG_FLOOR=${POLAR_TRAJECTORY_PG_FLOOR:-0.05}
export HCCL_INTER_HCCS_DISABLE=${HCCL_INTER_HCCS_DISABLE:-false}
export HCCL_INTRA_ROCE_ENABLE=${HCCL_INTRA_ROCE_ENABLE:-1}
export HCCL_INTRA_PCIE_ENABLE=${HCCL_INTRA_PCIE_ENABLE:-0}
export ASCEND_CONNECT_TIMEOUT=${ASCEND_CONNECT_TIMEOUT:-60000}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-512}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60255}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61255}

export OPERATOR_DATA_ROOT=${OPERATOR_DATA_ROOT:-/home/docker/datasets/op_tasks/op_assets_cudallm_filtered189}
export OPERATOR_TASK_JSONL=${OPERATOR_TASK_JSONL:-${OPERATOR_DATA_ROOT}/operator_tasks.16.jsonl}
export OPERATOR_TASKS_DIR=${OPERATOR_TASKS_DIR:-${OPERATOR_DATA_ROOT}/op_tasks}

# Keep the first E2E run deliberately small. Set PD_E2E_SMOKE=0 for the
# existing synchronous smoke scale, or override any individual value.
if [ "${PD_E2E_SMOKE:-1}" = "1" ]; then
   export NUM_ROLLOUT=${NUM_ROLLOUT:-1}
   export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-1}
   export N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-1}
   export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1}
else
   export NUM_ROLLOUT=${NUM_ROLLOUT:-2}
   export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-2}
   export N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-2}
   export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4}
fi

mkdir -p /home/docker/logs
echo "[start_sync_pd] actor=16 rollout=12 PD=2P4D/TP2 visible=${ASCEND_RT_VISIBLE_DEVICES}"
echo "[start_sync_pd] layout=${RESOURCE_LAYOUT} pd_config=${VLLM_PD_CONFIG}"
echo "[start_sync_pd] polar=${POLAR_ROLLOUT_URL} proxy=http://${VLLM_ROUTER_IP}:${VLLM_ROUTER_PORT}"
echo "[start_sync_pd] prefix_cache=${FEAT_PREFIX_CACHE} weight_probe=${VIME_WSYNC_WEIGHT_PROBE} post_wake_probe=${VIME_PD_POST_WAKE_PROBE}"

bash "${SCRIPT_DIR}/run-qwen36-35b-polar-multi-pd.sh"
