#!/usr/bin/env bash
# Isolated TP2 vLLM boundary probe for stage 8. This intentionally does not
# register with Polar or enable VIME's sleep/weight-update lifecycle.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

ASCEND_ROOT=${ASCEND_ROOT:-/usr/local/Ascend}
CANN_ROOT=${CANN_ROOT:-${ASCEND_ROOT}/cann}
CANN_TOOLKIT_ROOT="${ASCEND_ROOT}/ascend-toolkit/cann-9.0.0"
CANN_PYTHON_SITE_PACKAGES="${CANN_ROOT}/python/site-packages"
CANN_TBE_DIR="${CANN_ROOT}/opp/built-in/op_impl/ai_core/tbe"
# The host's empty /etc/ascend_install.info makes the CANN 9.2 set_env.sh exit
# midway under its internal `set -e`; spell out the same paths instead.
set +e
source /usr/local/Ascend/nnal/atb/set_env.sh
set -e

export ASCEND_OPP_PATH="${CANN_ROOT}/opp"
export ASCEND_AICPU_PATH="${CANN_ROOT}"
export ASCEND_HOME_PATH="${CANN_ROOT}"
export ASCEND_TOOLKIT_HOME="${CANN_ROOT}"
export TOOLCHAIN_HOME="${CANN_ROOT}/toolkit"
export PYTHONPATH="/usr/local/lib/python3.11/site-packages:/workspace/Megatron-LM:${VIME_ROOT}:${CANN_PYTHON_SITE_PACKAGES}:${CANN_TBE_DIR}:${CANN_TOOLKIT_ROOT}/python/site-packages:${PYTHONPATH:-}"
export PATH="${CANN_ROOT}/bin:${PATH:-}"
export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64:${CANN_ROOT}/lib64:${CANN_TOOLKIT_ROOT}/x86_64-linux/lib64:${CANN_TOOLKIT_ROOT}/x86_64-linux/devlib:${CANN_TOOLKIT_ROOT}/opp/lib64:${CANN_TOOLKIT_ROOT}/opp/lib64/plugin/opskernel:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CANN_ROOT}/opp/vendors/custom_transformer/op_api/lib:/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}"

export VLLM_VERSION=0.23.0
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export TASK_QUEUE_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4,5}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HCCL_IF_IP="${HCCL_IF_IP:-80.48.5.52}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-ens1f3}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens1f3}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-ens1f3}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-512}"
export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_INTRA_PCIE_ENABLE="${HCCL_INTRA_PCIE_ENABLE:-0}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-2400}"
export no_proxy="127.0.0.1,localhost,80.48.5.52${no_proxy:+,${no_proxy}}"
export NO_PROXY="${no_proxy}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
# The production VIME launcher removes this variable in the vLLM child.
unset PYTORCH_NPU_ALLOC_CONF

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
source "${SCRIPT_DIR}/models/qwen3.5-35B-A3B.sh"

HF_CKPT="${HF_CKPT:-/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16}"
HOST="${HOST:-80.48.5.52}"
PORT="${PORT:-15000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-270336}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
LOG_FILE="${LOG_FILE:-/mnt/pipeline-data/train_log/vllm_qwen36_yarn_stage8_boundary_$(date +%Y%m%d_%H%M%S).log}"

python3 - <<'PY'
from pathlib import Path
import vllm
import vllm_ascend

expected = {
    "vllm": Path("/workspace/vllm-023"),
    "vllm_ascend": Path("/workspace/vllm-ascend-023"),
}
actual = {
    "vllm": Path(vllm.__file__).resolve(),
    "vllm_ascend": Path(vllm_ascend.__file__).resolve(),
}
for name, path in actual.items():
    if expected[name] not in path.parents:
        raise SystemExit(f"[stage8-probe][FATAL] {name} resolved to {path}, expected {expected[name]}")
    print(f"[stage8-probe] {name}={path}")
PY

python3 "${VIME_ROOT}/tools/qwen36_yarn_preflight.py" \
  --model "${HF_CKPT}" \
  --hf-overrides "${QWEN36_VLLM_HF_OVERRIDES}" \
  --max-model-len "${MAX_MODEL_LEN}"

echo "[stage8-probe] cards=${ASCEND_RT_VISIBLE_DEVICES} endpoint=http://${HOST}:${PORT} log=${LOG_FILE}"
vllm serve "${HF_CKPT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${HF_CKPT}" \
  --trust-remote-code \
  --seed 1234 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --logprobs-mode processed_logprobs \
  --hf-overrides "${QWEN36_VLLM_HF_OVERRIDES}" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs 96 \
  --enable-chunked-prefill \
  --reasoning-parser qwen3 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"multistream_overlap_shared_expert":true}' \
  2>&1 | tee "${LOG_FILE}"
