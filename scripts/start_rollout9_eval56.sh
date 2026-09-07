#!/usr/bin/env bash
# .56: 16 NPU inference; .64: independently managed Polar and operator card pool.
# Run PREFLIGHT_ONLY=1 bash scripts/start_rollout9_eval56.sh for read-only checks.
set -euo pipefail

EVAL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unset RESOURCE_LAYOUT TRAIN_ENTRY FEAT_COLOCATE FEAT_OFFLOAD ROLLOUT_ONLY
unset POLAR_POLICY_TRANSITION_ENABLED VLLM_GPU_MEM_UTIL_DEDICATED
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

export HF_CKPT="${HF_CKPT:-/workspace/Qwen3.6-35B-A3B_vime_polar/rollout_19}"
# Both the HF path and these aliases are served; Polar need not hold the weights.
export VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16 /home/docker/Qwen3.6-35B-A3B}"
export CURRENT_IP=80.48.5.56 MASTER_ADDR=80.48.5.56 NNODES=1 NPUS_PER_NODE=16
export SOCKET_IFNAME="${SOCKET_IFNAME:-ens1f3}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ACTOR_NUM_NODES=1 ACTOR_NUM_GPUS_PER_NODE=0 ROLLOUT_NUM_GPUS=16
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
export TP="${ROLLOUT_NUM_GPUS_PER_ENGINE}" PP=1 CP=1 EP=1
export VLLM_ROUTER_PORT=8011 FEAT_LB_PROXY=1
export VLLM_LANGUAGE_MODEL_ONLY=1 VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
export FEAT_FLASHCOMM1=0 VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export FEAT_ROLLOUT_EP=0 FEAT_CROSS_DP_EP=0 FEAT_DP_EXTERNAL_LB=0 FEAT_BALANCE_SCHED=0
export FEAT_PREFIX_CACHE=1 FEAT_MULTISTREAM_SHARED_EXPERT=0 FEAT_STATIC_KERNEL=0
export FEAT_ASYNC_SCHED=0 FEAT_HCCL_AIV=1 FEAT_OPT2=0 FEAT_TRAIN_EXPANDABLE=0
export SEQ_LENGTH=262144 ROLLOUT_MAX_CONTEXT_LEN=262144 VLLM_MAX_MODEL_LEN=262144
# Match the effective trajectory cap in train_qwen36_polar_20260907-112711.log:
# 32768 * training CP8 = 262144. This evaluation has CP1 and no trainer;
# the bridge still uses max_tokens_per_gpu * CP to filter trajectories.
# vLLM's prefill batch budget remains 16384, independent of this filter cap.
export MAX_TOKENS_PER_GPU=262144 VLLM_MAX_NUM_BATCHED_TOKENS=16384 VLLM_MAX_NUM_SEQS=96
export ROLLOUT_MAX_RESPONSE_LEN=32768 ROLLOUT_TEMPERATURE=1.0
export ROLLOUT_TOP_P=1.0 ROLLOUT_TOP_K=-1 ROLLOUT_SEED=42 SEED=1234
# Training is skipped, but retain the baseline entropy setting in the args.
export ENTROPY_COEF=0.001

export POLAR_ROLLOUT_URL="${POLAR_ROLLOUT_URL:-http://80.48.5.64:8180}"
export POLAR_GATEWAY_URL="${POLAR_GATEWAY_URL:-http://80.48.5.64:8200}"
export POLAR_MAX_ACTIVE_SESSIONS=60 POLAR_ROLLOUT_REQUEST_TIMEOUT=14400
export POLAR_MAX_ASYNC_LEVEL=1 POLAR_MIN_COMPLETE_ACCEPT_FRACTION=0.6
export OPERATOR_DATA_ROOT="${OPERATOR_DATA_ROOT:-/home/docker/datasets/op_tasks/op_assets_cudallm_filtered189}"
export OPERATOR_TASK_JSONL="${OPERATOR_TASK_JSONL:-${OPERATOR_DATA_ROOT}/operator_tasks.ascendc.val20.jsonl}"
export OPERATOR_TASKS_DIR="${OPERATOR_DATA_ROOT}/op_tasks"
export ROLLOUT_BATCH_SIZE=20 N_SAMPLES_PER_PROMPT=3 GLOBAL_BATCH_SIZE=60 NUM_ROLLOUT=1
export USE_WANDB=0
export HCCL_INTER_HCCS_DISABLE=false HCCL_INTRA_ROCE_ENABLE=1 HCCL_INTRA_PCIE_ENABLE=0 HCCL_BUFFSIZE=512
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60255 HCCL_NPU_SOCKET_PORT_RANGE=61000-61255
export no_proxy=127.0.0.1,localhost,80.48.5.56,80.48.5.64,.huawei.com,local,.local
export NO_PROXY="${no_proxy}"
export RUN_ID="${RUN_ID:-qwen36_rollout9_eval_$(date +%Y%m%d-%H%M%S)}"
export LOG_FILE="${LOG_FILE:-/mnt/pipeline-data/train_log/train_${RUN_ID}.log}"
export POLAR_OUTPUT_DIR="${POLAR_OUTPUT_DIR:-/mnt/pipeline-data/polar_eval/${RUN_ID}}"

# Check before the existing launcher restarts Ray. GET requests only; never
# cancel old sessions or change the remote Polar policy/pause state here.
python3 - <<'PY'
import json
import os
from pathlib import Path
import sys
import urllib.request

def fail(message):
    sys.exit('[preflight][FAIL] ' + message)

def get_json(url):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=5) as response:
            return json.load(response)
    except Exception as exc:
        fail(f'{url}: {exc}')

model = Path(os.environ['HF_CKPT'])
for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json'):
    if not (model / name).is_file():
        fail(f'missing model asset: {model / name}')
index = json.loads((model / 'model.safetensors.index.json').read_text())
shards = set(index['weight_map'].values())
if not shards:
    fail('checkpoint index is empty')
for shard in shards:
    if not (model / shard).is_file() or (model / shard).stat().st_size == 0:
        fail(f'missing/empty weight shard: {model / shard}')
data = Path(os.environ['OPERATOR_TASK_JSONL'])
rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
if len(rows) < int(os.environ['ROLLOUT_BATCH_SIZE']):
    fail(f'dataset has only {len(rows)} rows')
if not Path(os.environ['OPERATOR_TASKS_DIR']).is_dir():
    fail('operator task assets directory does not exist')
tp = int(os.environ['ROLLOUT_NUM_GPUS_PER_ENGINE'])
if tp not in (1, 2, 4, 8, 16):
    fail('ROLLOUT_NUM_GPUS_PER_ENGINE must divide the 16 local devices')
base = os.environ['POLAR_ROLLOUT_URL'].rstrip('/')
gateway = os.environ['POLAR_GATEWAY_URL'].rstrip('/')
health = get_json(base + '/health')
status = get_json(base + '/rollout/status')
gh = get_json(gateway + '/health')
inference = get_json(gateway + '/admin/inference/status')
if health.get('status') != 'ok' or gh.get('status') != 'ok':
    fail(f'Polar/gateway health is not ok: {health}, {gh}')
print('[preflight] gateway inference=' + json.dumps({
    key: inference.get(key) for key in (
        'paused', 'epoch_enforced', 'policy_namespace', 'policy_version', 'transition_id', 'base_url'
    )
}, ensure_ascii=False))
if inference.get('paused') or inference.get('epoch_enforced'):
    fail('Gateway retains a paused/enforced training policy. A healthy /health response does not '
         'mean it accepts evaluation sessions. Resolve the old .64 training transition and initialize '
         'Polar/gateway for standalone evaluation first; this script does not resume old tasks automatically.')
expected_router = f'http://{os.environ["CURRENT_IP"]}:{os.environ["VLLM_ROUTER_PORT"]}'
if inference.get('base_url', '').rstrip('/') != expected_router:
    fail(f'gateway inference base_url must be {expected_router}')
admission = status.get('policy_admission', {})
if admission.get('closed') or admission.get('enforced'):
    fail(f'Polar retains policy admission state incompatible with this non-transactional run: {admission}')
nodes = status.get('nodes', {}).get('nodes', [])
if not any(n.get('gateway_url', '').rstrip('/') == gateway and n.get('healthy')
           and not n.get('draining') for n in nodes):
    fail(f'no healthy, non-draining registered gateway at {gateway}')
pending = status.get('pipeline', {}).get('pending_sessions', 0)
print(f'[preflight] Polar pending_sessions={pending}; gateway metrics={gh.get("metrics", {})}')
if pending or gh.get('active_sessions') or any(gh.get('metrics', {}).values()):
    fail('Polar still has previous work. If sessions are DISPATCHING while gateway queues are all zero, '
         'inspect the .64 rollout dispatch / POST /sessions logs before restarting this evaluation. '
         'No sessions have been cancelled and Ray has not been restarted.')
print(f'[preflight] PASS: {model}, {len(shards)} shards, {len(rows)} dataset rows, {16 // tp} TP{tp} engines')
print('[preflight] HTTP readiness is not an end-to-end dispatch test; confirm the first real session enters INIT/RUN.')
PY

if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
   exit 0
fi
mkdir -p -- "$(dirname -- "${LOG_FILE}")" "${POLAR_OUTPUT_DIR}"
printf '[eval] model=%s router=http://%s:%s log=%s\n' "${HF_CKPT}" "${CURRENT_IP}" "${VLLM_ROUTER_PORT}" "${LOG_FILE}"
# The child inherits environment variables, not this wrapper's nounset option
# (Ascend's vendor set_env.sh reads optional unset variables).
exec bash "${EVAL_SCRIPT_DIR}/run-qwen36-35b-polar-minimal-single-rollout-only.sh"



python3 - <<'PY'
import json
import urllib.request

base = "http://80.48.5.64:8180"
http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open(base + "/rollout/status", timeout=10) as r:
    status = json.load(r)

ids = [
    task_id for task_id, state in status["tasks"].items()
    if task_id.startswith("qwen36_polar_20260907-144336-")
    and state == "running"
]
if ids:
    req = urllib.request.Request(
        base + "/rollout/admin/tasks/cancel",
        data=json.dumps({
            "task_ids": ids,
            "reason": "switch_to_rollout9_evaluation"
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with http.open(req, timeout=30) as r:
        print(r.read().decode())
else:
    print("没有匹配的运行中任务")
PY
