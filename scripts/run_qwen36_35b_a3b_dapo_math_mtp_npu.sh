#!/usr/bin/env bash
# Qwen3.6-35B-A3B (GDN/MoE, Route B port) DAPO-math RL + MTP training — single node, 8 Ascend NPUs.
#
# Same as run_qwen36_35b_a3b_dapo_math_npu.sh, but ALSO trains the 1 MTP layer online:
#   --mtp-num-layers 1        -> model_provider builds the MTP block into the GPTModel.
#   --enable-mtp-training     -> mtp_labels fed to forward, MTP loss computed/logged (train/*mtp_loss).
#   --mtp-loss-scaling-factor -> MTP loss weight in the total objective (default 0.2).
# The MTP layer weights come from Qwen3.6-35B-A3B_torch_dist (verified to contain mtp.layers.0.*);
# the _fused_torch_dist checkpoint has NO MTP tensors and must not be used here.
# Note: MTP training is incompatible with combined-1f1b pipeline (model.py:594 asserts); this run
# uses PP=1 so that path is not taken.
#
# Validation vehicle for the GDN port + converted checkpoint:
#   - actor (policy/ref) = vime_plugins.models.qwen3_5 GDN model, loaded from the
#     converted torch_dist via native mcore load_checkpoint (--megatron-to-hf-mode raw).
#   - rollout = vllm Qwen3_5MoeForConditionalGeneration on the same HF safetensors.
#   - data = dapo-math-17k, reward = deepscaler (local \boxed{} checker, NO sandbox).
#   - DAPO = grpo advantage + decoupled clip (eps 0.2 / 0.28), kl 0.
# Colocate mode on physical NPUs 0-7: actor count is authoritative, rollout shares the same 8 visible NPUs.

pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "VLLM::" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
pkill -9 -f "from multiprocessing" 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
sleep 5

set -ex

# ============ NPU / megatron env (mirrors run-qwen3-30B-A3B-npu.sh) ============
export SLIME_SCRIPT_TRAIN_BACKEND=megatron
export PYTHONPATH="/workspace/Megatron-LM:/workspace/vime:/workspace/MindSpeed:$PYTHONPATH"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)
export DISABLE_L2_CACHE=1
export VLLM_ASCEND_ENABLE_NZ=0
# Our vllm is built 2 commits past the v0.21.0 tag → setuptools_scm reports
# 0.21.1.dev2, so vllm-ascend's vllm_version_is("0.21.0") guard would take the
# wrong import branch (expert_map_manager, which only exists on vllm main).
# Pin the version so vllm-ascend treats it as 0.21.0 (as in the working env).
export VLLM_VERSION=0.21.0
export QWEN36_CAUSAL_CONV1D_IMPL=triton
export WANDB_MODE=disabled   # wandb disabled (no auth/network needed)
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# [proxy] All RL traffic is in-cluster (vllm engines, router, health checks, weight sync).
# A leaked HTTP(S)_PROXY makes requests.get(node_ip:port/health) route through the proxy and
# hang forever in VLLMEngine._wait_server_healthy. Clear every variant (upper+lower) and put
# the node IP on no_proxy so nothing internal is proxied.
NODE_IP="$(hostname -I | awk '{print $1}')"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy="127.0.0.1,localhost,${NODE_IP}"
export NO_PROXY="${no_proxy}"

VIME_DIR=/workspace/vime
HF_CKPT=/mnt/weight/Qwen3.6-35B-A3B
REF_LOAD=/mnt/weight/Qwen3.6-35B-A3B_torch_dist
PROMPT_DATA=/mnt/share/l30055792/datasets/dapo-math-17k.jsonl

source "${VIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh"   # -> MODEL_ARGS

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${VIME_DIR}/runs/dapo_math_mtp_${STAMP}"
mkdir -p "${RUN_ROOT}"

# Start a single-node Ray head on the 8 visible NPUs — train.py connects via
# ray.init(address="auto"), so a head must already be running (this script only
# did `ray stop` at the top; the other run-*.sh scripts all start one here).
RAY_TEMP_DIR=/tmp/ray_dapo_math
rm -rf "${RAY_TEMP_DIR}"
ray start --head --node-ip-address "${NODE_IP}" \
  --num-gpus 8 --resources='{"NPU": 8}' \
  --temp-dir="${RAY_TEMP_DIR}" --dashboard-host=0.0.0.0 --disable-usage-stats

# Run from VIME_DIR, NOT /workspace: /workspace contains a `vllm/` repo dir that
# shadows the installed vllm as an empty namespace package (import vllm -> file None,
# "cannot import name 'SamplingParams'"). VIME_DIR has no such subdir.
cd "${VIME_DIR}"

python ${VIME_DIR}/train.py \
  --train-backend megatron \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --colocate \
  --rollout-num-gpus 8 \
  --rollout-num-gpus-per-engine 8 \
  ${MODEL_ARGS[@]} \
  --qwen-gdn-backend npu \
  \
  --hf-checkpoint ${HF_CKPT} \
  --ref-load ${REF_LOAD} \
  --megatron-to-hf-mode raw \
  \
  --prompt-data ${PROMPT_DATA} \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --rollout-shuffle \
  --rm-type deepscaler \
  \
  --rollout-backend vllm \
  --vllm-weight-sync-mode native \
  --vllm-gpu-memory-utilization 0.30 \
  --vllm-max-num-seqs 32 \
  --vllm-enable-sleep-mode \
  --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --vllm-max-model-len $((1024 * 18)) \
  \
  --num-rollout 50 \
  --rollout-batch-size 4 \
  --n-samples-per-prompt 4 \
  --rollout-max-response-len $((1024 * 16)) \
  --rollout-temperature 1.0 \
  --global-batch-size 8 \
  --balance-data \
  \
  --mtp-num-layers 1 \
  --enable-mtp-training \
  --mtp-loss-scaling-factor 0.2 \
  \
  --advantage-estimator grpo \
  --kl-loss-coef 0.0 \
  --kl-loss-type low_var_kl \
  --entropy-coef 0.0 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --optimizer-cpu-offload \
  --overlap-cpu-optimizer-d2h-h2d \
  --use-precision-aware-optimizer \
  \
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
  --max-tokens-per-gpu 1024 \
  --log-probs-chunk-size 1024 \
  \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --use-flash-attn \
  --no-gradient-accumulation-fusion \
  \
  --train-memory-margin-bytes 2147483648 \
  --distributed-timeout-minutes 60 \
  2>&1 | tee "${RUN_ROOT}/run.log"

echo "RUN_ROOT=${RUN_ROOT}"
