#!/usr/bin/env bash
# Qwen3.6-35B-A3B (GDN/MoE, Route B port) DAPO-math RL — reward curve variant on the first 8 Ascend NPUs.
#
# Key changes from base script:
#   --rollout-max-response-len 4096  (doubled from 2048, not OOM)
#   --num-rollout 80                 (more samples per step)
#   --global-batch-size 32           (larger batch, stable training)
#   --use-wandb --wandb-project ...  (wandb uploads)
#   CUDAGraph FULL=7
# Colocate mode on physical NPUs 0-7: actor count is authoritative, rollout shares the same 8 visible NPUs.

pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "VLLM::" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
pkill -9 -f "from multiprocessing" 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
sleep 5

set -ex

# ============ NPU / megatron env ============
export SLIME_SCRIPT_TRAIN_BACKEND=megatron
export PYTHONPATH="/root/Megatron-Bridge/src:/root/Megatron-LM/:/workspace/vime:$PYTHONPATH"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)
export DISABLE_L2_CACHE=1
export VLLM_ASCEND_ENABLE_NZ=0
export QWEN36_CAUSAL_CONV1D_IMPL=triton

NODE_IP="$(hostname -I | awk '{print $1}')"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy="127.0.0.1,localhost,${NODE_IP}"
export NO_PROXY="${no_proxy}"

VIME_DIR=/workspace/vime
HF_CKPT=/home/s50057377/Qwen3.6-35B-A3B
REF_LOAD=/home/s50057377/Qwen3.6-35B-A3B_torch_dist
PROMPT_DATA=/home/c00937190/dapo-math-17k.jsonl

source "${VIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh"   # -> MODEL_ARGS

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${VIME_DIR}/runs/dapo_reward_curve_${STAMP}"
mkdir -p "${RUN_ROOT}"

python ${VIME_DIR}/train.py \
  --train-backend megatron \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --colocate \
  --no-offload-train \
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
  --vllm-gpu-memory-utilization 0.25 \
  --vllm-enable-sleep-mode \
  --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --vllm-max-model-len $((1024 * 12)) \
  --vllm-max-num-seqs 32 \
  \
  --num-rollout 128 \
  --rollout-batch-size 4 \
  --n-samples-per-prompt 4 \
  --rollout-max-response-len $((1024 * 4)) \
  --rollout-temperature 1.0 \
  --global-batch-size 16 \
  --balance-data \
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
  --max-tokens-per-gpu 4096 \
  \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --use-flash-attn \
  --no-gradient-accumulation-fusion \
  \
  --train-memory-margin-bytes 2147483648 \
  \
  --use-wandb \
  --wandb-project vime-qwen36-dapo \
  --wandb-group dapo-reward-curve \
  2>&1 | tee "${RUN_ROOT}/run.log"

echo "RUN_ROOT=${RUN_ROOT}"
