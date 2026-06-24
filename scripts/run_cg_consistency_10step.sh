#!/usr/bin/env bash
# Qwen3.6-35B-A3B DAPO-math RL — 10-step consistency validation with CUDAGraph
#   - Measures throughput with cudagraph enabled
#   - Saves per-token logprobs for train-inference consistency analysis
#   - Logs to wandb
#
# Key metrics automatically logged by vime:
#   train_rollout_logprob_abs_diff — mean absolute diff between train & rollout logprobs
#   train/log_prob, train/pg_loss, train/entropy_loss, train/grad_norm
#
# Post-hoc analysis (provided by separate consistency script):
#   quantiles, scatter plots, Pearson/Spearman correlation, cosine similarity

set -ex

# ============ Cleanup ============
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "VLLM::" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
pkill -9 -f "from multiprocessing" 2>/dev/null || true
pkill -9 -f "train.py" 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
sleep 5

# ============ NPU / megatron env ============
export SLIME_SCRIPT_TRAIN_BACKEND=megatron
export PYTHONPATH="/root/Megatron-Bridge/src:/root/Megatron-LM/:/workspace/vime:$PYTHONPATH"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)
export DISABLE_L2_CACHE=1
export VLLM_ASCEND_ENABLE_NZ=0
export QWEN36_CAUSAL_CONV1D_IMPL=triton
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFERSIZE=1024

# Save per-token logprobs for consistency analysis (each rank saves its shard)
TIS_DIR=/workspace/vime/runs/tis_logprobs_$(date +%Y%m%d_%H%M%S)
mkdir -p "${TIS_DIR}"
export VIME_SAVE_TIS_LOGPROBS="${TIS_DIR}"

# proxy cleanup
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
RUN_ROOT="${VIME_DIR}/runs/dapo_cg_consistency_${STAMP}"
mkdir -p "${RUN_ROOT}"

# ============ Launch ============
# 40 prompts × 4 samples = 160 total, global_batch_size=16 → 10 training steps
export TORCHDYNAMO_DISABLE=1
python ${VIME_DIR}/train.py \
  --train-backend megatron \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 16 \
  --colocate \
  --no-offload-train \
  --rollout-num-gpus 16 \
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
  --vllm-enable-sleep-mode \
  --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --vllm-max-model-len $((1024 * 12)) \
  --vllm-max-num-seqs 32 \
  \
  --num-rollout 40 \
  --rollout-batch-size 4 \
  --n-samples-per-prompt 4 \
  --rollout-max-response-len $((1024 * 2)) \
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
  --tensor-model-parallel-size 2 \
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
  --wandb-group cg-consistency-check \
  2>&1 | tee "${RUN_ROOT}/run.log"

RC=$?
echo "TRAIN_EXIT_CODE=${RC}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "TIS_DIR=${TIS_DIR}"

# Save the TIS directory path for post-hoc analysis
echo "${TIS_DIR}" > "${RUN_ROOT}/tis_dir.txt"
echo "${RUN_ROOT}" > /tmp/last_consistency_run.txt