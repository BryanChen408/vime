#!/usr/bin/env bash
# =============================================================================
# Qwen3.5-9B · DAPO-Math GRPO RL · Ascend 910B3 全 8 卡 · 出效果
# TP2 × DP4 (8卡), vLLM per-engine 2卡(TP2, 4 engines), 32k 响应长度, util 0.7。
# storage-resize B 全卸(param+grad, rollout 残留逼近0) → 给 vLLM 32k KV 腾显存。
# wandb: 本机 cntrain21 localhost:8080。
# =============================================================================
set -ex

pkill -9 -f "vime/train.py" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
ray stop --force 2>/dev/null || true
sleep 5

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export SLIME_SCRIPT_TRAIN_BACKEND=megatron
# storage-resize hook(npu_mem_offload)放 PYTHONPATH 最前 → 注入 NPUWeightOffloader storage-resize。
# committed offloader 的 _release_ddp_buffers 是假卸载(reassign,不真释放),offload 后 used 不降 →
# vLLM wake_up 分配 31GB KV cache 时 OOM。hook 用 storage().resize_(0) 真释放 DDP flat buffer。
# 此前误判"hook 多 step 死锁"实为 CP ring crash(已在 actor.py 修)。B 模式已在 210652 跑通完整 rollout。
export PYTHONPATH="/workspace/vime/scripts/npu_mem_offload:/workspace/Megatron-Bridge-slime/src:/workspace/Megatron-LM/:/workspace/vime:$PYTHONPATH"
export VIME_OFFLOAD_PARAM_BUFFER=1   # B: param+grad 全卸, rollout 残留逼近0

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60100
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61100
export HYDRA_FULL_ERROR=1
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)
export DISABLE_L2_CACHE=1
export VLLM_ASCEND_ENABLE_NZ=0
export QWEN36_CAUSAL_CONV1D_IMPL=triton
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1
# Ascend 异步算子下发队列:默认开启会让自定义算子(GDN/ring attention)出现数值不稳定
# (backward NaN grad)。训练必须关掉走同步下发。
export TASK_QUEUE_ENABLE=0

export TRITON_CACHE_DIR=/tmp/triton_cache_qwen9b8
export TRITON_HOME=/tmp/triton_home_qwen9b8
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache_qwen9b8
mkdir -p "$TRITON_CACHE_DIR" "$TRITON_HOME" "$TORCHINDUCTOR_CACHE_DIR"

NODE_IP="$(hostname -I | awk '{print $1}')"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy="127.0.0.1,localhost,${NODE_IP}"
export NO_PROXY="${no_proxy}"

VIME_DIR=/workspace/vime
MODEL_ROOT=/root
HF_CKPT=${MODEL_ROOT}/models/Qwen3.5-9B
REF_LOAD=${MODEL_ROOT}/models/Qwen3.5-9B_torch_dist
PROMPT_DATA=${MODEL_ROOT}/datasets/dapo-math-17k/dapo-math-17k.jsonl
source "${VIME_DIR}/scripts/models/qwen3.5-9B.sh"   # → MODEL_ARGS

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${VIME_DIR}/runs/qwen35_9b_8card_${STAMP}"
mkdir -p "${RUN_ROOT}"

python ${VIME_DIR}/train.py \
  --train-backend megatron \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --colocate \
  --rollout-num-gpus 8 \
  --rollout-num-gpus-per-engine 2 \
  ${MODEL_ARGS[@]} \
  --qwen-gdn-backend npu \
  --linear-key-head-dim 128 \
  --linear-value-head-dim 128 \
  --linear-num-key-heads 16 \
  --linear-num-value-heads 32 \
  --linear-conv-kernel-dim 4 \
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
  --rm-type math \
  --save-debug-rollout-data "${RUN_ROOT}/rollout_data/{rollout_id}.pt" \
  \
  --rollout-backend vllm \
  --vllm-weight-sync-mode native \
  --vllm-gpu-memory-utilization 0.7 \
  --vllm-max-num-seqs 64 \
  --vllm-no-enable-prefix-caching \
  --vllm-enable-sleep-mode \
  --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --vllm-max-model-len 40960 \
  \
  --num-rollout 20 \
  --rollout-batch-size 32 \
  --n-samples-per-prompt 8 \
  --rollout-max-response-len 32768 \
  --rollout-temperature 1.0 \
  --global-batch-size 256 \
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
  --use-precision-aware-optimizer \
  --overlap-cpu-optimizer-d2h-h2d \
  \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 2 \
  --expert-model-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 40960 \
  --log-probs-chunk-size 1024 \
  \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --use-flash-attn \
  --no-gradient-accumulation-fusion \
  \
  --use-wandb \
  --wandb-host http://localhost:8080 \
  --wandb-key ${WANDB_KEY} \
  --wandb-project vime-dapo-math-9b \
  --wandb-group qwen35-9b-8card \
  \
  --train-memory-margin-bytes 2147483648 \
  --distributed-timeout-minutes 60 \
  2>&1 | tee "${RUN_ROOT}/run.log"

echo "RUN_ROOT=${RUN_ROOT}"
