#!/usr/bin/env bash
# KernelGym kernel-generation RL on a single 8-card Ascend NPU node.
#
# Topology ("8-card train + 4-infer/4-verify"):
#   * vime trains + serves rollout on all 8 cards, colocated (vLLM sleep-mode
#     offloads the actor during generation and vice-versa).
#   * The kernelGym verifier runs on the SAME node. Its workers use little GPU
#     memory, so they coexist with vLLM inference (no time-sharing needed).
#     Point them at e.g. GPU_DEVICES=[4,5,6,7] for a 4-infer / 4-verify split,
#     or share all 8 cards.
#   * vime talks to kernelGym only over HTTP (kernelgym.server_url in
#     kernelgym_config.yaml), so train/infer and verify stay decoupled.
#
# Prerequisites
#   1) Start kernelGym on this node (or a reachable host):
#        cd /path/to/kernelGym-NPU && ./start_all_with_monitor.sh
#        curl -s http://127.0.0.1:8002/health        # sanity check
#      Then set kernelgym.server_url in kernelgym_config.yaml accordingly.
#   2) Prepare data:
#        python -m examples.kernelgym.prepare_data \
#          --input /path/to/rllm-lilac/data/drkernel_rl_data.jsonl \
#          --output examples/kernelgym/data/kernelgym_train.jsonl
#   3) Place a Qwen3-8B checkpoint under ${MODEL_ROOT}/models/${MODEL_NAME}.
#
# Override any VAR via env, e.g. ROLLOUT_TP=2 MODEL_NAME=Qwen3-8B bash run_kernelgym_grpo_npu.sh

set -ex

export SLIME_SCRIPT_TRAIN_BACKEND=megatron
export PYTHONPATH="/root/Megatron-Bridge/src:/root/Megatron-LM/:${PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)
export DISABLE_L2_CACHE=1
export VLLM_ASCEND_ENABLE_NZ=0
# kernelGym is in-cluster; never route its HTTP through a proxy.
export no_proxy="127.0.0.1,localhost,${no_proxy}"
export NO_PROXY="${no_proxy}"

VIME_DIR="${VIME_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-/root}"
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
source "${VIME_DIR}/scripts/models/qwen3-8B.sh"

DATA_DIR="${DATA_DIR:-${VIME_DIR}/examples/kernelgym/data}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/kernelgym_train.jsonl}"
LOG_FILE="${LOG_FILE:-${VIME_DIR}/train_kernelgym_qwen3_8b.log}"

# GPU topology: rollout-num-gpus == actor gpus => colocate (8-card).
ACTOR_GPUS="${ACTOR_GPUS:-8}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-8}"
ROLLOUT_TP="${ROLLOUT_TP:-4}"   # inference tensor-parallel per vLLM engine
TRAIN_TP="${TRAIN_TP:-4}"       # training tensor-parallel

python "${VIME_DIR}/train.py" \
  --train-backend megatron \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node ${ACTOR_GPUS} \
  --rollout-num-gpus ${ROLLOUT_GPUS} \
  --rollout-num-gpus-per-engine ${ROLLOUT_TP} \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint ${MODEL_ROOT}/models/${MODEL_NAME}/ \
  --load ${MODEL_ROOT}/models/${MODEL_NAME} \
  --ref-load ${MODEL_ROOT}/models/${MODEL_NAME} \
  --megatron-to-hf-mode bridge \
  --custom-generate-function-path examples.kernelgym.rollout.generate \
  --custom-config-path ${VIME_DIR}/examples/kernelgym/kernelgym_config.yaml \
  --prompt-data ${TRAIN_DATA} \
  --input-key prompt \
  --label-key label \
  --metadata-key metadata \
  --apply-chat-template \
  --rollout-shuffle \
  --rollout-backend vllm \
  --vllm-weight-sync-mode native \
  --vllm-gpu-memory-utilization 0.55 \
  --vllm-enable-sleep-mode \
  --vllm-max-model-len 24576 \
  --num-rollout 500 \
  --rollout-batch-size 8 \
  --n-samples-per-prompt 8 \
  --rollout-max-context-len 24576 \
  --rollout-max-response-len 8192 \
  --rollout-temperature 1.0 \
  --global-batch-size 64 \
  --balance-data \
  --advantage-estimator grpo \
  --kl-loss-coef 0.0 \
  --kl-loss-type low_var_kl \
  --kl-coef 0.0 \
  --entropy-coef 0.0 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --tensor-model-parallel-size ${TRAIN_TP} \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 16384 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --micro-batch-size 1 \
  --use-flash-attn \
  --train-memory-margin-bytes 2147483648 \
  2>&1 | tee -a "${LOG_FILE}"
