#!/bin/bash

# Stabilised PPO variant of run-qwen3-30B-A3B-npu.sh, for multi-turn agentic rollouts.
#
# On top of the plain critic setup it uses the options that keep the value function usable
# when the trajectory is long and only partly written by the policy:
#
#   - the value head trains alone first, so early advantages are not noise
#   - the critic updates twice per policy update, to keep up with a policy that has moved
#   - the critic trains only its experts and value head, which is cheaper and drifts less
#   - tool output is kept out of the GAE recursion, since the critic never learned it
#   - the credit horizon follows the response length instead of being fixed
#   - clipping is asymmetric and wide, which suits code rewards; try 0.3/5.0 for maths
#
# The critic usually wants a larger learning rate than the policy. --lr is shared, so give it
# its own through --megatron-config-path.

# for rerun the task
pkill -9 -f '[v]llm serve|VLL[M]::'
pkill -9 -f VLLM
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 redis

set -ex

export PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export DISABLE_L2_CACHE=1
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_USE_AOT_COMPILE=0
export PYTHONPATH="/root/Megatron-Bridge/src:/root/Megatron-LM/:${PYTHONPATH:-}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"

DATA_ROOT="${DATA_ROOT:-/root}"

CKPT_ARGS=(
   --hf-checkpoint ${DATA_ROOT}/weights/Qwen3-30B-A3B/
   --load ${DATA_ROOT}/weights/Qwen3-30B-A3B/
   --ref-load ${DATA_ROOT}/weights/Qwen3-30B-A3B/
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_ROOT}/datasets/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len $((1024 * 8))
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime ${DATA_ROOT}/datasets/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
)

PPO_ARGS=(
   --advantage-estimator ppo
   --num-critic-only-steps 10
   --value-clip 0.2
   --gamma 1.0
   --kl-coef 0
   # Must stay 0: the critic has no reference model, so a reward-side KL has nothing to
   # subtract. Argument parsing rejects anything else.
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --use-tis

   # Wide and asymmetric: a code reward is sparse enough that a tight upper clip stalls it.
   --eps-clip 0.8
   --eps-clip-high 3.0

   --critic-update-steps 2
   --critic-only-train-params-name-list mlp.experts output_layer

   # Both belong together: skipping tool output only changes anything below lambda 1, and the
   # adaptive lambda is what puts it there.
   --skip-observation-gae
   --gae-lambda-alpha 1.5
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

VLLM_ARGS=(
   --rollout-num-gpus-per-engine 4
   --vllm-gpu-memory-utilization 0.7
   --vllm-cudagraph-capture-sizes 1 2 4 8 $(seq 16 8 256)
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash

   --use-flash-attn
   --no-gradient-accumulation-fusion
)

ray start --head --node-ip-address 127.0.0.1 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

ray job submit --address="http://127.0.0.1:8265" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${PPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${VLLM_ARGS[@]} \
   ${MISC_ARGS[@]}
