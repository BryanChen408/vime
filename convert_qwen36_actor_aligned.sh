#!/usr/bin/env bash
# Convert /home/weight/Qwen3.6-35B-A3B HF safetensors -> Megatron torch_dist
# using the SAME actor-side model-structure flags as the current 16-card async
# training launcher, so the resulting ref-load is structurally aligned with the
# actor training model and avoids checkpoint/model-spec mismatches at load time.
set -eo pipefail
set -x

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
set -u

export VIME_DIR=/workspace/vime
export PYTHONPATH="/workspace/Megatron-LM:/workspace/Megatron-Bridge-slime/src:/workspace/vime:/workspace/MindSpeed:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7,8,9,10,11
export CUDA_DEVICE_MAX_CONNECTIONS=1
export QWEN36_CAUSAL_CONV1D_IMPL=triton
export TORCHDYNAMO_DISABLE=1
export TASK_QUEUE_ENABLE=0
export VLLM_ASCEND_ENABLE_NZ=0
export MASTER_PORT=${MASTER_PORT:-29520}

HF_CKPT=/home/weight/Qwen3.6-35B-A3B
SAVE=${SAVE:-/home/weight/Qwen3.6-35B-A3B_torch_dist_actor_aligned_v2}

# Reuse the same base actor model args as the 16-card launcher.
source ${VIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh

cd ${VIME_DIR}
torchrun --nproc_per_node 8 --master_port ${MASTER_PORT} \
  tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint ${HF_CKPT} \
  --save ${SAVE} \
  --qwen-gdn-backend npu \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 8 \
  --mtp-num-layers 1 \
  --optimization-level 0 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1
