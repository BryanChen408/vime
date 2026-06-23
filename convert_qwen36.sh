#!/bin/bash
# Convert Qwen3.6-35B-A3B HF safetensors -> Megatron torch_dist (mcore fused GDN) for --ref-load.
# 35B MoE can't fit one 64GB NPU (megatron get_model moves model to device unconditionally),
# so shard across 8 NPUs via torchrun: the convert script auto-sets PP=world_size (PP=8 here,
# ~5 layers/rank ~9GB). dist-ckpt loads parallelism-agnostically at train time.
set -ex

export VIME_DIR=/workspace/vime
export PYTHONPATH="/root/Megatron-LM:/root/Megatron-Bridge/src:${VIME_DIR}:${PYTHONPATH}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export QWEN36_CAUSAL_CONV1D_IMPL=triton

HF_CKPT=/home/s50057377/Qwen3.6-35B-A3B
SAVE=/home/s50057377/Qwen3.6-35B-A3B_torch_dist

# MODEL_ARGS for Qwen3.6-35B-A3B (== vime scripts/models/qwen3.5-35B-A3B.sh; same HF arch qwen3_5_moe)
source ${VIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh

cd ${VIME_DIR}
torchrun --nproc_per_node 8 --master_port 29520 \
  tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint ${HF_CKPT} \
  --save ${SAVE} \
  --qwen-gdn-backend npu \
  --tensor-model-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1
