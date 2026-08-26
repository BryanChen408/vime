#!/bin/bash
# 纯评测入口：这里只配置权重、数据和评测规模。

# 评测 checkpoint 列表：每个条目写成「路径:TAG」，按列表顺序逐个评测。
CHECKPOINTS=(
  /home/docker/model_weights/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t_torch_dist/release:Qwen3.6-35B-A3B-sft
  /home/docker/model_weights/Qwen3.6-35B-A3B_fused_torch_dist/release:Qwen3.6-35B-A3B
  /home/docker/tmp/checkpoints/qwen36_35b_dapo_ascendc/version__20260821-1525/iter_0000009/iter_0000009:dapo-iter-09
)

# 用于补齐 config/tokenizer 的原始 HF 模型。
ORIGIN_HF_DIR=/home/docker/model_weights/Qwen3.6-35B-A3B
HF_STAGING_ROOT=/home/docker/vime_evaluation/hf_staging

# DATASET_TAG=holdout16_hard
# EVAL_DATA=/home/docker/datasets/rl_ops1_2_simple/holdout16_hard.jsonl
# OPERATOR_TASKS_DIR=/home/docker/datasets/rl_ops1_2_simple/op_tasks
DATASET_TAG=cudallm-ascendc
EVAL_DATA=/home/docker/datasets/op_assets_cudallm_filtered189/operator_tasks.ascendc.holdout.jsonl
OPERATOR_TASKS_DIR=/home/docker/datasets/op_assets_cudallm_filtered189/op_tasks

ROLLOUT_BATCH_SIZE=16
ROLLOUT_MAX_ACTIVE_SESSIONS=32

BASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
[ "${#CHECKPOINTS[@]}" -ge 1 ] || {
  echo "CHECKPOINTS must contain at least one entry" >&2
  exit 1
}

if [ ! -f "${ORIGIN_HF_DIR}/config.json" ]; then
  for ENTRY in "${CHECKPOINTS[@]}"; do
    candidate="${ENTRY%%:*}"
    if [ -f "${candidate}/config.json" ] && [ -f "${candidate}/model.safetensors.index.json" ]; then
      ORIGIN_HF_DIR="${candidate}"
      echo "[convert] use HF checkpoint as origin assets: ${ORIGIN_HF_DIR}"
      break
    fi
  done
fi
[ -f "${ORIGIN_HF_DIR}/config.json" ] || {
  echo "Origin HF model is invalid: ${ORIGIN_HF_DIR}" >&2
  echo "Set ORIGIN_HF_DIR to an existing HF model directory containing config.json." >&2
  exit 1
}

mkdir -p "${HF_STAGING_ROOT}"
for ENTRY in "${CHECKPOINTS[@]}"; do
  TORCH_DIST_CKPT="${ENTRY%%:*}"
  MODEL_TAG="${ENTRY##*:}"
  [ -d "${TORCH_DIST_CKPT}" ] || {
    echo "Checkpoint directory not found: ${TORCH_DIST_CKPT}" >&2
    exit 1
  }
  STAGED_CKPT=0
  if [ -f "${TORCH_DIST_CKPT}/config.json" ] && [ -f "${TORCH_DIST_CKPT}/model.safetensors.index.json" ]; then
    HF_CKPT="${TORCH_DIST_CKPT}"
    echo "[evaluation] use existing HF checkpoint=${HF_CKPT}"
  else
    HF_CKPT="${HF_STAGING_ROOT}/${MODEL_TAG}"
    bash "${BASE_DIR}/convert_checkpoint.sh" "${TORCH_DIST_CKPT}" "${HF_CKPT}" "${ORIGIN_HF_DIR}"
    STAGED_CKPT=1
  fi
  source "${BASE_DIR}/run_base.sh"
  if [ "${STAGED_CKPT}" -eq 1 ]; then
    case "${HF_CKPT}" in
      "${HF_STAGING_ROOT}"/*) rm -rf -- "${HF_CKPT}" ;;
      *) echo "Refusing to remove non-staging path: ${HF_CKPT}" >&2; exit 1 ;;
    esac
  fi
done
