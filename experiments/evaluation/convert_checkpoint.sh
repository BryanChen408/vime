#!/bin/bash
# 把单个 Megatron torch_dist checkpoint 转成 vLLM 可加载的 HF checkpoint。
# bash experiments/evaluation/convert_checkpoint.sh /home/docker/tmp/checkpoints/Qwen3.6-35B-A3B_ppo_critic/iter_0000004 /home/docker/vime_evaluation/hf_staging/iter_0000004 /home/docker/model_weights/Qwen3.6-35B-A3B

set -euo pipefail
[ "$#" -eq 3 ] || {
  echo "Usage: $0 INPUT_DIR OUTPUT_DIR ORIGIN_HF_DIR" >&2
  exit 1
}

INPUT_DIR=$1
OUTPUT_DIR=$2
ORIGIN_HF_DIR=$3
BASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${BASE_DIR}/../.." &>/dev/null && pwd)"
cd "${VIME_ROOT}"

[ -f "${INPUT_DIR}/common.pt" ] && [ -f "${INPUT_DIR}/.metadata" ] || {
  echo "Not a torch_dist checkpoint: ${INPUT_DIR}" >&2
  exit 1
}
[ -f "${ORIGIN_HF_DIR}/config.json" ] || {
  echo "Origin HF model is invalid: ${ORIGIN_HF_DIR}" >&2
  exit 1
}
[ ! -e "${OUTPUT_DIR}" ] || {
  if [ -f "${OUTPUT_DIR}/config.json" ] && [ -f "${OUTPUT_DIR}/model.safetensors.index.json" ]; then
    echo "[convert] reuse completed checkpoint: ${OUTPUT_DIR}"
    exit 0
  fi
  echo "Incomplete conversion output already exists: ${OUTPUT_DIR}" >&2
  exit 1
}

echo "[convert] ${INPUT_DIR} -> ${OUTPUT_DIR}"
python3 experiments/evaluation/convert/convert_torch_dist_to_hf_eval.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --origin-hf-dir "${ORIGIN_HF_DIR}"

[ -f "${OUTPUT_DIR}/config.json" ] && [ -f "${OUTPUT_DIR}/model.safetensors.index.json" ] || {
  echo "Converted HF checkpoint is incomplete: ${OUTPUT_DIR}" >&2
  exit 1
}
