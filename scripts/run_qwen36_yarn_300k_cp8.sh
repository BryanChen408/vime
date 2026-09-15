#!/usr/bin/env bash
# YaRN 300K validation: a diverse-token TP2/CP8 training replay.
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export VALIDATION_STEP=300k
export CP_SIZE=8
export TP_SIZE=2
export PP_SIZE=1
export EP_SIZE=8
export ACTOR_NUM_GPUS=16
export GLOBAL_BATCH_SIZE=2
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export SEQ_LENGTH=300000
export MAX_TOKENS_PER_GPU=37500
export ROLLOUT_DATA="${ROLLOUT_DATA:-${VIME_ROOT}/output/validation_inputs/qwen36_yarn_300000_2samples_diverse.pt}"
export RUN_ID="${RUN_ID:-qwen36_yarn_300k_cp8_$(date +%Y%m%d_%H%M%S)}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_yarn_300k}"

exec "${SCRIPT_DIR}/run_qwen36_yarn_step5_cp1.sh"
