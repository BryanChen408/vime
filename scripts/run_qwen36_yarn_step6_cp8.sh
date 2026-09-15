#!/usr/bin/env bash
# YaRN validation step 6: replay the step-5 tokens with production CP=8.
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export VALIDATION_STEP=step6
export CP_SIZE=8
export TP_SIZE=2
export PP_SIZE=1
export EP_SIZE=8
export ACTOR_NUM_GPUS=16
export GLOBAL_BATCH_SIZE=8
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export RUN_ID=${RUN_ID:-qwen36_yarn_step6_cp8_$(date +%Y%m%d_%H%M%S)}
export RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/ray_qwen36_yarn_step6_cp8}

exec "${SCRIPT_DIR}/run_qwen36_yarn_step5_cp1.sh"
