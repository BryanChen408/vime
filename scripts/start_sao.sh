#!/bin/bash
# Smoke run for run-qwen36-35b-polar-sao.sh: small batches and a short critic warmup, enough
# to see the critic-only phase hand over to joint training. Not a tuning configuration.
#
# Expects the polar operator to be up already and cards 4-15 to be free.
#
# GLOBAL_BATCH_SIZE has to equal ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT, otherwise
# train_iters lands on zero and the scheduler refuses to build.
#
# MAX_TOKENS_PER_GPU matters here: the adapter drops traces above it times the CP size, and
# the default is far below what these rollouts produce, so leaving it out silently discards
# every long trace and the run never progresses.
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

CURRENT_IP="${CURRENT_IP:-80.48.5.88}"  MASTER_ADDR="${MASTER_ADDR:-80.48.5.88}"  NNODES=1  NPUS_PER_NODE=12  SOCKET_IFNAME="${SOCKET_IFNAME:-ens1f3}" \
RESOURCE_LAYOUT="${RESOURCE_LAYOUT:-${SCRIPT_DIR}/resource_layout_actor_domain2.yaml}" \
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}" \
VIME_MEM_PROBE=1  VIME_EMPTY_CACHE_PER_STEP=1  FEAT_TRAIN_EXPANDABLE=1 \
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"  N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"  NUM_ROLLOUT="${NUM_ROLLOUT:-200}" \
NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-2}" \
CRITIC_UPDATE_STEPS="${CRITIC_UPDATE_STEPS:-2}" \
CRITIC_CONFIG="${CRITIC_CONFIG:-${SCRIPT_DIR}/sao_critic_config.yaml}" \
FEAT_DP_EXTERNAL_LB=0  FEAT_BALANCE_SCHED=0  FEAT_LB_PROXY=1  FEAT_CROSS_DP_EP=0 \
FEAT_ROLLOUT_EP=0  FEAT_FLASHCOMM1=0  FEAT_PREFIX_CACHE=1  FEAT_MULTISTREAM_SHARED_EXPERT=1  FEAT_STATIC_KERNEL=1  FEAT_HCCL_AIV=1 \
bash "${SCRIPT_DIR}/run-qwen36-35b-polar-sao.sh"
