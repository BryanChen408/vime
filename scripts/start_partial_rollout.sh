#!/usr/bin/env bash
# Opt-in wrapper; keeps the resource layout/model/dataset options of the main launcher.
# Vime negotiates session partial mode with Polar during bootstrap; no Polar-side
# partial environment variables are needed. Both Polar services must support it.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export POLAR_PARTIAL_ROLLOUT=1
export POLAR_POLICY_TRANSITION_ENABLED=1
export TRAIN_ENTRY=train.py
export FEAT_SYNC_ROLLOUT=0  # persistent agent scheduler; training itself stays synchronous
export FEAT_OFFLOAD=1
export POLAR_DISABLE_TIS=0
export POLAR_MAX_OFF_POLICY_STEPS=1
export POLAR_MIN_COMPLETE_ACCEPT_FRACTION=1.0
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export GLOBAL_BATCH_SIZE="$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))"
export POLAR_MAX_ACTIVE_SESSIONS="${POLAR_MAX_ACTIVE_SESSIONS:-64}"
export POLAR_MAX_OWNED_GROUPS="${POLAR_MAX_OWNED_GROUPS:-24}"
export POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-2}"
export POLAR_DRAIN_SESSIONS=0
exec bash "${SCRIPT_DIR}/run-qwen36-35b-polar-multi-pd.sh" "$@"
