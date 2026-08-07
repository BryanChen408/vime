#!/usr/bin/env bash
set -eo pipefail

# Example training launcher showing how VIME launchers can hard-require a W&B-ready preflight.

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
set -u

source /workspace/vime/scripts/common/wandb_ready.sh

WANDB_HOST="${WANDB_HOST:-http://127.0.0.1:8088}"
WANDB_PROJECT="${WANDB_PROJECT:-vime-example}"
WANDB_GROUP="${WANDB_GROUP:-example-run}"
RUN_ROOT="${RUN_ROOT:-./runs/example}"
WANDB_DIR="${RUN_ROOT}/wandb"
mkdir -p "${WANDB_DIR}"

assert_wandb_ready "${WANDB_HOST}"

python /workspace/vime/train_async.py \
  --use-wandb \
  --wandb-mode "${WANDB_MODE:-online}" \
  --wandb-host "${WANDB_HOST}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-dir "${WANDB_DIR}" \
  "$@"

# Notes for future launcher authors:
# 1. Canonical W&B CLI args live in vime/utils/arguments.py.
# 2. Primary/secondary W&B bootstrap lives in vime/utils/wandb_utils.py.
# 3. Training entrypoints should only use init_tracking / finish_tracking.
# 4. Self-hosted W&B should be injected via --wandb-host, not hard-coded in Python.
# 5. If a launcher wants to hard-require W&B, do a shell preflight before Python starts.
# 6. If offline operation is desired, use WANDB_MODE=offline or disabled and keep the helper sourced.
