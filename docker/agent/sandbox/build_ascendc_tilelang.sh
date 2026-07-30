#!/bin/bash
# =============================================================================
# Build the ascendc-tilelang Polar sandbox image.
# The TileLang wheel is compiled from source inside Dockerfile Stage 1 — no
# manual pre-build needed.
#
# Usage:
#   bash build_ascendc_tilelang.sh
#   TAG=ascendc-tilelang:v2 bash build_ascendc_tilelang.sh
#   TILELANG_SRC=/path/to/tilelang-ascend bash build_ascendc_tilelang.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TILELANG_SRC="${TILELANG_SRC:-/home/docker/tilelang-ascend}"
TAG="${TAG:-ascendc-tilelang:v1}"

# ── Step 1: submodule checkout ───────────────────────────────────────────────
echo "[1/4] git submodule update (tilelang 3rdparty)"
if ! git -C "${TILELANG_SRC}" rev-parse --git-dir >/dev/null 2>&1; then
  echo "[ERROR] ${TILELANG_SRC} is not a git repo" >&2
  exit 1
fi
git -C "${TILELANG_SRC}" submodule update --init --recursive
echo "  submodule status:"
git -C "${TILELANG_SRC}" submodule status | head -8

# ── Step 2: prepare build context (temp dir, auto-cleaned) ───────────────────
BUILD_CTX="$(mktemp -d /tmp/ascendc-tilelang-ctx.XXXXXX)"
trap 'echo "[cleanup] removing temp build context"; rm -rf "${BUILD_CTX}"' EXIT

echo ""
echo "[2/4] copy source into build context (plain cp, exclude .git to shrink)"
# Use cp (not rsync). Plain `cp -r` (no -a/-p) does NOT preserve ownership, so
# the copy carries no root/uid ownership. cp has no --exclude, so enumerate the
# top-level entries and skip the (large) .git / .gitmodules; submodule .git are
# tiny pointer files and the Dockerfile re-inits 3rdparty/tvm anyway.
mkdir -p "${BUILD_CTX}/tilelang-src"
( shopt -s dotglob
  for item in "${TILELANG_SRC}"/*; do
    name="$(basename "$item")"
    [[ "$name" == ".git" || "$name" == ".gitmodules" ]] && continue
    cp -r "$item" "${BUILD_CTX}/tilelang-src/"
  done )
# strip build artifacts + any nested submodule .git
find "${BUILD_CTX}/tilelang-src" -depth \
     \( -type d \( -name '__pycache__' -o -name '*.egg-info' \
                   -o -name 'build' -o -name 'dist' -o -name '.git' \) \
        -o -name '*.pyc' \) \
     -exec rm -rf {} + 2>/dev/null || true

cp "${SCRIPT_DIR}/Dockerfile.ascendc-tilelang" "${BUILD_CTX}/"
echo "  build context size: $(du -sh "${BUILD_CTX}" | cut -f1)"

# ── Step 3: docker build ─────────────────────────────────────────────────────
echo ""
echo "[3/4] docker build -> ${TAG}"
docker build \
  --progress=plain \
  -f "${BUILD_CTX}/Dockerfile.ascendc-tilelang" \
  -t "${TAG}" \
  "${BUILD_CTX}"

# ── Step 4: verify ───────────────────────────────────────────────────────────
echo ""
echo "[4/4] verify ${TAG}"
docker run --rm "${TAG}" bash -lc '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null ||
  source /usr/local/Ascend/cann-9.0.0/set_env.sh     2>/dev/null || true
  echo -n "cmake       : "; cmake --version | head -1
  echo -n "ccec        : "; which ccec
  echo -n "bishengir   : "; which bishengir-compile
  echo -n "claude      : "; claude --version 2>/dev/null | head -1 || echo "(needs network activation)"
  python3 -c "
import torch, torch_npu, tilelang, os
print(\"torch       :\", torch.__version__)
print(\"torch_npu   :\", torch_npu.__version__)
print(\"tilelang    :\", tilelang.__version__)
print(\"ACL_OP_INIT_MODE:\", os.environ.get(\"ACL_OP_INIT_MODE\", \"unset\"))
"
'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [ok] ${TAG} built"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next: point the polar profile image at ${TAG} and restart polar:"
echo ""
echo "  sed -i 's|image: ascendc-sandbox:v1|image: ${TAG}|g' \\"
echo "    ProRL-Agent-Server/deploy/ascend_operator/profile.ascendc.yaml \\"
echo "    ProRL-Agent-Server/deploy/ascend_operator/profile.t2a.yaml"
echo ""
echo "  cd ProRL-Agent-Server"
echo "  POLAR_PROFILE=deploy/ascend_operator/profile.ascendc.yaml \\"
echo "  POLAR_RUN_ID=polar_\$(date +%Y%m%d_%H%M%S) \\"
echo "  NO_PROXY=127.0.0.1,localhost,80.48.5.88,80.48.5.52 \\"
echo "  no_proxy=127.0.0.1,localhost,80.48.5.88,80.48.5.52 \\"
echo "    bash deploy/ascend_operator/restart_polar_host.sh"
