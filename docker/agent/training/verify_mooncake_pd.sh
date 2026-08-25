#!/bin/bash
# =============================================================================
# verify_mooncake_pd.sh — verify the Mooncake PD data plane inside a container
# built from Dockerfile.release-pd.
#
# Run inside the container (an NPU host is recommended — the ASCEND_DIRECT
# libs pull CANN runtime libraries):
#
#   docker run -it --device=/dev/davinci0 ... vime:a3-pd-release bash
#   bash /workspace/verify_mooncake_pd.sh
#
# Checks, in order (any failure exits non-zero with a diagnosis):
#   1. /workspace/Mooncake source tree present (PD-image marker);
#   2. mooncake python bindings import (installed by `make install` at build);
#   3. transfer-engine shared libraries resolve via ldd (ASCEND_DIRECT path);
#   4. vllm's KV connector registry contains MooncakeConnectorV1 — the exact
#      probe vime/backends/vllm_utils/vllm_engine.py runs before starting PD.
#
# Note: the image build already runs checks 2/4 as a build-time smoke step;
# this script exists for on-host re-verification after `docker commit`, driver
# mounts, or CANN env changes. Exits 0 (skip) on non-PD images.
# =============================================================================
set -eo pipefail

echo "=== Mooncake PD data plane verification ==="

# --- 1) PD-image marker ------------------------------------------------------
if [ ! -d /workspace/Mooncake ]; then
  echo "SKIP: /workspace/Mooncake not found — this is not a PD image"
  exit 0
fi
echo "OK: /workspace/Mooncake present (PD image)"

# --- CANN environment (probe both known layouts) -----------------------------
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null \
  || source /usr/local/Ascend/cann-9.0.0/set_env.sh
set -u

# --- 2) python bindings ------------------------------------------------------
python3 -c "import mooncake; print('OK: mooncake python bindings import')"

# --- 3) shared-library linkage ----------------------------------------------
mooncake_so="$(find /usr/local/lib /usr/local/lib64 \
    \( -name 'lib*mooncake*.so*' -o -name 'lib*transfer_engine*.so*' \) \
    2>/dev/null | head -1)"
if [ -z "$mooncake_so" ]; then
  echo "ERROR: no mooncake/transfer_engine .so under /usr/local/lib{,64}"
  echo "       Mooncake's make install may have failed at image build."
  exit 1
fi
missing="$(ldd "$mooncake_so" 2>/dev/null | grep 'not found' || true)"
if [ -n "$missing" ]; then
  echo "ERROR: $mooncake_so has unresolved libraries:"
  echo "$missing"
  echo "       Check the CANN env / driver mounts (ASCEND_DIRECT needs them)."
  exit 1
fi
echo "OK: linkage resolves ($mooncake_so)"

# --- 4) vllm KV connector registry -------------------------------------------
# Mirrors vime/backends/vllm_utils/vllm_engine.py: official vllm v0.23.0 ships
# "MooncakeConnector"; "MooncakeConnectorV1" only existed in the former fork.
python3 -c "\
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory; \
assert {'MooncakeConnectorV1', 'MooncakeConnector'} & set(KVConnectorFactory._registry), \
    'no Mooncake connector in vllm KV connector registry — PD will not start'; \
print('OK: Mooncake connector registered in vllm')"

echo "=== ✅ Mooncake PD data plane verified ==="
