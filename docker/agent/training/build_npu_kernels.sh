#!/bin/bash
# =============================================================================
# build_npu_kernels.sh — run ONCE inside the container, on an Ascend A3 (NPU-
# present) host, to compile the two hardware/env-dependent packages that the
# image build deferred: vllm-ascend custom AscendC kernels, and fla_npu GDN ops.
#
#   docker run -it --device=/dev/davinci0 ... vime:a3-release bash
#   bash /workspace/build_npu_kernels.sh
#   # optional: docker commit <container> vime:a3-final   (bake the result in)
#
# Idempotent-ish: safe to re-run (it purges stale build dirs first).
# =============================================================================
set -eo pipefail
set -x

# --- CANN environment (probe both known layouts) ---
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null \
  || source /usr/local/Ascend/cann-9.0.0/set_env.sh
set -u

# --- ensure bisheng (AscendC compiler) is on PATH ---
if ! command -v bisheng >/dev/null 2>&1; then
  b="$(find /usr/local/Ascend -name bisheng -type f 2>/dev/null | head -1)"
  [ -n "$b" ] && export PATH="$(dirname "$b"):$PATH"
fi
command -v bisheng >/dev/null 2>&1 \
  || { echo "ERROR: bisheng not found — check the CANN compiler component is installed."; exit 1; }
bisheng --version 2>/dev/null | head -1 || true

FLA_SOC="${FLA_SOC:-ascend910_93}"
FLA_OPS="${FLA_OPS:-chunk_bwd_dv_local,chunk_bwd_dqkwg,chunk_gated_delta_rule_bwd_dhu,prepare_wy_repr_bwd_da,prepare_wy_repr_bwd_full,chunk_fwd_o,chunk_gated_delta_rule_fwd_h,recurrent_gated_delta_rule,recompute_wu_fwd,causal_conv1d}"
export SOC_VERSION="${SOC_VERSION:-ascend910_9391}"     # A3; npu-smi also auto-detects

VENDOR_LIB="${ASCEND_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest/opp}/vendors/custom_transformer/op_api/lib"

# ---------------------------------------------------------------------------
# 1) vllm-ascend — custom AscendC kernels (COMPILE_CUSTOM_KERNELS=1)
# ---------------------------------------------------------------------------
cd /workspace/vllm-ascend
rm -rf csrc/build
find . -name CMakeCache.txt -delete 2>/dev/null || true
COMPILE_CUSTOM_KERNELS=1 pip install -v --no-cache-dir -e /workspace/vllm-ascend --no-build-isolation

# Fix vllm_ascend_C.so RUNPATH (cmake 4.4.0 emits literal `$$ORIGIN`)
for so in $(find /workspace/vllm-ascend/vllm_ascend -name 'vllm_ascend_C*.so' 2>/dev/null); do
  patchelf --set-rpath '$ORIGIN:$ORIGIN/lib:$ORIGIN/_cann_ops_custom/vendors/custom_transformer/op_api/lib' "$so"
  echo "patched RUNPATH: $so"
done

# ---------------------------------------------------------------------------
# 2) fla_npu — GDN AscendC ops (slime-ascend tutorial steps)
# ---------------------------------------------------------------------------
cd /workspace/flash-linear-attention-npu
bash build.sh --soc="${FLA_SOC}" --pkg --ops="${FLA_OPS}" </dev/null
bash build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-for-all </dev/null
export LD_LIBRARY_PATH="${VENDOR_LIB}:${LD_LIBRARY_PATH:-}"
cd torch_custom/fla_npu
bash build.sh

echo "=========================================================================="
echo "✅ NPU kernels built: vllm-ascend custom ops + fla_npu installed."
echo "   (optional) docker commit this container to bake the result into an image."
echo "=========================================================================="
