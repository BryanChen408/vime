#!/bin/bash
# Build the raw-ADXL remap staleness probe (no Mooncake in the loop).
set -euo pipefail
ASCEND=/usr/local/Ascend/cann-9.2.0
OUT_DIR="$(cd "$(dirname "$0")" && pwd)"
g++ -std=c++17 -O1 -o "$OUT_DIR/adxl_raw_remap_probe" \
  "$OUT_DIR/adxl_raw_remap_probe.cc" \
  -I"$ASCEND/x86_64-linux/include" \
  -I"$ASCEND/include" \
  -L"$ASCEND/x86_64-linux/lib64" \
  -L"$ASCEND/lib64" \
  -lcann_hixl -lllm_datadist -lascendcl -lgraph \
  -Wl,-rpath,"$ASCEND/x86_64-linux/lib64" \
  -Wl,-rpath,"$ASCEND/lib64"
echo "built: $OUT_DIR/adxl_raw_remap_probe"
