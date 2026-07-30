# sandbox — ascendc-tilelang

The per-task execution image the Polar agent runs AscendC / TileLang
kernel-generation tasks in. Target: **A3 / aarch64, CANN 9.0.0, py3.11**.

Contents:
- **TileLang** (compiled from source into a wheel in a builder stage)
- torch + torch_npu, `cmake<4` (AscendC CMakeLists needs cmake < 4)
- Claude Code CLI (Node 20), jemalloc

## Build

```
# TILELANG_SRC defaults to /home/docker/tilelang-ascend; override if needed.
TILELANG_SRC=/path/to/tilelang-ascend \
TAG=ascendc-tilelang:v1 \
  bash build_ascendc_tilelang.sh
```

The script: (1) `git submodule update` on the tilelang source, (2) **`cp`** the
source into a temp build context (no rsync; plain `cp -r`, so no root ownership;
`.git` excluded to shrink), (3) `docker build` (Stage 1 compiles the TileLang
wheel, Stage 2 installs it into the runtime image), (4) verifies
cmake / ccec / bishengir-compile / torch / torch_npu / tilelang.

## After build — point Polar at it

```
sed -i 's|image: ascendc-sandbox:v1|image: ascendc-tilelang:v1|g' \
  ProRL-Agent-Server/deploy/ascend_operator/profile.ascendc.yaml \
  ProRL-Agent-Server/deploy/ascend_operator/profile.t2a.yaml
```
then restart Polar (see `polar/README.md`).

## A3 / aarch64 notes

This was ported from an A2 / x86 setup. The aarch64/A3 specifics:
- base image `quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.11` (A3 = aarch64);
- `LD_PRELOAD` jemalloc path is `aarch64-linux-gnu` (NOT `x86_64-linux-gnu` —
  that path does not exist on aarch64 and makes every command fail with an
  `ld.so ... cannot be preloaded` error).

TODO / verify: torch is pinned at `2.8.0 / torch_npu 2.8.0.post4` (A2-era). Confirm
this runs on A3, or bump to an A3-capable torch_npu (the training image uses
2.10.0). Change via the `TORCH_VERSION` / `TORCH_NPU_VERSION` ARGs.
