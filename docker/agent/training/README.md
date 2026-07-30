# vime A3 release image — one-click build

Target: Ascend A3 (910_93-class) / aarch64, CANN 9.0.0.

Every dependency is cloned from its **official upstream** at a pinned ref and a
small vime patch (`patches/`) is applied — no private forks. See the header of
`Dockerfile.release` for the full baseline table.

---

## 0. Prerequisites you must supply

### (a) Base image — CANN 9.0.0 A3
Default base is `quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.12` (public; the
`-a3-` tag already ships the ascend910_93 operator kernels, so no separate
A3-ops package is needed). If you cannot pull it, download CANN 9.0.0 from:

    https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0

and build/point `--build-arg BASE_IMAGE=...` at an equivalent CANN 9.0.0 aarch64
base. NOTE: triton-ascend 3.2.1 (below) requires CANN 9.0.0 **B160 or later**.

### (b) triton-ascend wheel (not on public PyPI)
The stack was validated on **triton-ascend 3.2.1**, which is not published on
public PyPI. Download the aarch64 wheel from the official releases page:

    https://gitcode.com/Ascend/triton-ascend/releases

Pick the wheel matching CANN 9.0.0 + py3.12 + aarch64, put it under `wheels/`,
and pass its filename at build time:

    --build-arg TRITON_ASCEND_WHEEL=triton_ascend-3.2.1-<...>-aarch64.whl

A cp311 wheel (`triton_ascend-3.2.1-cp311-...-aarch64.whl`) is already shipped in
`wheels/` and is the default for the cann:900 test build. **cp311 requires a
py3.11 base** (matches cann:900); the py3.12 quay release base needs a **cp312**
wheel — download it from the releases page above and pass its filename.

There is **NO fallback**: 3.2.1 is not on public PyPI, so if neither a matching
wheel (in `wheels/` + `--build-arg TRITON_ASCEND_WHEEL=<file>`) nor a
3.2.1-serving `--build-arg PIP_INDEX_URL=<mirror>` is given, the build errors out
by design.

---

## 1. Build

```
docker build -f Dockerfile.release -t vime:a3-release .
```

With proxy + internal pip mirror + the triton-ascend 3.2.1 wheel in `wheels/`:

```
docker build -f Dockerfile.release -t vime:a3-release \
  --build-arg HTTP_PROXY=$HTTP_PROXY   --build-arg HTTPS_PROXY=$HTTPS_PROXY \
  --build-arg http_proxy=$http_proxy   --build-arg https_proxy=$https_proxy \
  --build-arg PIP_INDEX_URL=<mirror>   --build-arg PIP_TRUSTED_HOST=<host> \
  --build-arg TRITON_ASCEND_WHEEL=triton_ascend-3.2.1-<...>-aarch64.whl \
  .
```

---

## 2. First run on an NPU host — build the NPU kernels (REQUIRED)

The image build does NOT compile the two hardware/env-dependent packages
(vllm-ascend custom AscendC kernels + fla_npu GDN ops) — their source is cloned
+ patched into the image, but compiling them needs `npu-smi`, the `bisheng`
compiler and a real NPU env that a `docker build` doesn't have. Compile them
ONCE, inside the container, on an Ascend A3 host:

```
docker run -it --name vime \
  --device=/dev/davinci0 --device=/dev/davinci_manager \
  --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  vime:a3-release bash

# inside the container:
bash /workspace/build_npu_kernels.sh

# optional — bake the compiled kernels into a final image:
docker commit vime vime:a3-final
```

(Adjust the `--device` / driver mounts to your host.) After this, vllm-ascend
and fla_npu are installed and importable.

## 3. Notes

- **vllm-ascend custom kernels + fla_npu** are compiled by
  `build_npu_kernels.sh` on first run (section 2), NOT during image build.
  fla_npu follows the Ascend slime-ascend qwen3.5-9B tutorial exactly
  (`build.sh --soc=ascend910_93 --pkg --ops=<gdn set>` → install the `.run` →
  `torch_custom/fla_npu/build.sh`); its `install_deps.sh` is skipped (the apt
  packages are installed in the Dockerfile). torch-npu is kept at STOCK 2.10.0
  so the bundled `torchnpugen` matches what fla v26.1.0 expects.
- **vime** is cloned from the public repo `github.com/BryanChen408/vime`
  @ `feature/swe-tasks` — nothing to push. (Override with
  `--build-arg VIME_REPO/VIME_REF`.)
- **Polar / ProRL-Agent-Server** (agentic SWE/math RL server) is NOT included in
  this image; it is a separate concern.

### Local test build (Dockerfile.release.cann900)
`Dockerfile.release.cann900` is identical but its base defaults to the local
image `cann:900` (has CANN 9.0.0 + torch) — for testing the build without
pulling the quay base:

    docker build -f Dockerfile.release.cann900 -t vime:a3-test .

## 4. Baselines (all official upstream + patch)

| repo | upstream | ref | patch |
|------|----------|-----|-------|
| vllm | github.com/vllm-project/vllm | v0.21.0 | patches/vllm-vime.patch |
| vllm-ascend | github.com/vllm-project/vllm-ascend | v0.21.0rc1 | patches/vllm-ascend-vime.patch |
| Megatron-LM | github.com/NVIDIA/Megatron-LM | 3714d81d | patches/megatron-vime.patch |
| MindSpeed | gitcode.com/Ascend/MindSpeed | fc63de5c | patches/mindspeed-vime.patch |
| flash-linear-attention-npu | github.com/flashserve/flash-linear-attention-npu | v26.1.0 | (none) |
| vime | github.com/BryanChen408/vime | feature/swe-tasks | (none) |
