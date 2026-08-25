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

---

## 5. PD variant — `Dockerfile.release-pd` (vLLM v0.23.0 + Mooncake)

`Dockerfile.release-pd` is a variant of `Dockerfile.release` for the
**prefill/decode-disaggregated** rollout. It keeps
the release image's dependency model (official upstream + pinned ref + small
patch) and its entire training stack, and changes only what PD requires:

| area | Dockerfile.release | Dockerfile.release-pd |
|------|--------------------|-----------------------|
| vllm | v0.21.0 + `vllm-vime.patch` | **v0.23.0** + `vllm-pd.patch` (4 files: polar telemetry; qwen3.5/3.6 FusedMoE `weight_loader` re-attach for EP RL weight-sync) |
| vllm-ascend | v0.21.0rc1 + `vllm-ascend-vime.patch` | **`88eac271`** + `vllm-ascend-pd.patch` (3 files: camem assert→warning; disable `patch_dp_device_ids` for vLLM v0.23.0 compat) |
| PD data plane | — | **Mooncake v0.3.9** (official, no patch), built `-DUSE_ASCEND_DIRECT=ON`; pulls in Go 1.23.8 / MPICH 4.2.3 / yalantinglibs 0.5.7 + RDMA/gRPC apt deps |
| vime ref | `feature/swe-tasks` | **`a3-pd`** (carries the PD runtime: mooncake proxy, session-affinity router) |
| `VLLM_VERSION` | `0.21.0` | `0.23.0` |
| build-time smoke | — | asserts a Mooncake connector is in vllm's KV connector registry (what `vime/backends/vllm_utils/vllm_engine.py` probes at runtime — official v0.23.0 ships `MooncakeConnector`; `MooncakeConnectorV1` only existed in the former fork, either name passes) + `import mooncake` |

Everything else — base image, proxy machinery, torch 2.10.0 stack,
triton-ascend 3.2.1 wheel gate, MindSpeed/Megatron/fla_npu sections and their
patches, the deferred NPU-kernel build (section 2) — is identical to
`Dockerfile.release`.

### Baseline notes (PD-specific)

- **vllm-ascend's baseline is a BRANCH commit, not the v0.23.0 tag.** The two
  PD patches were extracted from the former private
  forks (`ljyrj/vllm@releases/v0.23.0`, `ljyrj/vllm-ascend-023@vime-adapter-v023`).
  The vllm fork sits directly on tag `v0.23.0`; the vllm-ascend fork is based
  on commit `88eac271` of the official `releases/v0.23.0` **branch** (which is
  ahead of the tag) and does not contain the tag commit. Do not repoint
  `VLLM_ASCEND_REF` at the tag — the patch will not apply.
- **Version cross-check against the official 0.23 baselines** (see the
  Dockerfile header for details): vllm-ascend@88eac271 declares
  `torch==2.10.0` / `triton-ascend==3.2.1` — an exact match for the pins in
  section 4's stack. It also declares `torch-npu==2.10.0.post2` and
  `transformers==5.5.4`; we deliberately keep STOCK torch-npu 2.10.0 (fla_npu
  `torchnpugen`) and transformers 5.12.1 (training-stack-validated; vllm
  v0.23.0 accepts both). If PD smoke hits transformers/torch-npu issues,
  these are the first two knobs to revisit.
- **Go tarball mirror** defaults to Aliyun (`--build-arg GO_MIRROR=...` to
  override); the Tsinghua mirror proved unreliable in the field.

### Build

Same prerequisites as section 0 (CANN 9.0.0 A3 base + triton-ascend 3.2.1
wheel), then:

```
docker build -f Dockerfile.release-pd -t vime:a3-pd-release \
  --build-arg HTTP_PROXY=$http_proxy --build-arg HTTPS_PROXY=$http_proxy \
  --build-arg TRITON_ASCEND_WHEEL=triton_ascend-3.2.1-<...>-aarch64.whl \
  .
```

The build fails fast if Mooncake does not compile or if vllm's KV connector
registry lacks a Mooncake connector (build-time smoke step, run from `/tmp`
with the CANN env sourced — running it from `/workspace` would shadow the
editable vllm install with the source repo root).

**Vendored tarballs (local-first / wget-fallback).** Large tarballs that are
slow or unreliable through the gateway can be dropped into `vendor/` in the
build context (`vendor/` is gitignored, like `wheels/`). A vendored file is
used as-is; otherwise the Dockerfile downloads from the pinned URL:

| file | fallback URL (build-arg) |
|------|--------------------------|
| `vendor/mpich-4.2.3.tar.gz` | GitHub releases (`MPICH_URL`) |


### First run

Identical to section 2 — `build_npu_kernels.sh` still compiles the vllm-ascend
AscendC kernels + fla_npu ops inside the container on an NPU host. Verifying
the Mooncake PD data plane is a **separate** step, via the standalone script
(also baked into the image at `/workspace/`):

```
bash /workspace/verify_mooncake_pd.sh
```

It checks, in order: the `/workspace/Mooncake` tree (skips silently on non-PD
images), the `mooncake` python bindings, transfer-engine `.so` linkage via
`ldd` (the ASCEND_DIRECT path resolves CANN libs — needs the CANN env, so run
on an NPU host / with driver mounts), and that a Mooncake connector
(`MooncakeConnector` on official v0.23.0) is in vllm's KV connector registry.
Re-run it after `docker commit`, driver-mount
or CANN env changes. PD serving itself (prefill/decode topology, proxy,
router) is driven by the vime `a3-pd` runtime configs, not by this image.
