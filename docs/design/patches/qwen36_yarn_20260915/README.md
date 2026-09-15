# Qwen3.6 YaRN external patch bundle

This directory makes the VIME `dev/yarn` branch a self-contained source of the
external Megatron-LM and vLLM changes used by the validated Qwen3.6 YaRN setup.
The patches do not modify Polar, MindSpeed, or vllm-ascend.

The patches are net diffs produced with `git diff --binary`. Apply them in the
listed order to clean worktrees at the exact base revisions.

## Megatron-LM

Repository used for validation: `/workspace/Megatron-LM`.

| Order | Patch | Input revision | Resulting revision/state | SHA-256 |
| ---: | --- | --- | --- | --- |
| 1 | `megatron-0001-vime-npu-prerequisite.patch` | `228e44c38` | content of `546dd1d12` | `721942340f2a5120719f9b345cbf814b8de4f47893890ea7b9d16db3607dc06d` |
| 2 | `megatron-0002-qwen36-yarn.patch` | content of `546dd1d12` | content of `78c0ead8c` | `aee0c07fc41249389c3351a6449772719c69822dc178f3cc47745e5174ecff8f` |

The prerequisite patch carries the pre-existing NPU/VIME compatibility state
needed by the tested runtime. The YaRN patch contains the non-MLA CLI/config
wiring, the Qwen3.6 `rotary_percent` fix, and their unit tests.

```bash
git -C /path/to/Megatron-LM switch --detach 228e44c38
git -C /path/to/Megatron-LM apply --check \
  /path/to/vime/docs/design/patches/qwen36_yarn_20260915/megatron-0001-vime-npu-prerequisite.patch
git -C /path/to/Megatron-LM apply \
  /path/to/vime/docs/design/patches/qwen36_yarn_20260915/megatron-0001-vime-npu-prerequisite.patch
git -C /path/to/Megatron-LM apply --check \
  /path/to/vime/docs/design/patches/qwen36_yarn_20260915/megatron-0002-qwen36-yarn.patch
git -C /path/to/Megatron-LM apply \
  /path/to/vime/docs/design/patches/qwen36_yarn_20260915/megatron-0002-qwen36-yarn.patch
```

## vLLM 0.23

Repository used for validation: `/workspace/vllm-023`.

| Order | Patch | Input revision | Resulting revision/state | SHA-256 |
| ---: | --- | --- | --- | --- |
| 1 | `vllm-0001-vime-runtime-prerequisites.patch` | `v0.23.0` / `0fc695fc6` | content of `99ce1f27b` | `d9616b03b71500a3913f6811ad5e7ebbc531c607c4b5e4fffed9127574428c81` |
| 2 | `vllm-0002-qwen36-mrope-yarn.patch` | content of `99ce1f27b` | validated MRoPE/YaRN working state | `2f57a90d85e18cdc89c47f830748f50cff8beb6fff7f8f462c7eea1d29375a04` |

The prerequisite patch carries the pre-existing Qwen3.5/VIME runtime fixes
used by the tested vLLM deployment. The YaRN patch separates the original
context length used for frequency correction from the enlarged MRoPE cache
capacity and adds focused regression tests.

```bash
git -C /path/to/vllm switch --detach 0fc695fc6
git -C /path/to/vllm apply --check \
  /path/to/vime/docs/design/patches/qwen36_yarn_20260915/vllm-0001-vime-runtime-prerequisites.patch
git -C /path/to/vllm apply \
  /path/to/vime/docs/design/patches/qwen36_yarn_20260915/vllm-0001-vime-runtime-prerequisites.patch
git -C /path/to/vllm apply --check \
  /path/to/vime/docs/design/patches/qwen36_yarn_20260915/vllm-0002-qwen36-mrope-yarn.patch
git -C /path/to/vllm apply \
  /path/to/vime/docs/design/patches/qwen36_yarn_20260915/vllm-0002-qwen36-mrope-yarn.patch
```

The staged local change to
`tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml` is unrelated and is
intentionally absent from this bundle.

## Verification

After applying both series, verify their resulting diffs and run the focused
tests in the configured environments:

```bash
git -C /path/to/Megatron-LM diff --check
git -C /path/to/vllm diff --check

cd /path/to/Megatron-LM
pytest -q \
  tests/unit_tests/test_yarn_arguments.py \
  tests/unit_tests/models/test_yarn_rotary_pos_embedding.py

cd /path/to/vllm
.venv/bin/python -m pytest -q tests/kernels/core/test_mrope_yarn.py
```

The VIME preflight and end-to-end launch commands are documented in
[`../../../README_qwen36_yarn.md`](../../../README_qwen36_yarn.md).
