import os

import torch


def _parse_version(version):
    version = version.split("+", 1)[0]
    parts = version.split(".")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0
    return major, minor


def _validate_flashqla_runtime():
    if _parse_version(torch.__version__) < (2, 8):
        raise RuntimeError(f"FlashQLA backend requires PyTorch 2.8 or newer, got PyTorch {torch.__version__}.")

    if not torch.cuda.is_available():
        raise RuntimeError("FlashQLA backend requires CUDA and an NVIDIA SM90 GPU.")

    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (9, 0):
        raise RuntimeError(f"FlashQLA backend requires NVIDIA SM90 or newer, got sm{major}{minor}.")

    cuda_version = torch.version.cuda
    if cuda_version is not None and _parse_version(cuda_version) < (12, 8):
        raise RuntimeError(f"FlashQLA backend requires CUDA 12.8 or newer, got CUDA {cuda_version}.")


def get_chunk_gated_delta_rule(backend: str):
    if backend == "fla":
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        except ImportError as exc:
            raise ImportError("Qwen GDN backend 'fla' requires flash-linear-attention.") from exc
        return chunk_gated_delta_rule

    if backend == "flashqla":
        try:
            from flash_qla import chunk_gated_delta_rule
        except ImportError as exc:
            raise ImportError(
                "Qwen GDN backend 'flashqla' requires FlashQLA. " "Install it from https://github.com/QwenLM/FlashQLA."
            ) from exc
        _validate_flashqla_runtime()
        return chunk_gated_delta_rule

    if backend == "npu":
        # Ascend NPU path: AscendC-hybrid GDN op shipped with MindSpeed
        # (mindspeed.ops.chunk_gated_delta_rule). This is the proven slime-ascend
        # training kernel; it has the same call signature as the fla op.
        try:
            from mindspeed.ops.chunk_gated_delta_rule import chunk_gated_delta_rule
        except ImportError as exc:
            raise ImportError(
                "Qwen GDN backend 'npu' requires mindspeed.ops.chunk_gated_delta_rule "
                "(MindSpeed with GDN support + fla_npu AscendC kernels)."
            ) from exc
        return chunk_gated_delta_rule

    raise ValueError(f"Unsupported Qwen GDN backend: {backend}")


def get_causal_conv1d(backend: str):
    """Return the depthwise causal-conv1d fn for the given GDN backend, or None.

    Only the Ascend NPU path replaces fla's ShortConvolution forward with an
    external kernel (mindspeed.ops.causal_conv1d, the Triton conv ported from the
    MindSpeed-MM Qwen3.6 SFT path). GPU backends keep using ShortConvolution and
    return None here. Set QWEN36_CAUSAL_CONV1D_IMPL=eager to force the eager
    F.silu(F.conv1d) fallback instead of the Triton kernel.
    """
    if backend != "npu":
        return None
    if os.environ.get("QWEN36_CAUSAL_CONV1D_IMPL", "triton") != "triton":
        return None
    try:
        from mindspeed.ops.causal_conv1d import causal_conv1d
    except ImportError as exc:
        raise ImportError(
            "Qwen GDN backend 'npu' requires mindspeed.ops.causal_conv1d "
            "(MindSpeed with GDN support). Set QWEN36_CAUSAL_CONV1D_IMPL=eager to "
            "fall back to the eager conv1d implementation."
        ) from exc
    return causal_conv1d
