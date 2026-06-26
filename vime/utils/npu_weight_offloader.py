"""NPU weight offloader — CPU offload for Megatron model parameters on Ascend NPU.

Drop-in replacement for ``torch_memory_saver`` on NPU, where the original
CUDA LD_PRELOAD hooks are unavailable.  Operates at the Python level:
iterates model parameters, copies data to CPU, and **replaces ``param.data``
with the CPU tensor** so that NPU storage is immediately dereferenced and
freed (CPython refcounting).  No tiny placeholder NPU tensors are created.

Usage::

    offloader = NPUWeightOffloader()
    offloader.offload(model)   # param.data → CPU, NPU HBM freed
    # ... vLLM rollout (actor NPU memory available) ...
    offloader.onload(model)    # CPU → param.data on NPU, saved tensors released
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)


class NPUWeightOffloader:
    """Save / restore Megatron model weights to CPU for NPU HBM offloading.

    **offload()** copies every leaf parameter from NPU → CPU, then sets
    ``param.data = cpu_copy``.  The old NPU tensor has zero references so
    CPython deallocates it immediately — no garbage left behind.

    **onload()** sends each saved CPU buffer back to NPU, restores
    ``param.data``, and frees the CPU copy.

    Before offload, ``param.grad`` is set to ``None`` on every parameter to
    release gradient buffers (which would otherwise keep NPU storage alive).
    After offload, ``torch.npu.empty_cache()`` + ``gc.collect()`` are called
    to force the NPU allocator and Python GC to release all freed blocks.
    """

    def __init__(self) -> None:
        self._saved: dict[str, torch.Tensor] = {}  # name → CPU tensor
        self._saved_devices: dict[str, torch.device] = {}  # name → original device
        self._saved_dtypes: dict[str, torch.dtype] = {}  # name → original dtype
        self._offloaded_bytes: int = 0
        self._offload_time: float = 0.0
        self._onload_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def offload(self, model: torch.nn.Module, verbose: bool = True) -> int:
        """Copy all model parameters from NPU to CPU and release NPU storage.

        Returns:
            Total bytes offloaded.
        """
        if self._saved:
            logger.warning(
                "offload() called but %d params already offloaded — skipping",
                len(self._saved),
            )
            return self._offloaded_bytes

        t0 = time.perf_counter()

        # 1. Release gradient buffers AND Megatron DDP flat-buffer storage.
        #    DDP._ParamAndGradBuffer holds grad_data (and optionally param_data)
        #    as large NPU flat tensors that share storage with model params
        #    via views.  Changing individual param.data does NOT free these
        #    flat buffers, so we must explicitly release them first.
        _zero_grads(model)
        _release_ddp_buffers(model)
        _force_npu_cleanup()

        # 2. Offload every NPU leaf parameter.
        #    (No storage dedup needed — untie-embeddings-and-output-weights is on,
        #    so there are no tied weights.  Each param has independent storage.)
        total_bytes = 0
        count = 0
        for name, param in _iter_params(model):
            if not _is_npu(param):
                continue
            if param.numel() == 0:
                continue
            device = param.device
            dtype = param.dtype

            # CPU copy — use plain .cpu() which is reliable on NPU.
            cpu_tensor = param.data.detach().cpu()
            total_bytes += cpu_tensor.numel() * cpu_tensor.element_size()

            # Store metadata and CPU tensor.
            self._saved[name] = cpu_tensor
            self._saved_devices[name] = device
            self._saved_dtypes[name] = dtype

            # ★ Release NPU HBM: replace param.data with the CPU copy.
            # nn.Parameter.data setter calls Tensor.set_() which changes
            # the underlying storage pointer and decrements the old NPU
            # storage's refcount. We explicitly drop any temporary
            # references to help CPython deallocate promptly.
            param.data = cpu_tensor
            count += 1

        self._offloaded_bytes = total_bytes
        self._offload_time = time.perf_counter() - t0

        # 4. Force cleanup: released blocks back to NPU driver.
        #    Call empty_cache() multiple times — the NPU caching
        #    allocator sometimes needs several rounds to release all
        #    cached blocks after mass deallocation.
        _force_npu_cleanup()
        _force_npu_cleanup()  # second pass for stragglers

        if verbose:
            self._log("offload", total_bytes, self._offload_time, count)

        return total_bytes

    def onload(self, model: torch.nn.Module, verbose: bool = True) -> int:
        """Restore model parameters from saved CPU buffers back to NPU.

        Returns:
            Total bytes restored.
        """
        if not self._saved:
            logger.warning("onload() called but no params saved — skipping")
            return 0

        t0 = time.perf_counter()
        total_bytes = 0
        restored = 0

        for name, param in _iter_params(model):
            cpu_tensor = self._saved.pop(name, None)
            if cpu_tensor is None:
                continue
            device = self._saved_devices.pop(name, param.device)
            dtype = self._saved_dtypes.pop(name, param.dtype)

            total_bytes += cpu_tensor.numel() * cpu_tensor.element_size()

            # Move back to NPU and assign.
            param.data = cpu_tensor.to(device=device, dtype=dtype)
            restored += 1

        self._onload_time = time.perf_counter() - t0

        # Clean up any stragglers and free CPU memory.
        self._saved.clear()
        self._saved_devices.clear()
        self._saved_dtypes.clear()
        gc.collect()

        if verbose:
            self._log("onload", total_bytes, self._onload_time, restored)

        return total_bytes

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_offloaded(self) -> bool:
        return len(self._saved) > 0

    @property
    def num_saved(self) -> int:
        return len(self._saved)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "offloaded_mb": self._offloaded_bytes / (1024 * 1024),
            "offload_time_s": round(self._offload_time, 2),
            "onload_time_s": round(self._onload_time, 2),
            "num_params": len(self._saved) or -1,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log(self, direction: str, total_bytes: int, elapsed: float, count: int) -> None:
        mb = total_bytes / (1024 * 1024)
        bw = mb / elapsed if elapsed > 0 else 0.0
        logger.info(
            "NPU weight %s: %d params, %.1f MiB in %.2fs (%.0f MiB/s)",
            direction,
            count,
            mb,
            elapsed,
            bw,
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _is_npu(param: torch.nn.Parameter) -> bool:
    return param.device.type in ("npu", "privateuseone")


def _iter_params(model):
    """Yield ``(name, param)`` from *model*, which may be a ``nn.Module`` or a
    ``list`` of ``DDP``-wrapped module chunks (Megatron pipeline parallelism).
    """
    if isinstance(model, list):
        for chunk in model:
            # Megatron DDP can wrap a list of model chunks; unwrap both.
            inner = chunk
            for _ in range(2):
                if hasattr(inner, "module"):
                    inner = inner.module
            if isinstance(inner, list):
                for m in inner:
                    yield from m.named_parameters()
            else:
                yield from inner.named_parameters()
    else:
        yield from model.named_parameters()


def _zero_grads(model) -> None:
    """Set ``.grad = None`` on every parameter to release gradient buffers."""
    for _, param in _iter_params(model):
        if param.grad is not None:
            param.grad = None


def _release_ddp_buffers(model) -> None:
    """Release Megatron DDP flat gradient/parameter buffers.

    Megatron's ``_ParamAndGradBuffer`` replaces param.data with views
    into a flat ``param_data`` tensor (when ``use_distributed_optimizer``
    is True) and always creates a ``grad_data`` flat tensor for gradient
    all-reduce.  These flat tensors prevent individual param offload from
    freeing NPU storage because the views share the flat buffer's storage.

    1. Copy each param out of the flat buffer into an independent tensor
       (so it owns its storage).
    2. Release the flat buffers.
    """
    ddps = _get_ddp_wrappers(model)

    for ddp in ddps:
        # Find the _ParamAndGradBuffer on the DDP wrapper (not inner module).
        grad_buf = None
        param_buf = None
        for attr in dir(ddp):
            if attr.startswith("__"):
                continue
            try:
                val = getattr(ddp, attr)
            except (AttributeError, RuntimeError):
                continue
            if val is None:
                continue
            if hasattr(val, "grad_data") and isinstance(val.grad_data, torch.Tensor):
                grad_buf = val
            if hasattr(val, "param_data") and isinstance(val.param_data, torch.Tensor):
                param_buf = val

        # Step 1: clone params out of the flat buffer.
        if param_buf is not None and param_buf.param_data is not None:
            flat_storage = param_buf.param_data.untyped_storage()
            for p in param_buf.params:
                if p.data.untyped_storage() is flat_storage:
                    p.data = p.data.clone()  # detach from flat buffer
            del flat_storage
            logger.debug("Cloned %d params out of flat param_data buffer", len(param_buf.params))

        if grad_buf is not None:
            flat_storage = grad_buf.grad_data.untyped_storage()
            for p in grad_buf.params:
                if hasattr(p, "main_grad") and p.main_grad is not None:
                    if p.main_grad.untyped_storage() is flat_storage:
                        p.main_grad = None  # will be recreated on next backward
            del flat_storage

        # Step 2: release flat buffers.
        if param_buf is not None and param_buf.param_data is not None:
            sz = param_buf.param_data.numel() * param_buf.param_data.element_size()
            param_buf.param_data = torch.empty(0, device="cpu")
            logger.debug("Released DDP param_data (%.0f MiB)", sz / (1024 * 1024))
        if grad_buf is not None:
            sz = grad_buf.grad_data.numel() * grad_buf.grad_data.element_size()
            grad_buf.grad_data = torch.empty(0, device="cpu")
            logger.debug("Released DDP grad_data (%.0f MiB)", sz / (1024 * 1024))


def _get_ddp_wrappers(model):
    """Return list of Megatron DDP wrapper objects from *model*.

    The DDP wrapper holds ``_ParamAndGradBuffer`` which we need to release.
    The inner module (``ddp.module``) does NOT hold these buffers.
    """
    if isinstance(model, list):
        return [chunk for chunk in model if hasattr(chunk, "module")]
    if hasattr(model, "module"):
        return [model]
    return []


def _force_npu_cleanup() -> None:
    """Force NPU allocator and Python GC to release all freed blocks."""
    try:
        torch.npu.empty_cache()
    except Exception:
        pass
    # Some torch_npu versions have a more aggressive sync.
    try:
        torch.npu.synchronize()
    except Exception:
        pass
    gc.collect()
