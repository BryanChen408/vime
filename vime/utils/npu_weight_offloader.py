"""NPU weight offloader — storage-resize release of Megatron DDP flat buffers on Ascend.

Drop-in replacement for ``torch_memory_saver`` on NPU, where the CUDA
LD_PRELOAD/VMM hooks are unavailable.

Why storage-resize (design history):
  v1 (fake offload, removed): copied each ``param.data`` to CPU and swapped the
  DDP flat-buffer attributes (``param_data``/``grad_data``) to empty tensors.
  Every Megatron bucket view still aliased the original flat storage, so not a
  single byte of HBM was returned (~13GB/rank residue for 9B/TP4) and vLLM
  wake_up OOMed allocating its KV cache (aclrtMallocPhysical at camem_allocator).

  v2 (current, verl-style): ``untyped_storage().resize_(0)`` on the flat buffer
  shrinks the one storage all views share — every view invalidated at once,
  tensor object identity preserved (optimizer/main_grad references stay valid),
  physical pages returned after ``empty_cache()``. ``onload()`` resizes the
  storage back and zeroes ``grad_data`` (contents are disposable; backward
  refills them). ``param_data`` can be restored either from the actor's existing
  ``TensorBackuper`` or, for legacy callers without one, from a private pinned
  CPU backup.

A/B modes (env ``VIME_OFFLOAD_PARAM_BUFFER``, retained for compatibility):
  "0" (default, A): release ``grad_data`` only (fp32, ~9GB/rank @ 9B/TP4);
      ``param_data`` stays on NPU for direct IPC export.
  "1" (B): release ``param_data`` + ``grad_data``. Rollout-stage training
      residue ≈ 0 — required when rollout shares the actor's devices. Actor
      callers reuse their already-present CPU weights, so this mode does not
      allocate a second model-sized host backup.

Measured (Qwen3.5-0.8B, 4×910B3, colocate): v1 left 4.5GB allocated during
rollout with 0 physically freed; A leaves 1.4GB; B leaves 0.0GB with training
metrics identical to A.

Usage::

    offloader = NPUWeightOffloader(release_param_buffer=True, param_restorer=restore_actor)
    offloader.offload(model)   # flat buffers resized to 0, HBM freed
    # ... vLLM rollout (actor NPU memory available) ...
    offloader.onload(model)    # storage restored, actor params restored, grads zeroed
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch

logger = logging.getLogger(__name__)

# (id(buffer), attr) -> (original_storage_size_bytes, pinned_cpu_backup_or_None)
_SavedEntry = tuple[int, torch.Tensor | None]


class NPUWeightOffloader:
    """Release / restore Megatron DDP flat buffers via storage-resize.

    Operates on the ``_ParamAndGradBuffer`` flat tensors (``grad_data`` fp32,
    ``param_data`` bf16) that Megatron's DDP wrapper owns.  Individual
    ``param.data`` / ``param.main_grad`` tensors are views into these buffers,
    so resizing the shared storage releases everything in one step and — once
    the storage is resized back — every view becomes valid again without any
    pointer surgery.
    """

    def __init__(
        self,
        release_param_buffer: bool | None = None,
        param_restorer: Callable[[], None] | None = None,
    ) -> None:
        """Create a Megatron flat-buffer offloader.

        Args:
            release_param_buffer: Whether to release ``param_data`` in addition
                to ``grad_data``. ``None`` preserves the legacy
                ``VIME_OFFLOAD_PARAM_BUFFER`` switch, evaluated at each offload.
                An explicit value takes precedence over the environment.
            param_restorer: Existing owner of a CPU parameter backup. When set,
                ``onload`` calls it after recreating the flat storages instead
                of allocating another model-sized pinned CPU copy. Actor workers
                pass their ``TensorBackuper`` restore operation; callers without
                such a backup retain the legacy private-backup behavior.
        """
        self._release_param_buffer = release_param_buffer
        self._param_restorer = param_restorer
        self._saved: dict[tuple[int, str], _SavedEntry] = {}
        self._offloaded_bytes: int = 0
        self._offloaded_param_buffer: bool = False
        self._offload_time: float = 0.0
        self._onload_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def offload(
        self,
        model: torch.nn.Module | Sequence[torch.nn.Module],
        verbose: bool = True,
    ) -> int:
        """Resize the DDP flat-buffer storages to 0 and return bytes released.

        ``grad_data`` is always released without backup (backward refills it).
        ``param_data`` is additionally released when explicitly requested or
        when ``VIME_OFFLOAD_PARAM_BUFFER=1`` in legacy mode. If
        ``param_restorer`` was supplied, its existing CPU backup is reused;
        otherwise a private pinned backup is retained for compatibility.
        """
        if self._saved:
            logger.warning(
                "offload() called but %d buffers already offloaded — skipping",
                len(self._saved),
            )
            return self._offloaded_bytes

        t0 = time.perf_counter()
        offload_param = self.release_param_buffer
        attrs = ("grad_data", "param_data") if offload_param else ("grad_data",)

        total_bytes = 0
        count = 0
        for buf in _iter_ddp_buffers(model):
            for attr in attrs:
                flat = getattr(buf, attr, None)
                if not (isinstance(flat, torch.Tensor) and flat.numel() > 0):
                    continue
                storage = flat.untyped_storage()
                size = storage.size()
                if size == 0:
                    continue
                if attr == "param_data":
                    # Actor workers already own a persistent pinned CPU copy in
                    # TensorBackuper. Only legacy callers without that owner need
                    # a private flat-buffer backup here.
                    if self._param_restorer is None:
                        backup = torch.empty_like(flat, device="cpu", pin_memory=True)
                        backup.copy_(flat.detach(), non_blocking=False)
                    else:
                        backup = None
                else:
                    # Gradients are disposable.
                    backup = None
                self._saved[(id(buf), attr)] = (size, backup)
                storage.resize_(0)
                total_bytes += size
                count += 1

        torch.npu.empty_cache()

        self._offloaded_bytes = total_bytes
        self._offloaded_param_buffer = offload_param
        self._offload_time = time.perf_counter() - t0
        if verbose:
            mode = "param+grad" if offload_param else "grad-only"
            if offload_param:
                mode += ", existing CPU backup" if self._param_restorer is not None else ", private CPU backup"
            logger.info(
                "NPU flat-buffer offload (%s): %d buffers, %.1f MiB released in %.2fs",
                mode,
                count,
                total_bytes / (1024 * 1024),
                self._offload_time,
            )
        return total_bytes

    def onload(
        self,
        model: torch.nn.Module | Sequence[torch.nn.Module],
        verbose: bool = True,
    ) -> int:
        """Resize storages back, restore parameters, and zero gradients."""
        if not self._saved:
            logger.warning("onload() called but nothing offloaded — skipping")
            return 0

        t0 = time.perf_counter()
        current: dict[tuple[int, str], torch.Tensor] = {}
        for buf in _iter_ddp_buffers(model):
            for attr in ("grad_data", "param_data"):
                flat = getattr(buf, attr, None)
                key = (id(buf), attr)
                if key in self._saved and isinstance(flat, torch.Tensor):
                    current[key] = flat

        missing = self._saved.keys() - current.keys()
        if missing:
            # Silently dropping entries leaves parameter views backed by a
            # zero-sized storage. Fail before mutating any storage so a rebuilt
            # model cannot continue with partially restored weights.
            raise RuntimeError(
                "NPU flat-buffer onload could not match "
                f"{len(missing)} saved buffers; was the model rebuilt between offload and onload?"
            )

        total_bytes = sum(size for size, _ in self._saved.values())
        for key, (size, backup) in self._saved.items():
            flat = current[key]
            flat.untyped_storage().resize_(size)
            if backup is not None:
                flat.copy_(backup)
            else:
                # Gradients are disposable. In external-restore mode this also
                # initializes DDP padding not covered by named parameters.
                flat.zero_()

        # Actor TensorBackuper.restore() refills all revived parameter views.
        # It is intentionally called even in grad-only mode: actor wake_up has
        # always switched back to the latest CPU "actor" tag at this point.
        if self._param_restorer is not None:
            self._param_restorer()

        restored = len(self._saved)
        self._saved.clear()
        self._offloaded_param_buffer = False

        self._onload_time = time.perf_counter() - t0
        if verbose:
            logger.info(
                "NPU flat-buffer onload: %d buffers, %.1f MiB restored in %.2fs",
                restored,
                total_bytes / (1024 * 1024),
                self._onload_time,
            )
        return total_bytes

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_offloaded(self) -> bool:
        return bool(self._saved)

    @property
    def release_param_buffer(self) -> bool:
        """Resolve the explicit policy or the legacy environment switch."""
        if self._release_param_buffer is not None:
            return self._release_param_buffer
        return os.environ.get("VIME_OFFLOAD_PARAM_BUFFER", "0") == "1"

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "offloaded_mb": self._offloaded_bytes / (1024 * 1024),
            "offload_time_s": round(self._offload_time, 2),
            "onload_time_s": round(self._onload_time, 2),
            "num_buffers": len(self._saved) or -1,
            "param_buffer_offloaded": self._offloaded_param_buffer,
            "param_backup_source": "external" if self._param_restorer is not None else "private",
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _iter_ddp_buffers(model: torch.nn.Module | Sequence[torch.nn.Module]):
    """Yield Megatron ``_ParamAndGradBuffer`` objects from *model*.

    *model* may be a single DDP-wrapped module or a list of DDP-wrapped chunks
    (pipeline parallelism).  Buffers live on the DDP wrapper itself under
    ``buffers`` and — for MoE — ``expert_parallel_buffers``.
    """
    chunks = model if isinstance(model, (list, tuple)) else [model]
    seen: set[int] = set()
    for chunk in chunks:
        if chunk is None:
            continue
        for list_attr in ("buffers", "expert_parallel_buffers"):
            bufs = getattr(chunk, list_attr, None)
            if not isinstance(bufs, (list, tuple)):
                continue
            for buf in bufs:
                if buf is not None and id(buf) not in seen:
                    seen.add(id(buf))
                    yield buf
