"""Release Megatron's DDP flat buffers by resizing their storage to zero.

An alternative to torch_memory_saver for the rollout phase. TMS takes over the torch
allocator, which means expandable_segments has to be off; this works at the Python level
and leaves the allocator configuration alone.

Megatron's DDP wrapper owns two flat buffers per ``_ParamAndGradBuffer`` — ``grad_data``
(fp32) and ``param_data`` (bf16) — and every ``param.data`` / ``param.main_grad`` is a view
into one of them. Swapping the buffer attributes for empty tensors therefore frees nothing:
the bucket views still alias the original storage. Resizing the shared storage instead
invalidates every view at once, keeps the tensor objects (so optimizer and main_grad
references stay valid), and returns the physical pages. ``onload`` resizes back and refills.

The buffers are matched between offload and onload by object identity, so the model must
not be rebuilt in between.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from vime.utils.memory_utils import clear_memory

logger = logging.getLogger(__name__)

# The flat buffers on a Megatron _ParamAndGradBuffer, in release order.
_FLAT_BUFFER_ATTRS = ("grad_data", "param_data")

# (original storage size in bytes, pinned CPU backup or None)
_SavedEntry = tuple[int, torch.Tensor | None]


class NPUWeightOffloader:
    """Release and restore Megatron DDP flat buffers via storage resize.

    Args:
        release_param_buffer: also release ``param_data``, backing it up to pinned CPU
            first. Gradients alone are enough to make room for a separate rollout engine,
            but a colocated engine needs the whole card. Weights are only read back by
            copy (the engine never aliases the actor's storage), so releasing them is safe.
    """

    def __init__(self, release_param_buffer: bool = False) -> None:
        self._release_param_buffer = release_param_buffer
        self._saved: dict[tuple[int, str], _SavedEntry] = {}
        self._offloaded_bytes: int = 0
        self._offload_count: int = 0
        self._offload_time: float = 0.0
        self._onload_time: float = 0.0

    @property
    def is_offloaded(self) -> bool:
        return bool(self._saved)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "offloaded_mb": round(self._offloaded_bytes / (1024 * 1024), 1),
            "buffers_currently_offloaded": len(self._saved),
            "buffers_offloaded_last_call": self._offload_count,
            "offload_time_s": round(self._offload_time, 2),
            "onload_time_s": round(self._onload_time, 2),
        }

    def offload(self, model: torch.nn.Module, verbose: bool = True) -> int:
        """Resize the flat-buffer storages to zero. Returns the bytes released."""
        if self._saved:
            logger.warning("offload() called with %d buffers already offloaded", len(self._saved))
            return self._offloaded_bytes

        started = time.perf_counter()
        attrs = _FLAT_BUFFER_ATTRS if self._release_param_buffer else ("grad_data",)

        total_bytes = 0
        count = 0
        for buffer in _iter_ddp_buffers(model):
            for attr in attrs:
                flat = getattr(buffer, attr, None)
                if not (isinstance(flat, torch.Tensor) and flat.numel() > 0):
                    continue
                storage = flat.untyped_storage()
                size = storage.size()
                if size == 0:
                    continue
                # Gradients are refilled by the next backward, weights are not.
                backup = flat.detach().to("cpu", copy=True).pin_memory() if attr == "param_data" else None
                self._saved[(id(buffer), attr)] = (size, backup)
                storage.resize_(0)
                total_bytes += size
                count += 1

        clear_memory()

        self._offloaded_bytes = total_bytes
        self._offload_count = count
        self._offload_time = time.perf_counter() - started
        if verbose:
            logger.info(
                "offloaded %d flat buffers (%s), %.1f MiB in %.2fs",
                count,
                "param+grad" if self._release_param_buffer else "grad only",
                total_bytes / (1024 * 1024),
                self._offload_time,
            )
        return total_bytes

    def onload(self, model: torch.nn.Module, verbose: bool = True) -> int:
        """Resize the storages back and refill them. Returns the bytes restored."""
        if not self._saved:
            logger.warning("onload() called with nothing offloaded")
            return 0

        started = time.perf_counter()
        total_bytes = 0
        restored = 0
        for buffer in _iter_ddp_buffers(model):
            for attr in _FLAT_BUFFER_ATTRS:
                flat = getattr(buffer, attr, None)
                if flat is None:
                    continue
                entry = self._saved.pop((id(buffer), attr), None)
                if entry is None:
                    continue
                size, backup = entry
                flat.untyped_storage().resize_(size)
                if backup is not None:
                    flat.copy_(backup)
                else:
                    # The new pages hold whatever was there before; zero them so a
                    # parameter that gets no gradient this step does not feed the
                    # optimizer garbage.
                    flat.zero_()
                total_bytes += size
                restored += 1

        if self._saved:
            logger.warning(
                "onload() dropped %d saved buffers with no matching DDP buffer; "
                "was the model rebuilt between offload and onload?",
                len(self._saved),
            )
            self._saved.clear()

        self._onload_time = time.perf_counter() - started
        if verbose:
            logger.info(
                "onloaded %d flat buffers, %.1f MiB in %.2fs",
                restored,
                total_bytes / (1024 * 1024),
                self._onload_time,
            )
        return total_bytes


def _iter_ddp_buffers(model):
    """Yield the ``_ParamAndGradBuffer`` objects of a DDP-wrapped model.

    `model` is either one DDP-wrapped module or a list of them, one per pipeline chunk.
    """
    chunks = model if isinstance(model, (list, tuple)) else [model]
    seen: set[int] = set()
    for chunk in chunks:
        if chunk is None:
            continue
        for attr in ("buffers", "expert_parallel_buffers"):
            buffers = getattr(chunk, attr, None)
            if not isinstance(buffers, (list, tuple)):
                continue
            for buffer in buffers:
                if buffer is not None and id(buffer) not in seen:
                    seen.add(id(buffer))
                    yield buffer
