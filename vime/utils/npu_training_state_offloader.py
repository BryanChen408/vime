"""Phase offload for live Megatron training state on colocated NPU actors.

The model DDP flat buffers are handled separately by
``NPUWeightOffloader``.  This helper closes the remaining gap between that
targeted storage release and the full allocator pause used by slime:

* preserve optimizer tensors that still have their only copy on the NPU by
  moving their *data* to CPU without replacing the Tensor/Parameter object;
* leave optimizer state that is already on CPU untouched (the normal
  ``optimizer_offload_fraction=1.0`` path);
* discard Megatron/TransformerEngine scratch caches and completed-forward MoE
  dispatcher state, which carry no training state and are recreated lazily on
  the next forward.

The optimizer traversal follows verl's ``offload_megatron_optimizer`` /
``load_megatron_optimizer`` implementation, including ChainedOptimizer and
HybridDeviceOptimizer support.  VIME additionally records each tensor's
original device so restore never guesses whether a state belongs on CPU or the
accelerator.  Megatron's generic ``restore_from_cpu`` cannot be used here: it
moves every CPU state tensor back to the accelerator, undoing HDO's configured
``optimizer_offload_fraction=1.0`` placement.
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)
_NPU_DEVICE_TYPES = frozenset({"npu"})

# Megatron's token dispatcher is a plain Python object rather than an nn.Module.
# These tensors are needed through ``combine_postprocess`` in the same forward,
# but upstream leaves the last values attached afterwards. In a continuous
# trainer the next forward overwrites them; in a colocated trainer they pin the
# completed autograd graph throughout rollout. Restrict cleanup to the exact
# dispatcher and fields observed in the current all-to-all training path.
_MOE_DISPATCHER_FORWARD_STATE = {
    "MoEAlltoAllTokenDispatcher": (
        "probs",
        "routing_map",
        "reversed_local_input_permutation_mapping",
    ),
}


@dataclass
class _SavedTensor:
    tensor: torch.Tensor
    original_device: torch.device
    name: str
    nbytes: int


class NPUTrainingStateOffloader:
    """Offload non-model Megatron state across a colocated rollout phase."""

    def __init__(
        self,
        *,
        global_memory_buffer_getter: Callable[[], Any] | None = None,
        accelerator_device_types: frozenset[str] = _NPU_DEVICE_TYPES,
        tensor_mover: Callable[[torch.Tensor, str | torch.device], None] | None = None,
    ) -> None:
        # The injectable hooks keep the lifecycle CPU-testable. Production uses
        # Megatron's singleton and an in-place .data device move.
        self._global_memory_buffer_getter = global_memory_buffer_getter
        self._accelerator_device_types = accelerator_device_types
        self._tensor_mover = tensor_mover or _move_tensor_data
        self._saved: list[_SavedTensor] = []
        self._is_offloaded = False
        self._optimizer_id: int | None = None
        self._offload_time = 0.0
        self._onload_time = 0.0
        self._optimizer_bytes = 0
        self._scratch_bytes = 0
        self._scratch_tensors = 0
        self._model_runtime_bytes = 0
        self._model_runtime_tensors = 0
        self._allocated_reclaimed_bytes = 0

    @torch.no_grad()
    def offload(self, optimizer: Any, verbose: bool = True, *, model: Any = None) -> int:
        """Move required device optimizer state to CPU and drop scratch caches.

        CPU-resident HDO state is deliberately skipped, so full optimizer CPU
        offload does not create a second host copy. On failure, already-moved
        tensors are restored before the exception propagates.
        """
        if self._is_offloaded:
            logger.warning("NPU training state is already offloaded; skipping duplicate offload")
            return self._optimizer_bytes + self._scratch_bytes

        t0 = time.perf_counter()
        allocated_before = _npu_memory_allocated()
        seen: set[int] = set()
        moved: list[_SavedTensor] = []
        try:
            for tensor, name in _iter_optimizer_device_tensors(optimizer):
                if id(tensor) in seen or tensor.device.type not in self._accelerator_device_types:
                    continue
                seen.add(id(tensor))
                saved = _SavedTensor(
                    tensor=tensor,
                    original_device=tensor.device,
                    name=name,
                    nbytes=_tensor_nbytes(tensor),
                )
                self._tensor_mover(tensor, "cpu")
                moved.append(saved)

            self._saved = moved
            self._optimizer_bytes = sum(item.nbytes for item in moved)
            (
                self._scratch_bytes,
                self._scratch_tensors,
                self._model_runtime_bytes,
                self._model_runtime_tensors,
            ) = self._drop_reconstructible_caches(model)
            _synchronize_and_empty_npu_cache()
            self._allocated_reclaimed_bytes = max(0, allocated_before - _npu_memory_allocated())
            self._is_offloaded = True
            self._optimizer_id = id(optimizer)
        except Exception:
            # A half-offloaded optimizer must never be allowed into either
            # training or checkpointing. Restore object data in reverse order.
            for item in reversed(moved):
                try:
                    self._tensor_mover(item.tensor, item.original_device)
                except Exception:
                    logger.exception("Failed to roll back optimizer tensor %s", item.name)
            self._saved.clear()
            self._optimizer_bytes = 0
            self._optimizer_id = None
            self._allocated_reclaimed_bytes = 0
            _synchronize_and_empty_npu_cache()
            raise

        self._offload_time = time.perf_counter() - t0
        if verbose:
            logger.info(
                "NPU training-state offload: optimizer=%d tensors %.1f MiB -> CPU; "
                "scratch=%d tensors %.1f MiB discarded "
                "(model-runtime=%d tensors %.1f MiB); "
                "allocated-reclaimed=%.1f MiB in %.2fs",
                len(self._saved),
                self._optimizer_bytes / (1024 * 1024),
                self._scratch_tensors,
                self._scratch_bytes / (1024 * 1024),
                self._model_runtime_tensors,
                self._model_runtime_bytes / (1024 * 1024),
                self._allocated_reclaimed_bytes / (1024 * 1024),
                self._offload_time,
            )
        return self._optimizer_bytes + self._scratch_bytes

    @torch.no_grad()
    def onload(self, optimizer: Any, verbose: bool = True) -> int:
        """Restore only optimizer tensors that originally lived on the NPU.

        Scratch buffers are intentionally not restored; Megatron recreates them
        from shape/dtype/name on first use. The owning optimizer identity must
        match so stale Tensor objects cannot be restored into a rebuilt model.
        """
        if not self._is_offloaded:
            logger.warning("NPU training state is not offloaded; skipping duplicate onload")
            return 0
        if id(optimizer) != self._optimizer_id:
            raise RuntimeError("Megatron optimizer was replaced between training-state offload and onload")

        t0 = time.perf_counter()
        for item in self._saved:
            self._tensor_mover(item.tensor, item.original_device)
        _synchronize_and_empty_npu_cache()

        restored_bytes = self._optimizer_bytes
        restored_tensors = len(self._saved)
        self._saved.clear()
        self._is_offloaded = False
        self._optimizer_id = None
        self._optimizer_bytes = 0
        self._onload_time = time.perf_counter() - t0
        if verbose:
            logger.info(
                "NPU training-state onload: optimizer=%d tensors %.1f MiB restored in %.2fs; "
                "scratch remains lazy",
                restored_tensors,
                restored_bytes / (1024 * 1024),
                self._onload_time,
            )
        return restored_bytes

    def _drop_reconstructible_caches(self, model: Any) -> tuple[int, int, int, int]:
        model_tensors = _drop_model_forward_state(model)
        model_bytes = _unique_storage_bytes(model_tensors)
        model_tensor_count = len({id(tensor) for tensor in model_tensors})
        total_bytes = model_bytes
        total_tensors = model_tensor_count
        # Drop the accounting references before collect/empty_cache. The owner
        # attributes are already reset, so this is the point where the pinned
        # autograd graph can actually become unreachable.
        del model_tensors

        getter = self._global_memory_buffer_getter
        if getter is None:
            try:
                from megatron.core.parallel_state import get_global_memory_buffer

                getter = get_global_memory_buffer
            except ImportError:
                getter = None

        if getter is not None:
            try:
                global_buffer = getter()
                buffer = getattr(global_buffer, "buffer", None)
                if isinstance(buffer, MutableMapping):
                    scratch = list(_iter_tensors(buffer))
                    total_bytes += _unique_storage_bytes(scratch)
                    total_tensors += len({id(tensor) for tensor in scratch})
                    buffer.clear()
            except (AssertionError, RuntimeError):
                # Megatron may not have initialized the singleton on empty
                # pipeline stages. There is no scratch state to release there.
                logger.debug("Megatron global memory buffer is unavailable", exc_info=True)

        try:
            from transformer_engine.pytorch.module.base import _dummy_wgrads

            if isinstance(_dummy_wgrads, MutableMapping):
                scratch = list(_iter_tensors(_dummy_wgrads))
                total_bytes += _unique_storage_bytes(scratch)
                total_tensors += len({id(tensor) for tensor in scratch})
                _dummy_wgrads.clear()
        except ImportError:
            pass

        gc.collect()
        return total_bytes, total_tensors, model_bytes, model_tensor_count

    @property
    def is_offloaded(self) -> bool:
        return self._is_offloaded

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "optimizer_offloaded_mb": self._optimizer_bytes / (1024 * 1024),
            "optimizer_tensors": len(self._saved),
            "scratch_discarded_mb": self._scratch_bytes / (1024 * 1024),
            "scratch_tensors": self._scratch_tensors,
            "model_runtime_discarded_mb": self._model_runtime_bytes / (1024 * 1024),
            "model_runtime_tensors": self._model_runtime_tensors,
            "allocated_reclaimed_mb": self._allocated_reclaimed_bytes / (1024 * 1024),
            "offload_time_s": round(self._offload_time, 2),
            "onload_time_s": round(self._onload_time, 2),
            "is_offloaded": self._is_offloaded,
        }


def _drop_model_forward_state(model: Any) -> list[torch.Tensor]:
    """Release completed-forward state through Megatron's owning objects.

    This runs only at the train-to-rollout phase boundary, after backward,
    ``optimizer.step`` and the actor CPU-weight backup have completed. Nothing
    cleared here is optimizer/model state: dispatcher metadata is assigned by
    ``dispatch_preprocess`` before every same-forward consumer.

    Returning the tensors lets the caller account unique storage before the
    last local references disappear. They are never copied to CPU.
    """
    if model is None:
        return []

    released: list[torch.Tensor] = []

    seen_dispatchers: set[int] = set()
    for module in _iter_model_modules(model):
        dispatcher = getattr(module, "token_dispatcher", None)
        state_names = _MOE_DISPATCHER_FORWARD_STATE.get(type(dispatcher).__name__)
        if state_names is not None and id(dispatcher) not in seen_dispatchers:
            seen_dispatchers.add(id(dispatcher))
            for name in state_names:
                released.extend(_iter_tensors(getattr(dispatcher, name, None)))
                if hasattr(dispatcher, name):
                    setattr(dispatcher, name, None)

    return released


def _iter_model_chunks(model: Any) -> Iterator[Any]:
    if isinstance(model, (list, tuple)):
        for child in model:
            yield from _iter_model_chunks(child)
    elif model is not None:
        yield model


def _iter_model_modules(model: Any) -> Iterator[torch.nn.Module]:
    seen: set[int] = set()
    for model_chunk in _iter_model_chunks(model):
        if not isinstance(model_chunk, torch.nn.Module):
            continue
        for module in model_chunk.modules():
            if id(module) not in seen:
                seen.add(id(module))
                yield module


def _iter_megatron_optimizers(optimizer: Any) -> Iterator[Any]:
    """Yield leaf Megatron optimizers, including nested ChainedOptimizer."""
    chained = getattr(optimizer, "chained_optimizers", None)
    if isinstance(chained, (list, tuple)):
        for child in chained:
            yield from _iter_megatron_optimizers(child)
        return
    if optimizer is not None:
        yield optimizer


def _iter_optimizer_device_tensors(optimizer: Any) -> Iterator[tuple[torch.Tensor, str]]:
    """Follow verl's Megatron/HDO optimizer-state traversal."""
    for optimizer_index, megatron_optimizer in enumerate(_iter_megatron_optimizers(optimizer)):
        prefix = f"optimizer[{optimizer_index}]"

        copy_params = getattr(megatron_optimizer, "shard_fp32_from_float16_groups", None)
        for index, tensor in enumerate(_iter_tensors(copy_params)):
            yield tensor, f"{prefix}.shard_fp32_from_float16_groups[{index}]"

        try:
            inner_optimizer = getattr(megatron_optimizer, "optimizer", None)
        except (AssertionError, RuntimeError):
            inner_optimizer = None
        if inner_optimizer is None:
            continue

        is_hdo = all(
            hasattr(inner_optimizer, attr)
            for attr in ("sub_optimizers", "inner_param_to_orig_param", "state")
        )
        if is_hdo:
            for sub_index, sub_optimizer in enumerate(inner_optimizer.sub_optimizers):
                for state_index, state in enumerate(getattr(sub_optimizer, "state", {}).values()):
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            yield value, f"{prefix}.hdo[{sub_index}].state[{state_index}].{key}"
            continue

        for state_index, state in enumerate(getattr(inner_optimizer, "state", {}).values()):
            for key in ("exp_avg", "exp_avg_sq", "master_param"):
                value = state.get(key)
                if isinstance(value, torch.Tensor):
                    yield value, f"{prefix}.state[{state_index}].{key}"


def _iter_tensors(value: Any) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_tensors(child)


def _move_tensor_data(tensor: torch.Tensor, device: str | torch.device) -> None:
    """Move storage while preserving every alias to the Tensor object itself."""
    if tensor.device == torch.device(device):
        return
    tensor.data = tensor.data.to(device, non_blocking=False)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    try:
        return tensor.untyped_storage().nbytes()
    except (AttributeError, RuntimeError):
        return tensor.numel() * tensor.element_size()


def _unique_storage_bytes(tensors: list[torch.Tensor]) -> int:
    total = 0
    seen: set[tuple[str, int]] = set()
    for tensor in tensors:
        try:
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            size = storage.nbytes()
        except (AttributeError, RuntimeError):
            key = (str(tensor.device), id(tensor))
            size = tensor.numel() * tensor.element_size()
        if key not in seen:
            seen.add(key)
            total += size
    return total


def _synchronize_and_empty_npu_cache() -> None:
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        return
    npu.synchronize()
    npu.empty_cache()


def _npu_memory_allocated() -> int:
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        return 0
    return int(npu.memory_allocated())
