import gc
import logging
import os

import psutil
import torch
import torch.distributed as dist
from vime.utils.common import is_npu

logger = logging.getLogger(__name__)

_GIB = 1024**3


def clear_memory(clear_host_memory: bool = False):
    if is_npu():
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    if is_npu():
        torch.npu.empty_cache()
    if clear_host_memory:
        if is_npu():
            torch.npu.empty_cache()
        else:
            torch._C._host_emptyCache()


def expandable_segments_enabled() -> bool:
    """Return the training allocator's configured expandable-segment policy."""
    config = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
    for setting in config.split(","):
        name, separator, value = setting.partition(":")
        if separator and name.strip().lower() == "expandable_segments":
            return value.strip().lower() in {"1", "true"}
    return False


def set_expandable_segments(enable: bool) -> None:
    """Apply verl's runtime expandable-segment switch on torch-npu."""
    if not is_npu():
        return

    try:
        torch.npu.memory._set_allocator_settings(f"expandable_segments:{enable}")
        logger.info("Set NPU allocator expandable_segments=%s", enable)
    except Exception:
        # Match verl's NPU platform behavior: older torch-npu builds keep their
        # existing allocator policy, while the rest of the handoff still runs.
        logger.warning(
            "Current torch-npu does not support runtime allocator settings; "
            "continuing with the existing expandable_segments policy",
            exc_info=True,
        )


def aggressive_empty_cache(force_sync: bool = True, max_retries: int = 3) -> None:
    """Release cached device blocks using verl's bounded multi-pass cleanup."""
    device = torch.npu if is_npu() else torch.cuda
    if not device.is_available():
        return

    for attempt in range(max_retries):
        before_reserved = device.memory_reserved()
        before_allocated = device.memory_allocated()

        gc.collect()
        device.empty_cache()
        if force_sync:
            device.synchronize()

        after_reserved = device.memory_reserved()
        after_allocated = device.memory_allocated()
        reserved_freed = before_reserved - after_reserved
        allocated_freed = before_allocated - after_allocated
        logger.info(
            "Memory cleanup attempt %d: freed %.2f GiB reserved, %.2f GiB allocated",
            attempt + 1,
            reserved_freed / _GIB,
            allocated_freed / _GIB,
        )

        # A following pass is only worthwhile when the last pass returned at
        # least 1 GiB. Synchronization can make more blocks reclaimable on the
        # next pass, but the loop remains strictly bounded.
        if reserved_freed < _GIB:
            break


def available_memory():
    if is_npu():
        device = torch.npu.current_device()
        free, total = torch.npu.mem_get_info(device)
        vm = psutil.virtual_memory()
        return {
            "gpu": str(device),
            "total_GB": _byte_to_gb(total),
            "free_GB": _byte_to_gb(free),
            "used_GB": _byte_to_gb(total - free),
            "allocated_GB": _byte_to_gb(torch.npu.memory_allocated(device)),
            "reserved_GB": _byte_to_gb(torch.npu.memory_reserved(device)),
            "host_total_GB": _byte_to_gb(vm.total),
            "host_available_GB": _byte_to_gb(vm.available),
            "host_used_GB": _byte_to_gb(vm.used),
            "host_free_GB": _byte_to_gb(vm.free),
        }
    else:
        device = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(device)
        vm = psutil.virtual_memory()
        return {
            "gpu": str(device),
            "total_GB": _byte_to_gb(total),
            "free_GB": _byte_to_gb(free),
            "used_GB": _byte_to_gb(total - free),
            "allocated_GB": _byte_to_gb(torch.cuda.memory_allocated(device)),
            "reserved_GB": _byte_to_gb(torch.cuda.memory_reserved(device)),
            "host_total_GB": _byte_to_gb(vm.total),
            "host_available_GB": _byte_to_gb(vm.available),
            "host_used_GB": _byte_to_gb(vm.used),
            "host_free_GB": _byte_to_gb(vm.free),
        }


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def cpu_tensor_breakdown():
    """Per-process CPU footprint + torch CPU-tensor split (pinned B-mode backups
    vs other = optimizer states + grad). Gated behind ``VIME_CPU_MEM_PROBE=1``
    because it walks ``gc.get_objects()``. Dedups by storage data_ptr so tensor
    views don't double-count. Returns None when disabled.
    """
    import os

    if os.environ.get("VIME_CPU_MEM_PROBE", "0") != "1":
        return None

    pinned = other = 0
    n_pinned = n_other = 0
    seen = set()
    for o in gc.get_objects():
        try:
            if isinstance(o, torch.Tensor) and o.device.type == "cpu" and o.numel() > 0:
                st = o.untyped_storage()
                key = st.data_ptr()
                if key in seen:
                    continue
                seen.add(key)
                nb = st.nbytes()
                if o.is_pinned():
                    pinned += nb
                    n_pinned += 1
                else:
                    other += nb
                    n_other += 1
        except Exception:
            continue

    # This process's own CPU footprint — PSS correctly accounts /dev/shm sharing.
    rss = pss = 0
    try:
        with open("/proc/self/smaps_rollup") as f:
            for line in f:
                if line.startswith("Rss:"):
                    rss = int(line.split()[1])
                elif line.startswith("Pss:"):
                    pss = int(line.split()[1])
    except Exception:
        pass

    return {
        "proc_rss_GB": round(rss / 1048576, 1),
        "proc_pss_GB": round(pss / 1048576, 1),
        "torch_cpu_pinned_GB": round(pinned / 1e9, 1),
        "torch_cpu_other_GB": round(other / 1e9, 1),
        "n_pinned": n_pinned,
        "n_other": n_other,
    }


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    cpu_bd = cpu_tensor_breakdown()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
        + (f" | CPU-BD {cpu_bd}" if cpu_bd else "")
    )
    return memory_info
