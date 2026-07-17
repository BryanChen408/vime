import gc
import logging

import psutil
import torch
import torch.distributed as dist
from vime.utils.common import is_npu

logger = logging.getLogger(__name__)


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
