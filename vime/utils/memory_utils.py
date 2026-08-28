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


def mem_probe_enabled() -> bool:
    """Whether ``VIME_MEM_PROBE`` asks for the memory probes.

    Strictly ``"1"``, matching what the probes themselves have always checked.
    A looser parse would let ``VIME_MEM_PROBE=true`` turn the hand-off probes on
    while leaving the train-step ones silent, which is worse than not having the
    helper at all.
    """
    return os.environ.get("VIME_MEM_PROBE", "0") == "1"


def _log_npu_mem(tag: str, step_id: int = -1) -> None:
    """[MEM PROBE] 实时 NPU 显存打印(env VIME_MEM_PROBE=1 开,默认关=零开销)。

    区分 torch 池内 vs 池外(非-torch)占用——即 OOM 消息里 `total - torch_reserved` 那 ~24 GiB
    torch 看不见的部分(CANN/HCCL/MindSpeed AscendC 算子 workspace)。
      non_torch = device_used - torch_reserved
    若 non_torch 很大 → 真·非-torch 占用(随 seq 长涨);若 device_used ≈ torch_reserved 却 OOM →
    是设备级碎片(torch 要不到连续块),而非池外占用。用 mem_get_info() 拿设备级 free/total。
    """
    if not mem_probe_enabled():
        return
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        return
    try:
        g = 1024 ** 3
        reserved = npu.memory_reserved() / g
        free, total = npu.mem_get_info()
        dev_used = (total - free) / g
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        logger.info(
            "[MEM %s] step=%d rank=%d torch_alloc=%.2f torch_reserved=%.2f max_reserved=%.2f "
            "device_used=%.2f non_torch=%.2f dev_free=%.2f total=%.2f GiB",
            tag, step_id, rank, npu.memory_allocated() / g, reserved, npu.max_memory_reserved() / g,
            dev_used, dev_used - reserved, free / g, total / g,
        )
    except Exception as e:  # a probe must never break training
        logger.warning("[MEM %s] probe failed: %s", tag, e)


def _log_npu_expandable(tag: str, step_id: int = -1) -> None:
    """[MEM PROBE] 证明 expandable_segments 是否**真激活**、覆盖范围、以及缓存空闲可归还比例。

    torch_npu 的 memory_snapshot() 逐 segment 暴露 `is_expandable`——env 设了但静默回退成 False
    (torch_npu 有 "expandable_segments setting failure, now change to False" 的兜底)时,这里会
    如实显示 expandable=0。回答三问:
      (1) PYTORCH_NPU_ALLOC_CONF 到底有没有进 actor 进程(env 转发验证);
      (2) 分配器里多少 segment 真的 is_expandable(生效 + 覆盖 MindSpeed 全部分配的验证);
      (3) cached_free(reserved−active)里多少落在**完全空闲**段 = empty_cache 保底能归还的量,
          剩余是卡在半用段里的碎片(expandable 下仍可按页 unmap,但保守下界看 fully_free)。
    默认关(VIME_MEM_PROBE=1 开);snapshot 有开销,仅探针开时调用。
    """
    if not mem_probe_enabled():
        return
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        return
    try:
        g = 1024 ** 3
        conf = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "<unset>")
        segs = npu.memory_snapshot()
        n = len(segs)
        n_exp = sum(1 for s in segs if s.get("is_expandable"))
        exp_res = sum(s["total_size"] for s in segs if s.get("is_expandable")) / g
        leg_res = sum(s["total_size"] for s in segs if not s.get("is_expandable")) / g
        cached_free = sum(s["total_size"] - s["active_size"] for s in segs) / g
        fully_free = sum(s["total_size"] for s in segs if s["active_size"] == 0) / g
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        logger.info(
            "[MEM-EXP %s] step=%d rank=%d conf=%s segs=%d expandable=%d exp_reserved=%.2f "
            "legacy_reserved=%.2f cached_free=%.2f fully_free_seg=%.2f GiB",
            tag, step_id, rank, conf, n, n_exp, exp_res, leg_res, cached_free, fully_free,
        )
    except Exception as e:  # a probe must never break training
        logger.warning("[MEM-EXP %s] probe failed: %s", tag, e)


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
