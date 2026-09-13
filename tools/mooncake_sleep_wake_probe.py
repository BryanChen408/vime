#!/usr/bin/env python3
"""Minimal Mooncake/ADXL sleep-wake staleness probe (root cause B discriminator).

Reproduces the exact vllm-ascend CaMem level-2 sleep/wake VMM sequence
(aclrtUnmapMem + aclrtFreePhysical -> aclrtMallocPhysical + aclrtMapMem on the
SAME VA) on a Mooncake-registered buffer, with no training stack involved.

Two processes:
  P (producer): allocates a CaMem-pool buffer on its NPU, registers it with
    the transfer engine, and walks the phase ladder.
  D (consumer): reads P's buffer via batch_transfer_sync_read at each phase
    and reports which pattern actually arrived.

Phases (each is independently decisive):
  phase0_baseline   : P fills 0xA0, D reads -> must see 0xA0 (sanity).
  phase1_no_refresh : P sleep+wake (same VA, NEW physical pages), fills 0xB0,
                      D reads with NO refresh anywhere -> root-cause-B repro.
                      Expected: NOT 0xB0 (zeros or stale 0xA0).
  phase2_p_rereg    : P unregister+register same VA; D reads without touching
                      its own engine -> is P-side re-registration enough?
  phase3_d_recreate : D builds a NEW TransferEngine (drops the cached segment
                      handle_map_) and reads -> is dual-side refresh enough?
  phase4_fresh_va   : P allocates a NEW VA, registers it, fills 0xC0; D reads
                      the new VA -> is the staleness VA-keyed?

Results are appended as JSON lines to <coord_dir>/results.jsonl so they
survive the known gflags heap-corruption abort at process exit.

Usage:
  python3 mooncake_sleep_wake_probe.py P <npu_id> <host_ip> <coord_dir>
  python3 mooncake_sleep_wake_probe.py D <npu_id> <host_ip> <coord_dir>
"""

import json
import os
import sys
import time

SIZE = 64 * 1024 * 1024  # 64 MiB, huge-page backed like production KV
# Single-slice transfer length; production per-layer chunks are far smaller
# than the whole region, and ADXL may reject oversized slices.
XFER_BYTES = int(os.environ.get("PROBE_XFER_BYTES", str(4 * 1024 * 1024)))
PHASE_TIMEOUT_S = 180

PATTERN = {"phase0": 0xA0, "phase1": 0xB0, "phase4": 0xC0}


def log(role, msg):
    print(f"[PROBE][{role}] {msg}", flush=True)


def record(coord_dir, role, phase, **kw):
    entry = {"role": role, "phase": phase, "ts": time.time(), **kw}
    with open(os.path.join(coord_dir, "results.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")
    log(role, f"RESULT {json.dumps(entry)}")


def publish(coord_dir, name, payload):
    tmp = os.path.join(coord_dir, f".{name}.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, os.path.join(coord_dir, name))


def wait_for(coord_dir, name, timeout=PHASE_TIMEOUT_S):
    path = os.path.join(coord_dir, name)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        time.sleep(0.5)
    raise TimeoutError(f"timed out waiting for {name}")


def classify(recv_tensor, expected_byte):
    """Classify what actually landed in the receive buffer."""
    import torch

    sample = recv_tensor
    uniq = torch.unique(sample)
    uniq_bytes = sorted(int(b) for b in uniq.tolist())
    total = sample.numel()
    match = int((sample == expected_byte).sum().item())
    zeros = int((sample == 0).sum().item())
    return {
        "expected": expected_byte,
        "match_ratio": round(match / total, 6),
        "zero_ratio": round(zeros / total, 6),
        "unique_bytes_head": uniq_bytes[:8],
        "n_unique": len(uniq_bytes),
    }


def main():
    role = sys.argv[1]
    npu_id = int(sys.argv[2])
    host_ip = sys.argv[3]
    coord_dir = sys.argv[4]
    os.makedirs(coord_dir, exist_ok=True)

    # Mirror production: each engine process only sees its own cards
    # (ASCEND_RT_VISIBLE_DEVICES set before torch_npu import), ADXL then
    # binds by logical device id like it does in the engine WorkerProcs.
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(npu_id)

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    log(role, f"torch npu device set to logical 0 (physical {npu_id})")

    from mooncake.engine import TransferEngine

    engine = TransferEngine()
    ret = engine.initialize(host_ip, "P2PHANDSHAKE", "ascend", "")
    assert ret == 0, f"engine initialize failed ret={ret}"
    rpc_port = engine.get_rpc_port()
    session = f"{host_ip}:{rpc_port}"
    log(role, f"engine up session={session}")

    if role == "P":
        if os.environ.get("PROBE_MODE") == "nosleep_dual":
            run_producer_nosleep(engine, session, coord_dir)
        elif os.environ.get("PROBE_MODE") == "alias":
            run_producer_alias(engine, session, coord_dir)
        else:
            run_producer(engine, session, npu_id, coord_dir)
    else:
        if os.environ.get("PROBE_MODE") == "nosleep_dual":
            run_consumer_nosleep(engine, session, coord_dir)
        elif os.environ.get("PROBE_MODE") == "alias":
            run_consumer_alias(engine, session, coord_dir)
        else:
            run_consumer(engine, session, npu_id, coord_dir)
    return 0


def _alias_lib():
    """ctypes bindings for the ACL VMM functions used to alias-map a CaMem
    physical handle onto a second (fresh) virtual address."""
    import ctypes

    lib = ctypes.CDLL("libascendcl.so")
    lib.aclrtReserveMemAddress.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_ulonglong,
    ]
    lib.aclrtReserveMemAddress.restype = ctypes.c_int
    lib.aclrtMapMem.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_ulonglong]
    lib.aclrtMapMem.restype = ctypes.c_int
    lib.aclrtUnmapMem.argtypes = [ctypes.c_void_p]
    lib.aclrtUnmapMem.restype = ctypes.c_int
    return lib, ctypes


def alias_map(lib, ct, p_mem_handle, size):
    """Reserve a fresh VA and map it to the physical handle stored at
    p_mem_handle (the C-level aclrtDrvMemHandle* owned by CaMem)."""
    handle = ct.c_uint64.from_address(p_mem_handle).value
    alias = ct.c_void_p()
    ret = lib.aclrtReserveMemAddress(ct.byref(alias), size, 0, None, 0)
    assert ret == 0, f"aclrtReserveMemAddress(alias) failed ret={ret}"
    ret = lib.aclrtMapMem(alias.value, size, 0, handle, 0)
    assert ret == 0, f"aclrtMapMem(alias) failed ret={ret}"
    return alias.value


def run_producer_alias(engine, session, coord_dir):
    """Alias-VA design validation (RoCE path):
    phase0: alias-map the pool tensor's physical handle to a fresh VA,
            register the ALIAS, D reads it -> proves alias + registration work.
    phase1: unregister+unmap alias, CaMem sleep/wake (new physical pages at
            the SAME compute VA), map a NEW alias VA to the new handle,
            register it, fill new pattern via compute VA -> D reads the new
            alias and must see the new pattern.  This is the root fix.
    """
    import torch

    alloc, kv = make_kv_tensor("kv")
    va = kv.data_ptr()
    handle4 = alloc.pointer_to_data[va].handle  # (device, alignedSize, d_mem, p_memHandle)
    size = handle4[1]
    lib, ct = _alias_lib()

    alias1 = alias_map(lib, ct, handle4[3], size)
    kv.fill_(PATTERN["phase0"])
    torch.npu.synchronize()
    ret = engine.register_memory(alias1, size)
    assert ret == 0, f"register alias failed ret={ret}"
    record(coord_dir, "P", "alias_phase0", compute_va=hex(va), alias_va=hex(alias1), size=size)
    publish(coord_dir, "p_session", {"session": session, "va": alias1, "size": size})
    publish(coord_dir, "p_phase0", {"pattern": PATTERN["phase0"]})
    wait_for(coord_dir, "d_phase0")

    # ---- sleep/wake with alias rotation ----
    ret = engine.unregister_memory(alias1)
    log("P", f"unregister alias1 ret={ret}")
    ret = lib.aclrtUnmapMem(alias1)
    assert ret == 0, f"aclrtUnmapMem(alias1) failed ret={ret}"
    camem_sleep_wake(alloc)  # unmap primary + free physical; wake: NEW handle mapped at same VA
    assert kv.data_ptr() == va
    kv.fill_(PATTERN["phase1"])
    torch.npu.synchronize()
    selfcheck = int(kv[: 1024 * 1024].sum().item()) == PATTERN["phase1"] * 1024 * 1024
    alias2 = alias_map(lib, ct, handle4[3], size)
    assert alias2 != alias1, "alias VA must be fresh (old one is NIC-poisoned)"
    ret = engine.register_memory(alias2, size)
    assert ret == 0, f"register alias2 failed ret={ret}"
    record(
        coord_dir,
        "P",
        "alias_phase1",
        alias2=hex(alias2),
        alias1=hex(alias1),
        alias_fresh=alias2 != alias1,
        selfcheck_ok=selfcheck,
    )
    publish(coord_dir, "p_phase1", {"va": alias2, "pattern": PATTERN["phase1"]})
    wait_for(coord_dir, "d_phase1")
    log("P", "alias mode done")


def run_consumer_alias(engine, session, coord_dir):
    import torch

    _alloc, recv = make_kv_tensor("recv")
    ret = engine.register_memory(recv.data_ptr(), SIZE)
    assert ret == 0, f"D register failed ret={ret}"

    info = wait_for(coord_dir, "p_session")
    p_session, p_alias = info["session"], info["va"]

    wait_for(coord_dir, "p_phase0")
    recv.zero_()
    ret, n = read_remote(engine, p_session, p_alias, recv)
    cls = classify(recv[:n], PATTERN["phase0"])
    record(coord_dir, "D", "alias_phase0_baseline", transfer_ret=ret, **cls)
    publish(coord_dir, "d_phase0", {})

    p1 = wait_for(coord_dir, "p_phase1")
    # fresh engine -> drops cached segment descriptor (which holds alias1)
    from mooncake.engine import TransferEngine

    engine2 = TransferEngine()
    ret = engine2.initialize(sys.argv[3], "P2PHANDSHAKE", "ascend", "")
    assert ret == 0, f"D engine2 init failed ret={ret}"
    ret = engine2.register_memory(recv.data_ptr(), SIZE)
    assert ret == 0
    recv.zero_()
    ret, n = read_remote(engine2, p_session, p1["va"], recv)
    cls = classify(recv[:n], PATTERN["phase1"])
    record(coord_dir, "D", "alias_phase1_post_wake", transfer_ret=ret, **cls)
    publish(coord_dir, "d_phase1", {})
    log("D", "alias mode done")


def run_producer_nosleep(engine, session, coord_dir):
    """No sleep at all: register two pool VAs upfront. Discriminates whether
    va2 transfer failures are caused by the sleep/wake cycle or by ADXL
    mishandling a second registered region.
    PROBE_DUAL_POOLS=0 -> both tensors from ONE pool context (production shape).
    """
    import torch

    from vllm_ascend.device_allocator.camem import CaMemAllocator

    alloc = CaMemAllocator.get_instance()
    if os.environ.get("PROBE_DUAL_POOLS", "1") == "1":
        _, kv = make_kv_tensor("kv")
        _, kv2 = make_kv_tensor("kv2")
    else:
        with alloc.use_memory_pool(tag="kv"):
            kv = torch.empty(SIZE, dtype=torch.uint8, device="npu")
            kv2 = torch.empty(SIZE, dtype=torch.uint8, device="npu")
    va, va2 = kv.data_ptr(), kv2.data_ptr()
    swap = os.environ.get("PROBE_SWAP_REG", "0") == "1"
    order = [(va2, PATTERN["phase4"], "va2"), (va, PATTERN["phase0"], "va1")] if swap else [
        (va, PATTERN["phase0"], "va1"),
        (va2, PATTERN["phase4"], "va2"),
    ]
    for addr, _pat, _name in order:
        assert engine.register_memory(addr, SIZE) == 0
    kv.fill_(PATTERN["phase0"])
    kv2.fill_(PATTERN["phase4"])
    torch.npu.synchronize()
    record(coord_dir, "P", "nosleep_dual_fill", va=hex(va), va2=hex(va2), swap=swap)
    publish(coord_dir, "p_session", {"session": session, "va": va, "va2": va2, "size": SIZE, "swap": swap})
    publish(coord_dir, "p_ready", {})
    wait_for(coord_dir, "d_done")
    log("P", "nosleep_dual done")


def run_consumer_nosleep(engine, session, coord_dir):
    import torch

    from vllm_ascend.device_allocator.camem import CaMemAllocator

    dual_recv = os.environ.get("PROBE_DUAL_RECV", "0") == "1"
    alloc = CaMemAllocator.get_instance()
    with alloc.use_memory_pool(tag="recv"):
        recv = torch.empty(SIZE, dtype=torch.uint8, device="npu")
        recv2 = torch.empty(SIZE, dtype=torch.uint8, device="npu") if dual_recv else None
    assert engine.register_memory(recv.data_ptr(), SIZE) == 0
    if recv2 is not None:
        assert engine.register_memory(recv2.data_ptr(), SIZE) == 0
    log("D", f"recv va={hex(recv.data_ptr())} recv2={hex(recv2.data_ptr()) if recv2 is not None else None}")
    info = wait_for(coord_dir, "p_session")
    p_session, p_va, p_va2 = info["session"], info["va"], info["va2"]
    wait_for(coord_dir, "p_ready")

    read_order = [(p_va2, PATTERN["phase4"], "nosleep_va2_first"), (p_va, PATTERN["phase0"], "nosleep_va1")] if info.get(
        "swap"
    ) else [(p_va, PATTERN["phase0"], "nosleep_va1"), (p_va2, PATTERN["phase4"], "nosleep_va2")]
    for i, (addr, pat, name) in enumerate(read_order):
        dst = recv
        if dual_recv and recv2 is not None:
            # index-matched dst: P's region[i] -> D's region[i]
            dst = recv if (addr == p_va) else recv2
        dst.zero_()
        ret, _n = read_remote(engine, p_session, addr, dst)
        record(coord_dir, "D", name, transfer_ret=ret, dst_va=hex(dst.data_ptr()), **classify(dst[: 4 * 1024 * 1024], pat))
    publish(coord_dir, "d_done", {})
    log("D", "nosleep_dual done")


def make_kv_tensor(tag):
    """Allocate a buffer through the production CaMem pool path."""
    from vllm_ascend.device_allocator.camem import CaMemAllocator

    alloc = CaMemAllocator.get_instance()
    import torch

    with alloc.use_memory_pool(tag=tag):
        kv = torch.empty(SIZE, dtype=torch.uint8, device="npu")
    return alloc, kv


def camem_sleep_wake(alloc):
    """Exact production level-2 sequence: KV tag is NOT offloaded -> the
    physical pages are released; wake remaps the SAME VA to NEW pages."""
    alloc.sleep(offload_tags=("weights",))  # our kv tag is discarded, not offloaded
    alloc.wake_up()


def run_producer(engine, session, npu_id, coord_dir):
    import torch

    alloc, kv = make_kv_tensor("kv")
    va = kv.data_ptr()
    log("P", f"kv tensor va={hex(va)} size={SIZE}")
    ret = engine.register_memory(va, SIZE)
    assert ret == 0, f"register_memory failed ret={ret}"
    log("P", "registered kv region")

    # ---- phase 0: baseline ----
    kv.fill_(PATTERN["phase0"])
    torch.npu.synchronize()
    selfcheck = int(kv[: 1024 * 1024].sum().item()) == PATTERN["phase0"] * 1024 * 1024
    record(coord_dir, "P", "phase0_fill", pattern=PATTERN["phase0"], va=hex(va), selfcheck_ok=selfcheck)
    publish(coord_dir, "p_session", {"session": session, "va": va, "size": SIZE})
    publish(coord_dir, "p_phase0", {"va": va, "pattern": PATTERN["phase0"]})
    wait_for(coord_dir, "d_phase0")

    # ---- phase 1: unregister WHILE STILL MAPPED -> sleep+wake -> re-register ----
    # Dereg-ordering hypothesis: DeregisterMem after unmap/remap operates on a
    # dangling handle (pages already freed) and may be swallowed, while the
    # follow-up RegisterMem dedupes by VA and keeps the stale NIC binding.
    # Tearing down the registration before unmap should force a genuinely
    # fresh binding to the remapped pages at wake.
    ret = engine.unregister_memory(va)
    log("P", f"pre-sleep unregister_memory ret={ret}")
    camem_sleep_wake(alloc)
    va_after = kv.data_ptr()
    kv.fill_(PATTERN["phase1"])
    torch.npu.synchronize()
    selfcheck = int(kv[: 1024 * 1024].sum().item()) == PATTERN["phase1"] * 1024 * 1024
    ret = engine.register_memory(va, SIZE)
    log("P", f"post-wake re-register_memory ret={ret}")
    record(
        coord_dir,
        "P",
        "phase1_refill",
        pattern=PATTERN["phase1"],
        va_before=hex(va),
        va_after=hex(va_after),
        va_same=va_after == va,
        selfcheck_ok=selfcheck,
        rereg_ret=ret,
    )
    publish(coord_dir, "p_phase1", {"pattern": PATTERN["phase1"]})
    wait_for(coord_dir, "d_phase1")
    wait_for(coord_dir, "d_phase3")

    # ---- phase 4: fresh VA ----
    alloc2, kv2 = make_kv_tensor("kv2")
    assert alloc2 is alloc
    va2 = kv2.data_ptr()
    kv2.fill_(PATTERN["phase4"])
    torch.npu.synchronize()
    ret = engine.register_memory(va2, SIZE)
    log("P", f"fresh va={hex(va2)} register ret={ret} (old va={hex(va)})")
    record(coord_dir, "P", "phase4_fresh_va", va2=hex(va2), va_same=va2 == va, ret=ret)
    publish(coord_dir, "p_phase4", {"va": va2, "pattern": PATTERN["phase4"]})
    wait_for(coord_dir, "d_phase4")
    wait_for(coord_dir, "d_phase5")

    # ---- phase 6: D remaps its destination VA; is incoming data landed in
    # the CURRENT pages? P re-fills the old va (P-side binding is stale, so
    # the wire carries whatever the old pages hold — 0xA0). If D reads back
    # 0xA0 the bytes landed in recv's current pages (D-side same-VA OK);
    # if recv still shows 0xEE the bytes went to the freed old pages
    # (D-side binding stale too -> both sides need fresh VA at wake).
    kv.fill_(0xD0)
    torch.npu.synchronize()
    publish(coord_dir, "p_phase6", {"pattern": 0xD0})
    wait_for(coord_dir, "d_phase6")

    log("P", "all phases done")


def read_remote(engine, session, remote_va, recv):
    n = min(XFER_BYTES, recv.numel())
    ret = engine.batch_transfer_sync_read(session, [remote_va], [recv.data_ptr()], [n])
    import torch

    torch.npu.synchronize()
    return ret, n


def run_consumer(engine, session, npu_id, coord_dir):
    import torch

    # Receive buffer through the same CaMem huge-page pool as production KV.
    _alloc, recv_t = make_kv_tensor("recv")
    recv = recv_t
    ret = engine.register_memory(recv.data_ptr(), SIZE)
    assert ret == 0, f"D register_memory failed ret={ret}"
    log("D", f"recv buffer va={hex(recv.data_ptr())} registered")

    p_session_info = wait_for(coord_dir, "p_session")
    p_session = p_session_info["session"]
    p_va = p_session_info["va"]
    log("D", f"peer session={p_session} peer va={hex(p_va)}")

    # ---- phase 0: baseline ----
    wait_for(coord_dir, "p_phase0")
    recv.zero_()
    ret, _n = read_remote(engine, p_session, p_va, recv)
    cls = classify(recv[: 4 * 1024 * 1024], PATTERN["phase0"])
    record(coord_dir, "D", "phase0_baseline", transfer_ret=ret, **cls)
    publish(coord_dir, "d_phase0", {})

    # ---- phase 1: P unregistered before sleep, re-registered after wake ----
    wait_for(coord_dir, "p_phase1")
    recv.zero_()
    ret, _n = read_remote(engine, p_session, p_va, recv)
    cls = classify(recv[: 4 * 1024 * 1024], PATTERN["phase1"])
    record(coord_dir, "D", "phase1_early_dereg_same_engine", transfer_ret=ret, **cls)
    publish(coord_dir, "d_phase1", {})

    # ---- phase 3: D rebuilds its engine (drops cached segment handle) ----
    del engine
    from mooncake.engine import TransferEngine

    engine2 = TransferEngine()
    ret = engine2.initialize(sys.argv[3], "P2PHANDSHAKE", "ascend", "")
    assert ret == 0, f"D engine2 initialize failed ret={ret}"
    ret = engine2.register_memory(recv.data_ptr(), SIZE)
    assert ret == 0, f"D engine2 register failed ret={ret}"
    log("D", f"engine2 up session={sys.argv[3]}:{engine2.get_rpc_port()}")
    recv.zero_()
    ret, _n = read_remote(engine2, p_session, p_va, recv)
    cls = classify(recv[: 4 * 1024 * 1024], PATTERN["phase1"])
    record(coord_dir, "D", "phase3_dual_refresh", transfer_ret=ret, **cls)
    publish(coord_dir, "d_phase3", {})

    # ---- phase 4: fresh VA on P ----
    p4 = wait_for(coord_dir, "p_phase4")
    # engine2 fetched the segment desc at phase3, before va2 was registered;
    # use a fresh engine so the new region is in the descriptor. ADXL pairs
    # transfers by registration INDEX (proven by run13): P's list is
    # [va, va2] here, so D must register two regions too and read va2 into
    # the second one.
    from mooncake.engine import TransferEngine

    from vllm_ascend.device_allocator.camem import CaMemAllocator

    alloc_d = CaMemAllocator.get_instance()
    with alloc_d.use_memory_pool(tag="recv2"):
        recv2 = torch.empty(SIZE, dtype=torch.uint8, device="npu")
    engine3 = TransferEngine()
    ret = engine3.initialize(sys.argv[3], "P2PHANDSHAKE", "ascend", "")
    assert ret == 0, f"D engine3 initialize failed ret={ret}"
    ret = engine3.register_memory(recv.data_ptr(), SIZE)
    assert ret == 0, f"D engine3 register recv failed ret={ret}"
    ret = engine3.register_memory(recv2.data_ptr(), SIZE)
    assert ret == 0, f"D engine3 register recv2 failed ret={ret}"
    recv2.zero_()
    ret, _n = read_remote(engine3, p_session, p4["va"], recv2)
    cls = classify(recv2[: 4 * 1024 * 1024], PATTERN["phase4"])
    record(coord_dir, "D", "phase4_fresh_va", transfer_ret=ret, dst_va=hex(recv2.data_ptr()), **cls)
    publish(coord_dir, "d_phase4", {})

    # ---- phase 5: control — engine3 re-reads the ORIGINAL va ----
    # If this returns stale data with ret=0, D-side engine3 is healthy and the
    # phase4 failure is specific to P's post-wake va2 registration.
    recv.zero_()
    ret, _n = read_remote(engine3, p_session, p_va, recv)
    cls = classify(recv[: 4 * 1024 * 1024], PATTERN["phase1"])
    record(coord_dir, "D", "phase5_engine3_old_va", transfer_ret=ret, **cls)
    publish(coord_dir, "d_phase5", {})

    # ---- phase 6: D remaps its OWN destination VA, then receives into it ----
    ret = engine3.unregister_memory(recv.data_ptr())
    log("D", f"phase6 pre-sleep unregister recv ret={ret}")
    camem_sleep_wake(alloc_d)
    recv.fill_(0xEE)
    torch.npu.synchronize()
    selfcheck = int(recv[: 1024 * 1024].sum().item()) == 0xEE * 1024 * 1024
    ret = engine3.register_memory(recv.data_ptr(), SIZE)
    log("D", f"phase6 post-wake re-register recv ret={ret} selfcheck={selfcheck}")
    wait_for(coord_dir, "p_phase6")
    ret, _n = read_remote(engine3, p_session, p_va, recv)
    cls = classify(recv[: 4 * 1024 * 1024], 0xA0)
    ee_ratio = round(int((recv[: 4 * 1024 * 1024] == 0xEE).sum().item()) / (4 * 1024 * 1024), 6)
    record(coord_dir, "D", "phase6_d_remapped_dst", transfer_ret=ret, ee_ratio=ee_ratio, **cls)
    publish(coord_dir, "d_phase6", {})

    log("D", "all phases done")


if __name__ == "__main__":
    sys.exit(main())
