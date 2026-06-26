#!/usr/bin/env python3
"""Quick standalone test of NPUWeightOffloader — no Megatron, no vLLM, no distributed.

Creates a dummy model on NPU, offloads to CPU, checks memory is freed,
onloads back, and verifies weight equality.

Usage: python tools/test_offloader_simple.py
Runtime: ~5 seconds
"""
import gc
import os
import sys
import time

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/workspace/vime")
from vime.utils.npu_weight_offloader import NPUWeightOffloader


def get_allocated_mb():
    """Return currently allocated NPU memory in MiB."""
    return torch.npu.memory_allocated() / (1024 * 1024)


def main():
    device = torch.device("npu:0")

    # Create a ~2 GiB dummy model on NPU
    print("Creating dummy model on NPU...")
    model = torch.nn.Sequential(*[
        torch.nn.Linear(8192, 8192, bias=False, device=device)
        for _ in range(8)  # 8 × 8192×8192 ≈ 2 GiB
    ])
    # Force allocation
    for p in model.parameters():
        p.data.zero_()
    torch.npu.synchronize()

    # Snapshot pre-offload weights
    pre_weights = {name: p.data.detach().cpu().clone() for name, p in model.named_parameters()}

    alloc_pre = get_allocated_mb()
    print(f"  Pre-offload allocated: {alloc_pre:.0f} MiB")

    # Offload
    offloader = NPUWeightOffloader()
    t0 = time.perf_counter()
    offloaded_bytes = offloader.offload(model)
    t_off = time.perf_counter() - t0
    alloc_off = get_allocated_mb()
    print(f"  Offloaded: {offloaded_bytes/1024**2:.0f} MiB in {t_off:.2f}s")
    print(f"  Post-offload allocated: {alloc_off:.0f} MiB (freed {alloc_pre - alloc_off:.0f} MiB)")

    assert alloc_off < alloc_pre * 0.1, \
        f"FAIL: expected >90% memory freed, got {alloc_off:.0f} / {alloc_pre:.0f} MiB"

    # Onload
    t0 = time.perf_counter()
    onloaded_bytes = offloader.onload(model)
    t_on = time.perf_counter() - t0
    alloc_on = get_allocated_mb()
    print(f"  Onloaded:  {onloaded_bytes/1024**2:.0f} MiB in {t_on:.2f}s")
    print(f"  Post-onload allocated: {alloc_on:.0f} MiB")

    # Verify weights match
    max_diff = 0.0
    for name, p in model.named_parameters():
        diff = (p.data - pre_weights[name].to(device)).abs().max().item()
        max_diff = max(max_diff, diff)
    print(f"  Max weight diff after round-trip: {max_diff:.2e}")

    assert max_diff < 1e-5, f"FAIL: weight mismatch after round-trip (max diff={max_diff})"

    # Verify params are back on NPU
    for name, p in model.named_parameters():
        assert p.device.type == "npu", f"FAIL: {name} on {p.device}, expected npu"

    print(f"\n=== ALL CHECKS PASSED ===")
    print(f"  Offload BW: {offloaded_bytes/1024**2/t_off:.0f} MiB/s")
    print(f"  Onload BW:  {onloaded_bytes/1024**2/t_on:.0f} MiB/s")


if __name__ == "__main__":
    main()
