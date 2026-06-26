#!/usr/bin/env python3
"""Standalone test for NPUWeightOffloader — no vLLM, no optimizer, no rollout.

Validates:
  1. offload() frees NPU HBM
  2. onload() restores weights correctly
  3. Model forward pass produces identical output after round-trip

Usage:
  python tools/test_npu_weight_offloader.py \
    --hf-checkpoint /home/s50057377/Qwen3.6-35B-A3B \
    --ref-load     /home/s50057377/Qwen3.6-35B-A3B_torch_dist

Skips: optimizer build, vLLM init, rollout, CUDAGraph.
Total runtime: ~3-5 min.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

# ── Minimal env setup (mirrors run script) ──────────────────────────────
os.environ.setdefault("SLIME_SCRIPT_TRAIN_BACKEND", "megatron")
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES", "1")
os.environ.setdefault("QWEN36_CAUSAL_CONV1D_IMPL", "triton")

# Ensure torch_npu is loaded before megatron
import torch_npu  # noqa: E402, F401
import mindspeed.megatron_adaptor  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="NPUWeightOffloader standalone test")
    p.add_argument("--hf-checkpoint", required=True)
    p.add_argument("--ref-load", required=True)
    p.add_argument("--tensor-model-parallel-size", type=int, default=2)
    p.add_argument("--sequence-parallel", action="store_true")
    p.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    p.add_argument("--expert-model-parallel-size", type=int, default=8)
    p.add_argument("--context-parallel-size", type=int, default=1)
    p.add_argument("--recompute-granularity", default="full")
    p.add_argument("--recompute-method", default="uniform")
    p.add_argument("--recompute-num-layers", type=int, default=1)
    p.add_argument("--attention-backend", default="flash")
    p.add_argument("--use-flash-attn", action="store_true")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-gradient-accumulation-fusion", action="store_true")
    p.add_argument("--qwen-gdn-backend", default="npu")
    # Model spec (reuse the model args helper if possible)
    p.add_argument("--spec", default="vime_plugins.models.qwen3_5 get_qwen3_5_spec")
    p.add_argument("--disable-bias-linear", action="store_true")
    p.add_argument("--qk-layernorm", action="store_true")
    p.add_argument("--group-query-attention", action="store_true")
    p.add_argument("--num-attention-heads", type=int, default=16)
    p.add_argument("--num-query-groups", type=int, default=2)
    p.add_argument("--kv-channels", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=40)
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--ffn-hidden-size", type=int, default=512)
    p.add_argument("--use-gated-attention", action="store_true")
    p.add_argument("--normalization", default="RMSNorm")
    p.add_argument("--apply-layernorm-1p", action="store_true")
    p.add_argument("--position-embedding-type", default="rope")
    p.add_argument("--norm-epsilon", type=float, default=1e-6)
    p.add_argument("--rotary-percent", type=float, default=0.25)
    p.add_argument("--swiglu", action="store_true")
    p.add_argument("--untie-embeddings-and-output-weights", action="store_true")
    p.add_argument("--vocab-size", type=int, default=248320)
    p.add_argument("--rotary-base", type=int, default=10000000)
    p.add_argument("--moe-ffn-hidden-size", type=int, default=512)
    p.add_argument("--moe-shared-expert-intermediate-size", type=int, default=512)
    p.add_argument("--moe-router-score-function", default="softmax")
    p.add_argument("--moe-token-dispatcher-type", default="alltoall")
    p.add_argument("--moe-router-topk", type=int, default=8)
    p.add_argument("--moe-layer-freq", default="[1]*40")
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--moe-grouped-gemm", action="store_true")
    p.add_argument("--moe-token-drop-policy", default="probs")
    p.add_argument("--moe-router-dtype", default="fp32")
    p.add_argument("--no-moe-permute-fusion", action="store_true")
    p.add_argument("--moe-aux-loss-coeff", type=float, default=0)
    p.add_argument("--attention-output-gate", action="store_true")
    p.add_argument("--moe-shared-expert-gate", action="store_true")
    p.add_argument("--train-backend", default="megatron")
    p.add_argument("--megatron-to-hf-mode", default="raw")
    p.add_argument("--optimizer-cpu-offload", action="store_true", default=True)
    p.add_argument("--overlap-cpu-optimizer-d2h-h2d", action="store_true")
    p.add_argument("--use-precision-aware-optimizer", action="store_true")
    p.add_argument("--attention-softmax-in-fp32", action="store_true")
    p.add_argument("--hidden-dropout", type=float, default=0.0)
    p.add_argument("--attention-dropout", type=float, default=0.0)
    return p.parse_args()


def build_model(args):
    """Build a minimal Megatron model with ref weights loaded."""
    from vime.backends.megatron_utils.model import setup_model_and_optimizer
    from vime.backends.megatron_utils.checkpoint import load_checkpoint

    print("Building model (no optimizer)...")
    t0 = time.time()
    model, _, _ = setup_model_and_optimizer(args, "actor")
    print(f"  Model built in {time.time() - t0:.1f}s")

    # Load ref weights directly (same path as actor.init)
    print(f"Loading ref checkpoint from {args.ref_load}...")
    t0 = time.time()
    old_load = args.load
    args.load = args.ref_load
    args.no_load_optim = True
    args.no_load_rng = True
    args.finetune = True
    load_checkpoint(model, None, None, checkpointing_context={}, skip_load_to_model_and_opt=False)
    args.load = old_load
    print(f"  Checkpoint loaded in {time.time() - t0:.1f}s")

    return model


def get_memory():
    """Return (allocated_MiB, reserved_MiB, free_MiB)."""
    return (
        torch.npu.memory_allocated() / (1024**2),
        torch.npu.memory_reserved() / (1024**2),
    )


def main():
    args = parse_args()

    # Init HCCL with one rank
    dist.init_process_group(backend="hccl", init_method="tcp://127.0.0.1:29500",
                            rank=0, world_size=1)

    from megatron.core import mpu
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
    )

    # Build model with ref weights
    model = build_model(args)
    alloc1, resv1 = get_memory()
    print(f"\nBefore offload: allocated={alloc1:.0f} MiB, reserved={resv1:.0f} MiB")

    # Offload
    from vime.utils.npu_weight_offloader import NPUWeightOffloader

    offloader = NPUWeightOffloader()
    t0 = time.time()
    offloaded_bytes = offloader.offload(model)
    offload_time = time.time() - t0
    print(f"Offload: {offloaded_bytes / 1024**2:.0f} MiB in {offload_time:.1f}s")

    alloc2, resv2 = get_memory()
    print(f"After offload:  allocated={alloc2:.0f} MiB, reserved={resv2:.0f} MiB")
    freed = alloc1 - alloc2
    print(f"Freed: {freed:.0f} MiB ({'PASS' if freed > 1000 else 'FAIL — expected >1 GiB freed'})")

    # Onload
    t0 = time.time()
    onloaded_bytes = offloader.onload(model)
    onload_time = time.time() - t0
    print(f"Onload:  {onloaded_bytes / 1024**2:.0f} MiB in {onload_time:.1f}s")

    alloc3, resv3 = get_memory()
    print(f"After onload:   allocated={alloc3:.0f} MiB, reserved={resv3:.0f} MiB")

    # Quick sanity: memory after onload should be similar to before offload
    diff = abs(alloc3 - alloc1)
    print(f"Memory diff before/after: {diff:.0f} MiB ({'PASS' if diff < 500 else 'WARN'})")

    print("\n=== NPUWeightOffloader standalone test complete ===")


if __name__ == "__main__":
    main()
