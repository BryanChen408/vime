#!/usr/bin/env python3
"""Evaluation-only torch_dist -> HF converter.

Some checkpoints store GDN projections as split ShardedTensor entries.  The
general converter expects the pre-split fused entries, so normalize those
entries here before delegating to the existing converter implementation.
"""

import argparse
import os
import time

import torch
import torch.distributed.checkpoint as dist_cp
from transformers import AutoConfig

import convert_patch

base = convert_patch.base


_IN_PROJ_PARTS = ("query", "key", "value", "z", "beta", "alpha")
_CONV_PARTS = ("query", "key", "value")


def _merge_split_entries(state_dict, tp_size=1):
    """Merge split GDN sharded entries back to the names expected by vime."""
    result = dict(state_dict)
    merged_bases = set()
    for name in list(state_dict):
        for parts in (_IN_PROJ_PARTS, _CONV_PARTS):
            suffix = "." + parts[0]
            if not name.endswith(suffix):
                continue
            base_name = name[: -len(suffix)]
            if not (base_name.endswith("in_proj.weight") or base_name.endswith("conv1d.weight")):
                continue
            keys = [base_name + "." + part for part in parts]
            if not all(key in state_dict for key in keys):
                continue
            if base_name.endswith("in_proj.weight") and parts is not _IN_PROJ_PARTS:
                continue
            if base_name.endswith("conv1d.weight") and parts is not _CONV_PARTS:
                continue
            tensors = [state_dict[key] for key in keys]
            rest = tensors[0].shape[1:]
            interleaved = [t.reshape(tp_size, -1, *rest) for t in tensors]
            result[base_name] = torch.cat(interleaved, dim=1).reshape(-1, *rest).contiguous()
            for key in keys:
                result.pop(key, None)
            merged_bases.add(base_name)
            break
    split_names = [
        name
        for name in result
        if any(name.endswith("." + part) for part in _IN_PROJ_PARTS + _CONV_PARTS)
        and ("in_proj.weight." in name or "conv1d.weight." in name)
    ]
    if split_names:
        raise ValueError(f"Incomplete split GDN parameter group(s): {split_names[:5]}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--origin-hf-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=5 * 1024**3)
    parser.add_argument("--vocab-size", type=int, default=None)
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        raise ValueError(f"Output directory {args.output_dir} already exists")
    if not os.path.isfile(os.path.join(args.input_dir, "common.pt")):
        raise ValueError(f"Not a torch_dist checkpoint: {args.input_dir}")

    hf_config = AutoConfig.from_pretrained(args.origin_hf_dir, trust_remote_code=True)
    megatron_args = torch.load(os.path.join(args.input_dir, "common.pt"), weights_only=False)["args"]
    state_dict = {}
    print(f"loading model from {args.input_dir}")
    started = time.time()
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=base.WrappedStorageReader(args.input_dir),
        planner=base.EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    print(f"model loaded in {time.time() - started:.2f} sec.")
    state_dict = _merge_split_entries(state_dict, tp_size=megatron_args.tensor_model_parallel_size)
    base.save_tensors(
        megatron_args,
        type(hf_config).__name__.lower(),
        state_dict,
        args.output_dir,
        args.chunk_size,
        args.vocab_size,
        args.origin_hf_dir,
    )
    base.copy_assets(args.origin_hf_dir, args.output_dir)


if __name__ == "__main__":
    main()
