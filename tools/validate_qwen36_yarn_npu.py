#!/usr/bin/env python3
"""Validate Megatron YaRN numerics on one Ascend NPU at 300K positions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest import mock

import torch
import torch_npu  # noqa: F401 - registers the NPU backend
from transformers import PreTrainedConfig
from transformers.modeling_rope_utils import _compute_yarn_parameters

from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding,
)


POSITIONS = [0, 262_143, 262_144, 270_335, 299_999]


def _stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    errors = (actual.float().cpu() - expected.float().cpu()).abs().flatten()
    return {
        "max_abs_error": float(errors.max().item()),
        "mean_abs_error": float(errors.mean().item()),
        "p95_abs_error": float(torch.quantile(errors, 0.95).item()),
    }


def _position_stats(actual: torch.Tensor, expected: torch.Tensor) -> list[dict[str, float | int]]:
    return [
        {"position": position, **_stats(actual[index], expected[index])}
        for index, position in enumerate(POSITIONS)
    ]


def _oracle() -> tuple[torch.Tensor, float]:
    config = PreTrainedConfig(
        hidden_size=4096,
        num_attention_heads=16,
        head_dim=256,
        max_position_embeddings=262_144,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 10_000_000.0,
            "partial_rotary_factor": 0.25,
            "factor": 4.0,
            "original_max_position_embeddings": 262_144,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 0.0,
            "truncate": True,
        },
    )
    inv_freq, attention_scale = _compute_yarn_parameters(
        config, device=torch.device("cpu")
    )
    return inv_freq, float(attention_scale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.npu.set_device(device)
    inv_freq, attention_scale = _oracle()
    position_values = torch.tensor(POSITIONS, dtype=torch.float32)
    expected_freqs = torch.outer(position_values, inv_freq)
    expected_embeddings = torch.cat((expected_freqs, expected_freqs), dim=-1)[:, None, None]
    expected_cos = expected_embeddings.cos() * attention_scale
    expected_sin = expected_embeddings.sin() * attention_scale

    with mock.patch.object(torch.cuda, "current_device", return_value=device):
        rope = YarnRotaryEmbedding(
            kv_channels=256,
            rotary_percent=0.25,
            rotary_base=10_000_000,
            scaling_factor=4.0,
            original_max_position_embeddings=262_144,
            beta_fast=32.0,
            beta_slow=1.0,
            mscale=1.0,
            mscale_all_dim=0.0,
            correction_range_round_to_int=True,
            use_cpu_initialization=False,
        )
        npu_embeddings = torch.cat(
            [rope.get_emb(1, offset=position)[0] for position in POSITIONS], dim=0
        )
        standard_rope = RotaryEmbedding(
            kv_channels=256,
            rotary_percent=0.25,
            rotary_base=10_000_000,
            use_cpu_initialization=False,
        )
        standard_npu_embeddings = torch.cat(
            [standard_rope.get_emb(1, offset=position) for position in POSITIONS], dim=0
        )

    actual_cos = npu_embeddings.cos() * attention_scale
    actual_sin = npu_embeddings.sin() * attention_scale
    cos_stats = _stats(actual_cos, expected_cos)
    sin_stats = _stats(actual_sin, expected_sin)
    phase_stats = _stats(npu_embeddings, expected_embeddings)

    standard_inv_freq = 1.0 / (
        10_000_000
        ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64)
    )
    standard_freqs = torch.outer(position_values, standard_inv_freq)
    standard_expected_embeddings = torch.cat(
        (standard_freqs, standard_freqs), dim=-1
    )[:, None, None]
    standard_cos_stats = _stats(
        standard_npu_embeddings.cos(), standard_expected_embeddings.cos()
    )
    standard_sin_stats = _stats(
        standard_npu_embeddings.sin(), standard_expected_embeddings.sin()
    )
    standard_phase_stats = _stats(
        standard_npu_embeddings, standard_expected_embeddings
    )

    generator = torch.Generator().manual_seed(20260915)
    values_cpu = torch.randn(5, 1, 2, 256, generator=generator).to(torch.bfloat16)
    actual_rotated = _apply_rotary_pos_emb_bshd(
        values_cpu.to(device), npu_embeddings, mscale=attention_scale
    ).cpu()
    expected_rotated = _apply_rotary_pos_emb_bshd(
        values_cpu, expected_embeddings, mscale=attention_scale
    )
    qk_stats = _stats(actual_rotated, expected_rotated)
    tail_exact = bool(torch.equal(actual_rotated[..., 64:], values_cpu[..., 64:]))
    standard_actual_rotated = _apply_rotary_pos_emb_bshd(
        values_cpu.to(device), standard_npu_embeddings
    ).cpu()
    standard_expected_rotated = _apply_rotary_pos_emb_bshd(
        values_cpu, standard_expected_embeddings
    )
    standard_qk_stats = _stats(standard_actual_rotated, standard_expected_rotated)

    cos_baseline_limit = standard_cos_stats["max_abs_error"] * attention_scale + 1e-6
    sin_baseline_limit = standard_sin_stats["max_abs_error"] * attention_scale + 1e-6
    passed = bool(
        cos_stats["max_abs_error"] <= cos_baseline_limit
        and sin_stats["max_abs_error"] <= sin_baseline_limit
        and qk_stats["max_abs_error"] <= 5e-2
        and tail_exact
    )
    result = {
        "schema_version": 1,
        "device": args.device,
        "dtype": "bfloat16",
        "positions": POSITIONS,
        "original_max_position_embeddings": 262_144,
        "maximum_validated_position": max(POSITIONS),
        "attention_scale": attention_scale,
        "cos": cos_stats,
        "cos_by_position": _position_stats(actual_cos, expected_cos),
        "sin": sin_stats,
        "sin_by_position": _position_stats(actual_sin, expected_sin),
        "phase": phase_stats,
        "qk_apply": qk_stats,
        "non_rotary_tail_exact": tail_exact,
        "standard_rope_npu_baseline": {
            "cos": standard_cos_stats,
            "sin": standard_sin_stats,
            "phase": standard_phase_stats,
            "qk_apply": standard_qk_stats,
        },
        "thresholds": {
            "cos_max_abs_error_from_standard_rope_baseline_times_mscale": cos_baseline_limit,
            "sin_max_abs_error_from_standard_rope_baseline_times_mscale": sin_baseline_limit,
            "bf16_qk_max_abs_error": 5e-2,
        },
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
