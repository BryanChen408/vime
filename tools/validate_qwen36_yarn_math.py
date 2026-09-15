#!/usr/bin/env python3
"""Compare Qwen3.6 YaRN implementations with the Transformers oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from transformers import PreTrainedConfig
from transformers.modeling_rope_utils import _compute_yarn_parameters

from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding,
)
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import (
    YaRNScalingRotaryEmbedding,
)


POSITIONS = [0, 262_143, 262_144, 270_335, 299_999]


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


def _reference_cache(
    inv_freq: torch.Tensor, attention_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.tensor(POSITIONS, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos() * attention_scale, freqs.sin() * attention_scale


def _rotate_reference(
    values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int
) -> torch.Tensor:
    rotary = values[..., :rotary_dim]
    tail = values[..., rotary_dim:]
    left, right = rotary.chunk(2, dim=-1)
    rotated = torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)
    return torch.cat((rotated, tail), dim=-1)


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def _validate_megatron(
    inv_freq: torch.Tensor, attention_scale: float, cos: torch.Tensor, sin: torch.Tensor
) -> dict[str, float | bool]:
    with (
        mock.patch.object(torch.cuda, "current_device", return_value=torch.device("cpu")),
        mock.patch.object(YarnRotaryEmbedding, "_set_cos_sin_cache", return_value=None),
    ):
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
            use_cpu_initialization=True,
        )
        rows = []
        for position in POSITIONS:
            emb, scale = rope.get_emb(1, offset=position)
            rows.append(emb[0, 0, 0])
            if float(scale) != attention_scale:
                raise AssertionError(f"Megatron attention scale mismatch at {position}")
        embeddings = torch.stack(rows)

    expected_emb = torch.cat(
        (
            torch.outer(torch.tensor(POSITIONS, dtype=torch.float32), inv_freq),
            torch.outer(torch.tensor(POSITIONS, dtype=torch.float32), inv_freq),
        ),
        dim=-1,
    )
    cos_error = _max_abs(embeddings.cos() * attention_scale, torch.cat((cos, cos), dim=-1))
    sin_error = _max_abs(embeddings.sin() * attention_scale, torch.cat((sin, sin), dim=-1))
    return {
        "cos_max_abs_error": cos_error,
        "sin_max_abs_error": sin_error,
        "passed": cos_error <= 1e-6 and sin_error <= 1e-6,
    }


def _validate_vllm(
    inv_freq: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> dict[str, float | int | bool]:
    with set_current_vllm_config(VllmConfig()):
        rope = MRotaryEmbedding(
            head_size=256,
            rotary_dim=64,
            max_position_embeddings=262_144,
            base=10_000_000.0,
            is_neox_style=True,
            dtype=torch.float32,
            mrope_section=[11, 11, 10],
            mrope_interleaved=True,
            scaling_factor=4.0,
            beta_fast=32,
            beta_slow=1,
            truncate=True,
        )

    expected_inv_freq = YaRNScalingRotaryEmbedding._compute_inv_freq(
        SimpleNamespace(
            base=rope.base,
            rotary_dim=rope.rotary_dim,
            beta_fast=rope.beta_fast,
            beta_slow=rope.beta_slow,
            max_position_embeddings=rope.original_max_position_embeddings,
            truncate=rope.truncate,
            extrapolation_factor=rope.extrapolation_factor,
        ),
        rope.scaling_factor,
    )
    actual_inv_freq = rope._compute_inv_freq(rope.scaling_factor)
    inv_freq_error = _max_abs(actual_inv_freq, expected_inv_freq)
    oracle_inv_freq_error = _max_abs(actual_inv_freq, inv_freq)

    positions = torch.tensor(POSITIONS, dtype=torch.long)
    actual_cos, actual_sin = rope.cos_sin_cache[positions].chunk(2, dim=-1)
    cos_error = _max_abs(actual_cos, cos)
    sin_error = _max_abs(actual_sin, sin)

    generator = torch.Generator().manual_seed(20260915)
    query = torch.randn(len(POSITIONS), 256, generator=generator)
    key = torch.randn(len(POSITIONS), 256, generator=generator)
    actual_query, actual_key = rope.forward_native(positions, query.clone(), key.clone())
    expected_query = _rotate_reference(query, cos, sin, rope.rotary_dim)
    expected_key = _rotate_reference(key, cos, sin, rope.rotary_dim)
    query_error = _max_abs(actual_query, expected_query)
    key_error = _max_abs(actual_key, expected_key)

    maximum_error = max(
        inv_freq_error,
        oracle_inv_freq_error,
        cos_error,
        sin_error,
        query_error,
        key_error,
    )
    return {
        "cache_max_position_num": rope.cache_max_position_num,
        "inv_freq_max_abs_error": inv_freq_error,
        "oracle_inv_freq_max_abs_error": oracle_inv_freq_error,
        "cos_max_abs_error": cos_error,
        "sin_max_abs_error": sin_error,
        "query_max_abs_error": query_error,
        "key_max_abs_error": key_error,
        "passed": maximum_error <= 1e-6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inv_freq, attention_scale = _oracle()
    cos, sin = _reference_cache(inv_freq, attention_scale)
    megatron = _validate_megatron(inv_freq, attention_scale, cos, sin)
    vllm = _validate_vllm(inv_freq, cos, sin)
    result = {
        "schema_version": 1,
        "positions": POSITIONS,
        "original_max_position_embeddings": 262_144,
        "maximum_validated_position": max(POSITIONS),
        "attention_scale": attention_scale,
        "megatron": megatron,
        "vllm_023": vllm,
        "passed": bool(megatron["passed"] and vllm["passed"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
