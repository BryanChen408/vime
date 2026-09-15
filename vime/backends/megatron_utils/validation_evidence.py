from __future__ import annotations

import hashlib
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
from megatron.core import mpu

from .cp_utils import slice_log_prob_with_cp


SCHEMA_VERSION = 1


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().contiguous()


def _cpu_tensor_list(values: list[torch.Tensor]) -> list[torch.Tensor]:
    return [_cpu_tensor(value) for value in values]


def _token_digest(tokens: torch.Tensor) -> str:
    payload = _cpu_tensor(tokens).to(dtype=torch.int64).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _parallel_value(name: str, default: int = 0) -> int:
    getter = getattr(mpu, name, None)
    return int(getter()) if getter is not None else default


def _packed_seq_snapshot(packed_seq_params: Any) -> dict[str, torch.Tensor | None]:
    if packed_seq_params is None:
        return {}

    snapshot = {}
    for name in (
        "cu_seqlens_q",
        "cu_seqlens_kv",
        "cu_seqlens_q_padded",
        "cu_seqlens_kv_padded",
        "cu_seqlens_gdn",
    ):
        value = getattr(packed_seq_params, name, None)
        snapshot[name] = _cpu_tensor(value) if isinstance(value, torch.Tensor) else None
    return snapshot


def _yarn_fingerprint(args: Namespace) -> dict[str, Any]:
    return {
        "position_embedding_type": getattr(args, "position_embedding_type", None),
        "rotary_base": getattr(args, "rotary_base", None),
        "rotary_percent": getattr(args, "rotary_percent", None),
        "scaling_factor": getattr(args, "rotary_scaling_factor", None),
        "original_max_position_embeddings": getattr(args, "yarn_original_max_position_embeddings", None),
        "beta_fast": getattr(args, "yarn_beta_fast", None),
        "beta_slow": getattr(args, "yarn_beta_slow", None),
        "mscale": getattr(args, "mscale", None),
        "mscale_all_dim": getattr(args, "mscale_all_dim", None),
        "correction_range_round_to_int": getattr(args, "yarn_correction_range_round_to_int", None),
        "seq_length": getattr(args, "seq_length", None),
        "max_position_embeddings": getattr(args, "max_position_embeddings", None),
    }


def save_tis_validation_evidence(
    args: Namespace,
    batch: dict[str, Any],
    *,
    train_log_probs: list[torch.Tensor],
    current_log_probs: list[torch.Tensor],
) -> Path | None:
    """Persist response-aligned tensors used to judge train/rollout consistency."""

    output_dir = os.environ.get("VIME_SAVE_TIS_LOGPROBS", "")
    if not output_dir or "rollout_log_probs" not in batch:
        return None
    if _parallel_value("get_tensor_model_parallel_rank") != 0 or not mpu.is_pipeline_last_stage():
        return None

    total_lengths = [int(value) for value in batch["total_lengths"]]
    response_lengths = [int(value) for value in batch["response_lengths"]]
    max_seq_lens = batch.get("max_seq_lens")
    qkv_format = getattr(args, "qkv_format", "thd")

    local_masks = []
    local_positions = []
    local_target_token_ids = []
    for index, (tokens, loss_mask, total_length, response_length) in enumerate(
        zip(
            batch["unconcat_tokens"],
            batch["loss_masks"],
            total_lengths,
            response_lengths,
            strict=True,
        )
    ):
        max_seq_len = max_seq_lens[index] if max_seq_lens is not None else None
        prompt_length = total_length - response_length
        positions = torch.arange(prompt_length - 1, total_length - 1, device=loss_mask.device)
        response_tokens = tokens[-response_length:] if response_length else tokens[:0]
        local_masks.append(
            slice_log_prob_with_cp(loss_mask, total_length, response_length, qkv_format, max_seq_len)
        )
        local_positions.append(
            slice_log_prob_with_cp(positions, total_length, response_length, qkv_format, max_seq_len)
        )
        local_target_token_ids.append(
            slice_log_prob_with_cp(response_tokens, total_length, response_length, qkv_format, max_seq_len)
        )

    invocation = int(getattr(args, "_tis_validation_evidence_index", 0))
    setattr(args, "_tis_validation_evidence_index", invocation + 1)
    global_rank = int(torch.distributed.get_rank())
    step = int(getattr(args, "curr_iteration", 0))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / f"tis_evidence_step{step}_mb{invocation}_rank{global_rank}.pt"

    payload = {
        "schema_version": SCHEMA_VERSION,
        "step": step,
        "microbatch_index": invocation,
        "global_rank": global_rank,
        "tp_rank": _parallel_value("get_tensor_model_parallel_rank"),
        "tp_size": _parallel_value("get_tensor_model_parallel_world_size", 1),
        "cp_rank": _parallel_value("get_context_parallel_rank"),
        "cp_size": _parallel_value("get_context_parallel_world_size", 1),
        "dp_rank": _parallel_value("get_data_parallel_rank"),
        "dp_size": _parallel_value("get_data_parallel_world_size", 1),
        "sample_indices": list(batch.get("sample_indices") or range(len(total_lengths))),
        "rollout_ids": list(batch.get("rollout_ids") or range(len(total_lengths))),
        "total_lengths": total_lengths,
        "response_lengths": response_lengths,
        "sample_token_sha256": [_token_digest(tokens) for tokens in batch["unconcat_tokens"]],
        "loss_masks": _cpu_tensor_list(batch["loss_masks"]),
        "local_loss_masks": _cpu_tensor_list(local_masks),
        "local_logit_positions": _cpu_tensor_list(local_positions),
        "local_target_token_ids": _cpu_tensor_list(local_target_token_ids),
        "train_log_probs": _cpu_tensor_list(train_log_probs),
        "current_log_probs": _cpu_tensor_list(current_log_probs),
        "rollout_log_probs": _cpu_tensor_list(batch["rollout_log_probs"]),
        "packed_seq_params": _packed_seq_snapshot(batch.get("packed_seq_params")),
        "parallel_config": {
            "tensor_model_parallel_size": getattr(args, "tensor_model_parallel_size", None),
            "pipeline_model_parallel_size": getattr(args, "pipeline_model_parallel_size", None),
            "context_parallel_size": getattr(args, "context_parallel_size", None),
            "expert_model_parallel_size": getattr(args, "expert_model_parallel_size", None),
        },
        "yarn_fingerprint": _yarn_fingerprint(args),
    }

    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path
