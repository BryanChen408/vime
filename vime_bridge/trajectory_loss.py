"""Trajectory-aware policy-gradient loss reducers for Polar traces."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
import logging
from typing import Any

import torch
from megatron.core import mpu

logger = logging.getLogger(__name__)
_LOGGED_REDUCER_EVENTS: set[str] = set()


def get_trajectory_pg_loss_reducer(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    calculate_per_token_loss: bool = False,
    *,
    trajectory_keys: Sequence[Any] | None = None,
    trajectory_trace_counts: Sequence[int | float | None] | None = None,
    trajectory_loss_scale: int | float | None = None,
    batch: dict[str, Any] | None = None,
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
    **_: Any,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a reducer that gives every Polar trajectory equal PG weight.

    Slime's default reducer computes ``sum(trace_mean_loss)`` over flat samples.
    Polar sessions can expand into multiple trace samples, so that gives sessions
    with more traces more weight.

    The training-side outer loss scale still divides by the flat dynamic global
    batch size. To make the final scaled loss equal to:

    ``mean_trajectory(mean_trace(masked_token_loss))``

    this reducer weights each trace by
    ``flat_sample_count / trajectory_count / kept_trace_count``. The rollout
    conversion computes those counts after overlength/drop filtering, so the
    result is independent of micro-batch boundaries.
    """

    if calculate_per_token_loss:
        # Keep Slime's per-token mode semantics: return token-sum loss. Polar E2E
        # uses sample-mean mode, where trajectory weighting matters.
        return _token_sum_reducer(
            total_lengths,
            response_lengths,
            loss_masks,
            batch=batch,
            qkv_format=qkv_format,
            max_seq_lens=max_seq_lens,
        )

    if trajectory_keys is None:
        trajectory_keys = _fallback_trajectory_keys(batch, len(response_lengths))
    if trajectory_trace_counts is None and batch:
        trajectory_trace_counts = batch.get("trajectory_trace_counts")
    if trajectory_loss_scale is None and batch:
        trajectory_loss_scale = batch.get("trajectory_loss_scale")
    if trajectory_keys is None or len(trajectory_keys) != len(response_lengths):
        _reject_incomplete_batch(
            "trajectory keys",
            f"got {_safe_len(trajectory_keys)} for {len(response_lengths)} samples",
        )
    if trajectory_trace_counts is None or len(trajectory_trace_counts) != len(response_lengths):
        _reject_incomplete_batch(
            "trace counts",
            f"got {_safe_len(trajectory_trace_counts)} for {len(response_lengths)} samples",
        )

    split_lengths, local_masks = _local_response_masks(
        total_lengths,
        response_lengths,
        loss_masks,
        batch=batch,
        qkv_format=qkv_format,
            max_seq_lens=max_seq_lens,
    )
    scale = _positive_float(trajectory_loss_scale, default=1.0)
    trace_denominators = [_positive_float(value, default=1.0) for value in trajectory_trace_counts]
    _log_once(
        "weighted",
        "Using trajectory-weighted PG reducer: samples=%d trajectories=%d "
        "trajectory_loss_scale=%s trace_counts=%s",
        len(response_lengths),
        len({_normalize_key(key, idx) for idx, key in enumerate(trajectory_keys)}),
        scale,
        list(trajectory_trace_counts),
    )

    def reduce_pg_loss(pg_loss: torch.Tensor) -> torch.Tensor:
        total = pg_loss.new_zeros(())
        for x_i, local_mask, global_mask, trace_count in zip(
            pg_loss.split(split_lengths, dim=0),
            local_masks,
            loss_masks,
            trace_denominators,
            strict=False,
        ):
            trace_loss = (x_i * local_mask).sum() / torch.clamp_min(global_mask.sum(), 1)
            total = total + trace_loss * (scale / trace_count)
        return total

    return reduce_pg_loss


def _safe_len(value: Any) -> int | None:
    try:
        return len(value) if value is not None else None
    except TypeError:
        return None


def _reject_incomplete_batch(kind: str, reason: str) -> None:
    """Refuse to weight trajectories from a batch that does not carry what it takes.

    Falling back to a different weighting would still train, quietly optimising a different
    objective than the one asked for, and policy-gradient losses are sensitive to exactly that.
    """
    raise ValueError(
        f"Trajectory-weighted policy gradient needs {kind} data this batch does not have: {reason}"
    )


def _log_once(key: str, message: str, *args: Any) -> None:
    if key in _LOGGED_REDUCER_EVENTS:
        return
    _LOGGED_REDUCER_EVENTS.add(key)
    logger.info(message, *args)


def _fallback_trajectory_keys(batch: dict[str, Any] | None, n: int) -> list[Any] | None:
    if not batch:
        return None
    group_indices = batch.get("group_indices")
    sample_indices = batch.get("sample_indices")
    if group_indices is None or sample_indices is None:
        return None
    if len(group_indices) != n or len(sample_indices) != n:
        return None
    return [[g, s] for g, s in zip(group_indices, sample_indices, strict=True)]


def _normalize_key(key: Any, idx: int) -> tuple[Any, ...]:
    if key is None:
        return ("sample", idx)
    if isinstance(key, (list, tuple)):
        return tuple(key)
    return (key,)


def _positive_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _local_response_masks(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    *,
    batch: dict[str, Any] | None,
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
) -> tuple[list[int], list[torch.Tensor]]:
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size == 1:
        return list(response_lengths), list(loss_masks)

    if max_seq_lens is None and batch:
        max_seq_lens = batch.get("max_seq_lens")
    from vime.backends.megatron_utils.cp_utils import get_logits_and_tokens_offset_with_cp

    split_lengths: list[int] = []
    local_masks: list[torch.Tensor] = []
    for i, (total_length, response_length, loss_mask) in enumerate(
        zip(total_lengths, response_lengths, loss_masks, strict=False)
    ):
        prompt_length = total_length - response_length
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
        _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
            total_length,
            response_length,
            qkv_format,
            max_seq_len,
        )
        parts = []
        for start, end in token_offsets:
            res_start = max(0, start - prompt_length)
            res_end = max(0, end - prompt_length)
            if res_end > res_start:
                parts.append(loss_mask[res_start:res_end])
        if parts:
            local = torch.cat(parts, dim=0)
        else:
            local = loss_mask.new_zeros((0,))
        local_masks.append(local)
        split_lengths.append(local.size(0))
    return split_lengths, local_masks


def _token_sum_reducer(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    *,
    batch: dict[str, Any] | None,
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    split_lengths, local_masks = _local_response_masks(
        total_lengths,
        response_lengths,
        loss_masks,
        batch=batch,
        qkv_format=qkv_format,
        max_seq_lens=max_seq_lens,
    )

    def reduce_pg_loss(pg_loss: torch.Tensor) -> torch.Tensor:
        return sum(
            (x_i * local_mask).sum()
            for x_i, local_mask in zip(
                pg_loss.split(split_lengths, dim=0),
                local_masks,
                strict=False,
            )
        )

    return reduce_pg_loss
