"""Token-weighted train/behaviour probability diagnostics (verl's definition)."""

import math

import torch


STAT_KEYS = tuple(f"_rollout_probs_diff_{name}" for name in ("count", "sum", "sum_sq", "max"))


def probability_diff_stats(train_log_probs, rollout_log_probs, mask):
    """Return sufficient statistics; masks include actions from every policy epoch."""
    if train_log_probs.shape != rollout_log_probs.shape or mask.shape != train_log_probs.shape:
        raise ValueError("train/rollout logprobs and action mask must have identical shapes")
    with torch.no_grad():
        valid = mask.bool()
        diff = (train_log_probs.float()[valid].exp() - rollout_log_probs.float()[valid].exp()).abs()
        if not torch.isfinite(diff).all():
            raise ValueError("non-finite train/rollout probabilities on trainable tokens")
        zero = diff.new_zeros(())
        return dict(zip(STAT_KEYS, (
            diff.new_tensor(diff.numel()), diff.sum(), diff.square().sum(),
            diff.max() if diff.numel() else zero,
        ), strict=True))


def probability_diff_metrics(count, total, total_sq, maximum):
    """Finalize globally reduced statistics. std uses correction=1, as in verl."""
    mean = total / count if count else math.nan
    variance = max(0.0, (total_sq - total * mean) / (count - 1)) if count > 1 else math.nan
    return {
        "rollout_probs_diff_mean": mean,
        "rollout_probs_diff_max": maximum if count else math.nan,
        "rollout_probs_diff_std": math.sqrt(variance),
        "rollout_probs_diff_count": count,
    }
