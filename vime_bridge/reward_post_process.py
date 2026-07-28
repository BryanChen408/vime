"""Trajectory-aware reward post-processor for Slime.

Registered via Slime's ``--custom-reward-post-process-path`` hook.  Treats
every sample whose ``(group_index, index)`` pair matches as belonging to
the same trajectory and keeps exactly one reward per trajectory, then
normalizes across trajectories inside each group.  The normalized value is
broadcast back to every sample of the trajectory.

Adapter contract:
    All Slime samples produced from the same Polar ``SessionResult`` share
    the same ``Sample.index`` (the trajectory's position within the group).
    Samples from different sessions in the same group get distinct indices.

FAILED/ABORTED trajectories (agent ERROR or TIMEOUT) are excluded from the
group baseline: they don't contribute to mean/std, and their own normalized
reward is forced to 0.  This prevents agent-side failures — which are
semantically "no outcome" rather than "outcome = 0" — from biasing the
advantage of the surviving samples in the group.

Degenerate case (one trace per trajectory) collapses to plain GRPO
group-normalization — no special-casing needed.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def post_process_rewards(
    args: Any,
    samples: list[Any],
) -> tuple[list[float], list[float]]:
    """Slime reward-post-process hook. Returns (raw_rewards, rewards)."""
    raw_rewards = [float(sample.get_reward_value(args)) for sample in samples]

    if not getattr(args, "rewards_normalization", True):
        return raw_rewards, list(raw_rewards)

    estimator = getattr(args, "advantage_estimator", None)
    if estimator not in ("grpo", "gspo", "reinforce_plus_plus_baseline"):
        return raw_rewards, list(raw_rewards)

    std_norm = estimator in ("grpo", "gspo") and bool(
        getattr(args, "grpo_std_normalization", False)
    )

    # Key each sample by its trajectory; first-seen reward per trajectory wins.
    # A trajectory is marked failed if *any* of its traces has FAILED/ABORTED
    # status (in practice all traces share the session status, but be safe).
    traj_reward: dict[tuple[int, int], float] = {}
    traj_failed: dict[tuple[int, int], bool] = {}
    group_keys: dict[int, list[tuple[int, int]]] = {}
    key_by_sample: list[tuple[int, int]] = []
    for i, sample in enumerate(samples):
        group_idx = int(sample.group_index) if sample.group_index is not None else -1
        traj_idx = int(sample.index) if sample.index is not None else i
        key = (group_idx, traj_idx)
        key_by_sample.append(key)
        failed = _is_failed_trajectory(sample)
        if key not in traj_reward:
            traj_reward[key] = raw_rewards[i]
            traj_failed[key] = failed
            group_keys.setdefault(group_idx, []).append(key)
        elif failed:
            traj_failed[key] = True

    normalized: dict[tuple[int, int], float] = {}
    for keys in group_keys.values():
        valid_mask = torch.tensor([not traj_failed[k] for k in keys], dtype=torch.bool)
        if not bool(valid_mask.any()):
            # All trajectories in this group failed — no signal available.
            for k in keys:
                normalized[k] = 0.0
            continue
        vals = torch.tensor([traj_reward[k] for k in keys], dtype=torch.float32)
        valid_vals = vals[valid_mask]
        vals = vals - valid_vals.mean()
        if std_norm:
            vals = vals / (valid_vals.std() + 1e-6) if len(valid_vals) > 1 else torch.zeros_like(vals)
        # Failed trajectories' loss_mask is already 0, but zeroing here keeps
        # their advantage out of any downstream stats/logging too.
        vals = vals * valid_mask.to(vals.dtype)
        for k, v in zip(keys, vals.tolist(), strict=True):
            normalized[k] = float(v)

    rewards = [normalized[k] for k in key_by_sample]
    return raw_rewards, rewards


def _is_failed_trajectory(sample: Any) -> bool:
    """True if the sample's status marks it as agent ERROR or TIMEOUT."""
    status = getattr(sample, "status", None)
    name = getattr(status, "name", None) or str(status).rsplit(".", 1)[-1]
    return name.upper() in ("FAILED", "ABORTED")
