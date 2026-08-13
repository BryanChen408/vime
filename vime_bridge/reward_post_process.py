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

from vime_bridge import attempt_credit

logger = logging.getLogger(__name__)

# P1-a: GRPO std-normalization floor(slime 侧实战踩坑后加的,移植时带过):近全同组
# (std≈0.1 档差)会把微小 reward 差白化放大成巨优势 —— slime 实测「正确但慢的 0.82 在
# 全 1.0 组拿到 adv≈-0.94」,即因正确受罚。除数下限 0.05(约一个 reward 阶梯档)后,
# 小分差回到比例型信号;floor 之上的组间差异仍是满强度。
STD_FLOOR = 0.05


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
            vals = vals / torch.clamp(valid_vals.std(), min=STD_FLOOR) if len(valid_vals) > 1 else torch.zeros_like(vals)
        # Failed trajectories' loss_mask is already 0, but zeroing here keeps
        # their advantage out of any downstream stats/logging too.
        vals = vals * valid_mask.to(vals.dtype)
        for k, v in zip(keys, vals.tolist(), strict=True):
            normalized[k] = float(v)

    # P3 event-level credit: precompute the per-token attempt-advantage term and
    # stash it on train_metadata (consumed in vime compute_advantages_and_returns).
    # Bridge-only write; env-gated (POLAR_ATTEMPT_CREDIT); never changes the
    # returned trajectory rewards.
    if attempt_credit.enabled():
        _write_attempt_advantage(samples, key_by_sample, group_keys, traj_reward, traj_failed)

    rewards = [normalized[k] for k in key_by_sample]
    return raw_rewards, rewards


def _write_attempt_advantage(
    samples: list[Any],
    key_by_sample: list[tuple[int, int]],
    group_keys: dict[int, list[tuple[int, int]]],
    traj_reward: dict[tuple[int, int], float],
    traj_failed: dict[tuple[int, int], bool],
) -> None:
    """Compute the per-token attempt-advantage term and stash on train_metadata.
    Never raises into the reward path."""
    try:
        # Per-group reward std (same quantity A_traj is normalized by), for scale
        # consistency of the additive term.
        group_std: dict[int, float] = {}
        for keys in group_keys.values():
            if not keys:
                continue
            g = keys[0][0]
            valid = [traj_reward[k] for k in keys if not traj_failed.get(k)]
            if len(valid) > 1:
                group_std[g] = float(torch.tensor(valid, dtype=torch.float32).std())
            else:
                group_std[g] = 1.0
        response_len = [
            len(s.loss_mask) if getattr(s, "loss_mask", None) is not None else 0 for s in samples
        ]
        terms = attempt_credit.build_batch(
            samples, key_by_sample, group_keys, group_std, response_len
        )
        for i, sample in enumerate(samples):
            if terms[i] is None:
                continue
            if getattr(sample, "train_metadata", None) is None:
                sample.train_metadata = {}
            sample.train_metadata["attempt_advantage"] = terms[i]
    except Exception:  # best-effort; never break training
        logger.exception("attempt-advantage precompute failed")


def _is_failed_trajectory(sample: Any) -> bool:
    """True if the sample's status marks it as agent ERROR or TIMEOUT."""
    status = getattr(sample, "status", None)
    name = getattr(status, "name", None) or str(status).rsplit(".", 1)[-1]
    return name.upper() in ("FAILED", "ABORTED")
