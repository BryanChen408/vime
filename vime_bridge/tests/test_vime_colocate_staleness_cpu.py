"""CPU-only tests for the synchronous (colocate) weight-boundary contract.

In colocate the engines sleep for the whole training step, so a polar session that is
in flight when training starts cannot make progress; it resumes afterwards under NEW
weights and produces a mixed-weight trajectory (v_N prefix + v_{N+1} suffix). The
polar-side version-span guard (POST /admin/policy_version) is what normally rejects
those continuations, but that endpoint is absent from the deployed gateway, so the
exclusion has to happen on the vime side -- via the staleness gate in drain_completed.

The derived bound cannot express it: max_off_policy_steps = max_async_level +
update_weights_interval and BOTH are forced > 0, so the floor is 2 while "spans exactly
one weight update" is staleness 1. Hence the explicit override tested here.

NON-INVASIVE: public API + dataclass construction only; the worker is never .start()ed.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import pytest

import vime_bridge.rollout as R
from vime_bridge.rollout import AsyncPolarRolloutWorker


class _DummyDataSource:
    def get_samples(self, n):
        return []


def _args(max_off_policy_steps=None, max_async_level=1, update_weights_interval=1):
    return SimpleNamespace(
        polar_url="http://polar:8080", polar_rollout_url=None, polar_run_id="run1",
        polar_reward_key="score", reward_key="score",
        polar_task_id_template="{args.polar_run_id}-op-{rollout_id}-{sample.group_index}",
        operator_tasks_dir="/tmp/op_tasks", polar_tasks_dir=None,
        rollout_scheduler_mode="session_pool", rollout_max_active_sessions=16,
        rollout_release_on_postrun=True, rollout_min_complete_accept_fraction=0.6,
        rollout_max_async_level=max_async_level, rollout_request_timeout=4000,
        rollout_batch_size=2, n_samples_per_prompt=4,
        update_weights_interval=update_weights_interval,
        rollout_max_off_policy_steps=max_off_policy_steps,
        hf_checkpoint="/tmp/hf", start_rollout_id=0,
    )


def _make_ready(worker, group_id, policy_version):
    """Park a completed group in the worker's ready map, as the async loop would."""
    completed = R._CompletedGroup(
        group_id=group_id, group=[], samples=[SimpleNamespace()],
        task_id=f"task-{group_id}", submitted_rollout_id=policy_version,
        policy_version=policy_version, session_count=4,
    )
    worker._ready_groups[group_id] = R._ReadyGroup(completed=completed)


# ------------------------------------------------------------------ config

def test_derived_bound_cannot_reach_zero():
    """Why the override exists: the floor of the derived value is 2, but a group that
    spans exactly one weight update has staleness 1 -- always inside the window."""
    w = AsyncPolarRolloutWorker(_args(max_async_level=1, update_weights_interval=1),
                                _DummyDataSource())
    assert w.config.max_off_policy_steps == 2


def test_override_zero_is_honoured_not_treated_as_unset():
    """0 is falsy; _first_configured must not confuse it with 'not configured'."""
    w = AsyncPolarRolloutWorker(_args(max_off_policy_steps=0), _DummyDataSource())
    assert w.config.max_off_policy_steps == 0


def test_override_rejects_negative():
    with pytest.raises(ValueError):
        AsyncPolarRolloutWorker(_args(max_off_policy_steps=-1), _DummyDataSource())


# ------------------------------------------------------------------ drain gate

def test_zero_bound_keeps_current_round_and_drops_the_carry_over():
    """The colocate contract: groups opened this round are accepted; anything left over
    from before the weight update is discarded."""
    w = AsyncPolarRolloutWorker(_args(max_off_policy_steps=0), _DummyDataSource())
    _make_ready(w, group_id=0, policy_version=5)   # opened this round  -> staleness 0
    _make_ready(w, group_id=1, policy_version=4)   # carried over       -> staleness 1

    accepted = w.drain_completed(max_groups=8, rollout_id=5)

    assert [g.group_id for g in accepted] == [0]
    assert w.snapshot_metrics().get("polar/dropped_stale_groups") == 1.0


def test_default_bound_would_have_accepted_the_carry_over():
    """Same input, derived bound: the mixed-weight group gets trained on. This is the
    behaviour the override exists to change -- pinned so a default change is visible."""
    w = AsyncPolarRolloutWorker(_args(), _DummyDataSource())
    _make_ready(w, group_id=0, policy_version=5)
    _make_ready(w, group_id=1, policy_version=4)

    accepted = w.drain_completed(max_groups=8, rollout_id=5)

    assert [g.group_id for g in accepted] == [0, 1]


def test_a_dropped_group_does_not_block_the_ones_behind_it():
    """A stale group must be skipped, not left blocking the queue behind it."""
    w = AsyncPolarRolloutWorker(_args(max_off_policy_steps=0), _DummyDataSource())
    _make_ready(w, group_id=0, policy_version=4)   # stale -> dropped
    _make_ready(w, group_id=1, policy_version=5)   # fresh -> must still be reachable

    accepted = w.drain_completed(max_groups=8, rollout_id=5)

    assert [g.group_id for g in accepted] == [1]
    assert w._ready_groups == {}


def test_a_missing_group_does_not_block_the_completed_ones():
    """Head-of-line blocking: draining is by completion order, not by group_id, so one
    group that never finishes cannot hold back groups that already have."""
    w = AsyncPolarRolloutWorker(_args(max_off_policy_steps=0), _DummyDataSource())
    # group 0 is the straggler that never completes -> never enters _ready_groups.
    for gid in (1, 2, 3):
        _make_ready(w, group_id=gid, policy_version=5)

    accepted = w.drain_completed(max_groups=8, rollout_id=5)

    assert [g.group_id for g in accepted] == [1, 2, 3]
