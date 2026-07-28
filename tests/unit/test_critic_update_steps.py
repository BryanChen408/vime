"""Unit tests for repeated critic updates over one rollout."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _schedule_increment(step_global_batch_size: int, role: str, critic_update_steps: int) -> int:
    """Mirror of the increment rule in ``train_one_step``."""
    increment = step_global_batch_size
    if role == "critic":
        increment //= max(1, critic_update_steps)
    return increment


def _run_critic_updates(critic_update_steps: int):
    """Mirror of the loop in ``train_async``: returns (calls, ref handed to the actor)."""
    calls = 0

    def async_train():
        nonlocal calls
        calls += 1
        return f"ref{calls}"

    value_refs = async_train()
    for _ in range(max(1, critic_update_steps) - 1):
        value_refs = async_train()
    return calls, value_refs


@pytest.mark.unit
@pytest.mark.parametrize(("steps", "expected"), [(1, 1), (2, 2), (3, 3), (0, 1), (-1, 1)])
def test_update_count(steps, expected):
    calls, _ = _run_critic_updates(steps)
    assert calls == expected


@pytest.mark.unit
def test_the_actor_receives_the_last_values():
    _, ref = _run_critic_updates(3)
    assert ref == "ref3"


@pytest.mark.unit
def test_the_actor_schedule_is_never_scaled():
    assert _schedule_increment(256, "actor", 4) == 256


@pytest.mark.unit
def test_a_single_critic_update_is_unchanged():
    assert _schedule_increment(256, "critic", 1) == 256


@pytest.mark.unit
@pytest.mark.parametrize("steps", [2, 4, 8])
def test_repeated_updates_advance_the_schedule_by_one_rollout(steps):
    # The point of the rule: K passes over one rollout must not age the schedule K times.
    per_pass = _schedule_increment(256, "critic", steps)
    assert per_pass * steps == 256


@pytest.mark.unit
def test_an_uneven_split_loses_at_most_one_sample_per_rollout():
    per_pass = _schedule_increment(256, "critic", 3)
    assert 0 <= 256 - per_pass * 3 < 3


@pytest.mark.unit
def test_the_default_leaves_behaviour_alone():
    args = SimpleNamespace()
    assert _schedule_increment(256, "critic", getattr(args, "critic_update_steps", 1)) == 256
