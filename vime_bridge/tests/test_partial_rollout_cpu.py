from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import vime_bridge.rollout as R
from vime_bridge.config import resolve_polar_slime_config


def args(**overrides):
    values = dict(
        polar_url="http://polar:8080",
        polar_run_id="partial-test",
        rollout_scheduler_mode="session_pool",
        rollout_batch_size=16,
        n_samples_per_prompt=8,
        rollout_max_active_sessions=64,
        rollout_max_owned_groups=24,
        polar_partial_rollout=True,
        polar_policy_transition_enabled=True,
        use_tis=True,
        use_rollout_logprobs=False,
        get_mismatch_metrics=True,
        offload_rollout=True,
        update_weights_interval=1,
        rollout_min_complete_accept_fraction=1.0,
    )
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize(
    "override",
    [
        {"rollout_max_off_policy_steps": 2},
        {"rollout_max_off_policy_steps": 0},
        {"mask_offpolicy_in_partial_rollout": True},
        {"use_tis": False},
        {"use_rollout_logprobs": True},
        {"get_mismatch_metrics": False},
        {"rollout_scheduler_mode": "group"},
        {"polar_policy_transition_enabled": False},
        {"offload_rollout": False},
        {"rollout_min_complete_accept_fraction": 0.6},
    ],
)
def test_partial_rejects_unsafe_configuration(override):
    with pytest.raises(ValueError):
        resolve_polar_slime_config(args(**override))


def worker():
    return R.AsyncPolarRolloutWorker(args(), SimpleNamespace(get_samples=lambda n: []))


def ready(w, gid, epoch):
    sample = SimpleNamespace(metadata={"polar": {"trajectory_metadata": {"oldest_policy_version": epoch}}})
    group = R._CompletedGroup(gid, [], [sample], f"task-{gid}", epoch, epoch, 8)
    w._ready_groups[gid] = R._ReadyGroup(completed=group)


def test_age_one_ready_groups_are_prioritized_age_two_expires():
    w = worker()
    assert w.config.max_off_policy_steps == 1
    w.update_policy_version(2)
    ready(w, 10, 2)
    ready(w, 11, 0)
    ready(w, 12, 1)
    accepted = w.drain_completed(max_groups=2, rollout_id=2)
    assert [g.group_id for g in accepted] == [12, 10]
    assert w.snapshot_metrics()["polar/dropped_stale_groups"] == 1


def test_pause_stops_even_missing_sibling_admission_and_cutoff_retains_age_one():
    async def run():
        w = worker()
        w.begin_policy_update_drain(2)
        assert not w._can_admit_session_pool_unit({}, set(), {})
        ready(w, 0, 0)
        ready(w, 1, 1)
        w._policy_cutoff_requested = 1  # target epoch 2 minus supported lag 1
        groups = {i: R._SessionGroupAccumulator(i, i, [object()] * 8, i, i, f"parent-{i}") for i in (0, 1)}
        await w._apply_pending_session_pool_policy_cutoff({}, set(), {}, groups)
        assert list(groups) == [1]
        assert list(w._ready_groups) == [1]
        w.finish_policy_update_drain()
        assert w._can_admit_session_pool_unit({}, set(), {})

    asyncio.run(run())


def test_old_group_missing_sibling_is_submitted_under_current_policy(monkeypatch):
    async def run():
        w = worker()
        w.update_policy_version(1)
        unit = SimpleNamespace(policy_version=0)
        monkeypatch.setattr(R, "_build_session_unit_payload", lambda **kw: {"metadata": {"policy_version": 0}})

        async def submit(client, payload):
            return payload

        w._submit_payload = submit
        payload = await w._submit_session_unit(None, unit)
        assert payload["metadata"] == {"policy_version": 1, "group_policy_version": 0, "partial_rollout": True}

    asyncio.run(run())


def test_seven_completed_sessions_survive_update_until_eighth_finishes():
    from vime_bridge import wire

    async def run():
        w = worker()
        group = [SimpleNamespace(index=i, group_index=0) for i in range(8)]
        accumulator = R._SessionGroupAccumulator(0, 0, group, 0, 0, "parent")
        units = [R._next_session_pool_unit(accumulator) for _ in range(8)]

        def result_for(i):
            trace = wire.Trace(prompt_ids=[1], response_ids=[2], loss_mask=[1], response_logprobs=[-0.2], finish_reason="stop", reward=float(i % 2))
            session = wire.SessionResult(
                session_id=f"s{i}",
                task_id=units[i].task_id,
                status=wire.SessionStatus.COMPLETED,
                trajectory=wire.Trajectory(status="COMPLETED", traces=[trace], metadata={"oldest_policy_version": 0}),
            )
            return wire.TaskResult(task_id=units[i].task_id, status="completed", results=[session])

        for i in range(7):
            R._record_session_unit_result(config=w.config, accumulator=accumulator, unit=units[i], task_result=result_for(i))
        original_slots = dict(accumulator.slots)
        open_groups = {0: accumulator}
        eighth = asyncio.get_running_loop().create_future()
        poll = asyncio.ensure_future(eighth)
        active = {poll: units[7]}
        run_pending = {units[7].task_id}
        w.begin_policy_update_drain(1)
        w._policy_cutoff_requested = 0
        await w._apply_pending_session_pool_policy_cutoff(active, run_pending, {}, open_groups)
        w._finish_terminal_session_pool_groups(open_groups)
        assert open_groups[0] is accumulator and not poll.cancelled()
        assert not w._ready_groups and accumulator.completed_count == 7
        assert all(accumulator.slots[i] is original_slots[i] for i in range(7))
        assert R._next_session_pool_unit(accumulator) is None  # no resubmission of 7 siblings
        w.update_policy_version(1)
        w.finish_policy_update_drain()
        eighth.set_result(result_for(7))
        R._record_session_unit_result(config=w.config, accumulator=accumulator, unit=units[7], task_result=await poll)
        w._finish_terminal_session_pool_groups(open_groups)
        accepted = w.drain_completed(max_groups=1, rollout_id=1)
        assert not open_groups and len(accepted) == 1
        assert accepted[0].session_count == 8
        assert [R._sample_session_id(s) for s in accepted[0].samples] == [f"s{i}" for i in range(8)]
        assert all(s.loss_mask == [1] for s in accepted[0].samples)
        assert all(accepted[0].samples[i] is original_slots[i].samples[0] for i in range(7))

    asyncio.run(run())
