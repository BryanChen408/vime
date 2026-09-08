"""CPU-only scheduler/concurrency/staleness tests for the vime+polar bridge.

NON-INVASIVE: touches only public API + test-local monkeypatch of module state;
NO source changes. Delete this file to remove. No NPU or Polar server is required.

Focus: reproduce the G2-1 hang condition and prove the fix (policy-version advance),
plus abort->dummy and multi-group GRPO reward parity with the slime oracle behavior.
"""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import httpx
import pytest

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import vime_bridge.rollout as R
from vime_bridge import wire
from vime_bridge.adapter import session_result_to_samples
from vime_bridge.reward_post_process import post_process_rewards
from vime_bridge.rollout import AsyncPolarRolloutWorker, PolarRolloutSchedulerError


class _DummyDataSource:
    def get_samples(self, n):
        return []


def _polar_args(start_rollout_id=0, max_async_level=2, update_weights_interval=1,
                scheduler_mode="session_pool"):
    return SimpleNamespace(
        polar_url="http://polar:8080", polar_rollout_url=None, polar_run_id="run1",
        polar_reward_key="score", reward_key="score",
        polar_task_id_template="{args.polar_run_id}-op-{rollout_id}-{sample.group_index}",
        operator_tasks_dir="/tmp/op_tasks", polar_tasks_dir=None,
        rollout_scheduler_mode=scheduler_mode, rollout_max_active_sessions=16,
        rollout_max_owned_groups=4,
        rollout_release_on_postrun=True, rollout_min_complete_accept_fraction=0.6,
        rollout_max_async_level=max_async_level, rollout_request_timeout=4000,
        rollout_batch_size=2, n_samples_per_prompt=8,
        update_weights_interval=update_weights_interval,
        hf_checkpoint="/tmp/hf", start_rollout_id=start_rollout_id,
    )


# ---------------------------------------------------------------- G2-1

def test_g2_1_policy_version_freeze_reproduced_and_fix_validated():
    """The exact G2-1 mechanism, no threads/HTTP: with a frozen policy version the
    staleness gate drops every group past max_off_policy_steps (=hang); advancing
    the version each step (what our train_async fix does) keeps staleness bounded."""
    args = _polar_args(start_rollout_id=0, max_async_level=2, update_weights_interval=1)
    worker = AsyncPolarRolloutWorker(args, _DummyDataSource())
    assert worker.config.max_off_policy_steps == 3   # 2 + 1
    assert worker._policy_version == 0

    # BUG repro: only set_rollout_context advances rollout_id; version stays frozen.
    for rid in range(0, 5):
        worker.set_rollout_context(rid)
    assert worker._policy_version == 0
    staleness_frozen = worker._current_rollout_id - worker._policy_version  # 4 - 0
    assert staleness_frozen == 4
    assert staleness_frozen > worker.config.max_off_policy_steps  # -> group dropped -> hang

    # FIX: our train_async calls update_policy_version(rid+1) after each update_weights.
    worker2 = AsyncPolarRolloutWorker(args, _DummyDataSource())
    for rid in range(0, 5):
        worker2.set_rollout_context(rid)
        worker2.update_policy_version(rid + 1)
    assert worker2._policy_version == 5
    staleness_fixed = worker2._current_rollout_id - worker2._policy_version  # 4 - 5 -> clamped 0
    assert max(0, staleness_fixed) <= worker2.config.max_off_policy_steps  # accepted -> no hang


def test_module_update_policy_version_hook_dispatch(monkeypatch):
    """RolloutManager.update_policy_version -> module hook -> worker (the chain the
    vime-core fix relies on). Set the module global to a NON-started worker."""
    args = _polar_args()
    worker = AsyncPolarRolloutWorker(args, _DummyDataSource())  # not started; no thread
    monkeypatch.setattr(R, "_global_async_worker", worker)
    assert worker._policy_version == 0
    R.update_policy_version(args, 7)   # module-level hook the RolloutManager calls
    assert worker._policy_version == 7
    # idempotent / monotonic
    R.update_policy_version(args, 3)
    assert worker._policy_version == 7


# ---------------------------------------------------------------- adapter / reward parity

def _trace(rids, reward, finish="stop"):
    n = len(rids)
    return wire.Trace(prompt_ids=[1, 2, 3], response_ids=rids, loss_mask=[1] * n,
                      prompt_messages=[{"role": "user", "content": "hi"}],
                      response_messages=[{"role": "assistant", "content": "ok"}],
                      finish_reason=finish, response_logprobs=[-0.1] * n, reward=reward)


def _session(sid, traces, status=wire.SessionStatus.COMPLETED, tstatus="COMPLETED", terr=None):
    return wire.SessionResult(session_id=sid, task_id="t-" + sid, status=status,
                              trajectory=wire.Trajectory(status=tstatus, traces=traces, error=terr))


def test_aborted_session_yields_removable_dummy_sample():
    # A session whose only trace has no response tokens -> all dropped -> one fully
    # masked dummy carrying remove_sample so the group stays trainable (oracle behavior).
    bad = wire.Trace(prompt_ids=[1, 2], response_ids=[], loss_mask=[],
                     prompt_messages=[{"role": "user", "content": "hi"}], response_messages=[])
    result = _session("bad", [bad], status=wire.SessionStatus.ERROR,
                      tstatus="ERROR", terr="boom")
    samples = session_result_to_samples(result, group_index=0, trajectory_index=0,
                                        reward_key="score")
    assert len(samples) == 1
    s = samples[0]
    assert getattr(s, "remove_sample", False) is True
    assert s.status.name in ("ABORTED", "FAILED")
    assert s.reward == {"score": 0.0}


def test_grpo_reward_normalization_across_two_groups():
    args = SimpleNamespace(rewards_normalization=True, advantage_estimator="grpo",
                           grpo_std_normalization=True, reward_key="score", polar_reward_key=None)
    s = []
    # group 0: two trajectories (rewards 1.0 / 0.0); group 1: two (0.5 / 0.5)
    s += session_result_to_samples(_session("a", [_trace([1, 2], 1.0)]),
                                   group_index=0, trajectory_index=0, reward_key="score")
    s += session_result_to_samples(_session("b", [_trace([1, 2], 0.0)]),
                                   group_index=0, trajectory_index=1, reward_key="score")
    s += session_result_to_samples(_session("c", [_trace([1, 2], 0.5)]),
                                   group_index=1, trajectory_index=0, reward_key="score")
    s += session_result_to_samples(_session("d", [_trace([1, 2], 0.5)]),
                                   group_index=1, trajectory_index=1, reward_key="score")
    raw, rewards = post_process_rewards(args, s)
    assert len(raw) == len(s) == 4
    assert len(rewards) == 4
    # group 1 has zero variance -> normalized rewards ~0; group 0 has spread -> non-zero
    assert abs(rewards[2]) < 1e-6 and abs(rewards[3]) < 1e-6


def test_session_pool_config_resolution():
    cfg = AsyncPolarRolloutWorker(_polar_args(scheduler_mode="session_pool"),
                                  _DummyDataSource()).config
    assert cfg.scheduler_mode == "session_pool"
    assert cfg.max_active_sessions == 16
    assert cfg.max_owned_groups == 4
    assert cfg.session_pool_release_on_postrun is True
    assert abs(cfg.min_complete_accept_fraction - 0.6) < 1e-9
    assert cfg.max_off_policy_steps == 3


def test_session_pool_owned_group_limit_preserves_postrun_release():
    worker = AsyncPolarRolloutWorker(_polar_args(scheduler_mode="session_pool"),
                                     _DummyDataSource())
    active = {}
    run_pending = set()

    # Three ready groups leave room for one new group under the configured limit 4.
    worker._ready_group_count = 3
    assert worker._can_admit_session_pool_unit(active, run_pending, {}) is True

    # Four scheduler-owned groups stop a new group even though RUN admission is empty.
    worker._ready_group_count = 4
    assert worker._can_admit_session_pool_unit(active, run_pending, {}) is False

    # A partial group is allowed to finish so an 8-sample GRPO group is never split.
    worker._ready_group_count = 3
    partial = SimpleNamespace(group_id=9, rejected_reason=None, partial=True)
    open_groups = {9: partial}
    assert worker._can_admit_session_pool_unit(active, run_pending, open_groups) is True

    # The owned-group exception never bypasses the existing RUN concurrency cap.
    run_pending = {f"task-{index}" for index in range(worker.config.max_active_sessions)}
    assert worker._can_admit_session_pool_unit(active, run_pending, open_groups) is False


def test_session_pool_owned_group_limit_is_optional():
    args = _polar_args(scheduler_mode="session_pool")
    args.rollout_max_owned_groups = None
    worker = AsyncPolarRolloutWorker(args, _DummyDataSource())
    worker._ready_group_count = 100
    assert worker.config.max_owned_groups is None
    assert worker._can_admit_session_pool_unit({}, set(), {}) is True


def test_policy_cutoff_reaps_old_owned_groups_and_unblocks_admission():
    async def scenario():
        args = _polar_args(scheduler_mode="session_pool")
        # Even when generic off-policy reuse is enabled, a no-drain weight boundary is a
        # hard cutoff: no group owned by the previous policy may survive it.
        args.rollout_max_off_policy_steps = 10
        worker = AsyncPolarRolloutWorker(
            args,
            _DummyDataSource(),
        )

        def accumulator(group_id: int, policy_version: int):
            return R._SessionGroupAccumulator(
                group_id=group_id,
                group_pos=group_id,
                group=[object()] * 8,
                submitted_rollout_id=policy_version,
                policy_version=policy_version,
                parent_task_id=f"group-{group_id}",
                next_submit_pos=8,
            )

        old_open = {group_id: accumulator(group_id, 0) for group_id in range(3)}
        current_open = accumulator(4, 1)
        open_groups = {**old_open, current_open.group_id: current_open}

        old_ready = R._CompletedGroup(
            group_id=3,
            group=[object()] * 8,
            samples=[],
            task_id="old-ready",
            submitted_rollout_id=0,
            policy_version=0,
            session_count=8,
        )
        current_ready = R._CompletedGroup(
            group_id=5,
            group=[object()] * 8,
            samples=[],
            task_id="current-ready",
            submitted_rollout_id=1,
            policy_version=1,
            session_count=8,
        )
        worker._ready_groups = {
            old_ready.group_id: R._ReadyGroup(completed=old_ready),
            current_ready.group_id: R._ReadyGroup(completed=current_ready),
        }
        worker._ready_group_count = 2

        old_unit = R._PendingSessionUnit(
            group_id=0,
            group_pos=0,
            sample_pos=0,
            sample=object(),
            parent_group=[object()] * 8,
            parent_task_id="group-0",
            task_id="old-unit",
            submitted_rollout_id=0,
            policy_version=0,
        )
        old_task = asyncio.create_task(asyncio.sleep(60))
        active = {old_task: old_unit}
        run_pending = {old_unit.task_id}
        next_status_poll_at = {old_unit.task_id: 123.0}
        remote_cutoffs = []

        async def cancel_remote(task_ids, *, policy_version):
            remote_cutoffs.append((task_ids, policy_version))

        worker._cancel_remote_policy_tasks = cancel_remote

        # Three old open groups + one current open group + two ready groups fill the
        # configured owned limit. This is the post-weight-update deadlock seen in the
        # 20260827 run: Polar is idle but VIME refuses to admit the new policy.
        assert worker._can_admit_session_pool_unit(
            active,
            run_pending,
            open_groups,
        ) is False

        worker._policy_cutoff_requested = 1
        worker._policy_cutoff_complete.clear()
        await worker._apply_pending_session_pool_policy_cutoff(
            active,
            run_pending,
            next_status_poll_at,
            open_groups,
        )

        assert old_task.cancelled()
        assert remote_cutoffs == [(["old-unit"], 1)]
        assert active == {}
        assert run_pending == set()
        assert next_status_poll_at == {}
        assert set(open_groups) == {current_open.group_id}
        assert set(worker._ready_groups) == {current_ready.group_id}
        assert worker._ready_group_count == 1
        assert worker._policy_cutoff_applied == 1
        assert worker._policy_cutoff_requested is None
        assert worker._policy_cutoff_complete.is_set()
        assert worker._can_admit_session_pool_unit(
            active,
            run_pending,
            open_groups,
        ) is True

    asyncio.run(scenario())


def test_policy_cutoff_request_runs_on_session_pool_worker_thread():
    worker = AsyncPolarRolloutWorker(
        _polar_args(scheduler_mode="session_pool"),
        _DummyDataSource(),
    )
    worker.start()
    try:
        worker.begin_policy_update_drain(1)
        worker.abandon_policy_versions_before(1, timeout=2.0)

        assert worker._policy_cutoff_applied == 1
        assert worker._policy_cutoff_requested is None
        assert worker._policy_cutoff_complete.is_set()
    finally:
        worker.stop()

    assert worker.is_alive() is False


def test_remote_policy_cutoff_requires_polar_ack(monkeypatch):
    calls = []

    class Response:
        status_code = 200
        text = "ok"

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "all_acknowledged": True,
                "sessions_cancel_requested": 2,
            }

    class Client:
        def __init__(self, *, timeout):
            calls.append(("timeout", timeout))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json):
            calls.append((url, json))
            return Response()

    monkeypatch.setattr(R.httpx, "AsyncClient", Client)
    worker = AsyncPolarRolloutWorker(
        _polar_args(scheduler_mode="session_pool"),
        _DummyDataSource(),
    )

    asyncio.run(
        worker._cancel_remote_policy_tasks(
            ["task-a", "task-b"],
            policy_version=3,
        )
    )

    assert calls[-1] == (
        "http://polar:8080/rollout/admin/tasks/cancel",
        {
            "task_ids": ["task-a", "task-b"],
            "reason": "policy_cutoff",
        },
    )


def test_remote_policy_cutoff_is_fail_closed_without_ack(monkeypatch):
    class Response:
        status_code = 200
        text = "not acknowledged"

        def raise_for_status(self):
            return None

        def json(self):
            return {"all_acknowledged": False}

    class Client:
        def __init__(self, *, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json):
            return Response()

    monkeypatch.setattr(R.httpx, "AsyncClient", Client)
    worker = AsyncPolarRolloutWorker(
        _polar_args(scheduler_mode="session_pool"),
        _DummyDataSource(),
    )

    with pytest.raises(PolarRolloutSchedulerError, match="not acknowledged"):
        asyncio.run(
            worker._cancel_remote_policy_tasks(
                ["task-a"],
                policy_version=3,
            )
        )


def test_session_pool_drain_primitives():
    # What prepare_policy_update / finish_policy_update drive in session_pool mode
    # (drain open groups before serving weights advance, then clear).
    worker = AsyncPolarRolloutWorker(_polar_args(scheduler_mode="session_pool"),
                                     _DummyDataSource())
    worker.begin_policy_update_drain(5)
    assert worker._policy_update_draining is True
    assert worker._policy_update_target_version == 5
    worker.finish_policy_update_drain()
    assert worker._policy_update_draining is False


def test_group_mode_admission_pause_resume():
    # What prepare_policy_update / finish_policy_update drive in group mode.
    worker = AsyncPolarRolloutWorker(_polar_args(scheduler_mode="group"),
                                     _DummyDataSource())
    worker.pause_admission()
    assert worker._admission_paused is True
    worker.resume_admission()
    assert worker._admission_paused is False


def test_task_status_poll_has_bounded_timeout_and_retries():
    poll_timeouts = []

    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json={"task_id": "task-1"})
        poll_timeouts.append(request.extensions["timeout"]["read"])
        if len(poll_timeouts) == 1:
            raise httpx.ReadTimeout("stalled task-status response", request=request)
        return httpx.Response(200, json={
            "task_id": "task-1",
            "status": "completed",
            "total_sessions": 0,
            "completed_sessions": 0,
            "results": [],
            "result_paths": [],
        })

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await R._submit_and_wait_for_task(
                client,
                "http://polar:8080",
                {"task_id": "task-1"},
                poll_interval=0,
            )

    result = asyncio.run(run())

    assert result.status == "completed"
    assert poll_timeouts == [
        R._TASK_STATUS_POLL_TIMEOUT_SECONDS,
        R._TASK_STATUS_POLL_TIMEOUT_SECONDS,
    ]


def test_scheduler_loops_open_clients_in_their_own_scope(monkeypatch):
    opened = []

    class Client:
        def __init__(self, *args, **kwargs):
            opened.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    async def run_loop(mode):
        worker = AsyncPolarRolloutWorker(_polar_args(scheduler_mode=mode),
                                         _DummyDataSource())
        worker._running = False

        async def start_callback_listener():
            return SimpleNamespace(should_exit=False), asyncio.create_task(asyncio.sleep(0))

        monkeypatch.setattr(worker, "_start_callback_listener", start_callback_listener)
        await getattr(worker, f"_async_{mode}_loop")()

    monkeypatch.setattr(R.httpx, "AsyncClient", Client)
    asyncio.run(run_loop("group"))
    assert len(opened) == 1
    opened.clear()
    asyncio.run(run_loop("session_pool"))
    assert len(opened) == 2
