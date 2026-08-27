"""CPU-only scheduler/concurrency/staleness tests for the vime+polar bridge.

NON-INVASIVE: touches only public API + test-local monkeypatch of module state;
NO source changes. Delete this file to remove. No NPU, no polar server, no threads
(the AsyncPolarRolloutWorker is constructed but never .start()ed, so no HTTP loop).

Focus: reproduce the G2-1 hang condition and prove the fix (policy-version advance),
plus abort->dummy and multi-group GRPO reward parity with the slime oracle behavior.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import vime_bridge.rollout as R
from vime_bridge import wire
from vime_bridge.adapter import session_result_to_samples
from vime_bridge.reward_post_process import post_process_rewards
from vime_bridge.rollout import AsyncPolarRolloutWorker


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
