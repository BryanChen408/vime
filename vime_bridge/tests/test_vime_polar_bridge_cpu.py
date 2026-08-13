"""CPU-only regression tests for the vime+polar bridge port.

Run: TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=<vime repo root> \
     python -m pytest vime_bridge/tests/test_vime_polar_bridge_cpu.py -q

No NPU, no polar server, no training — pure logic checks on the ported bridge.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

from vime_bridge import wire
from vime_bridge.adapter import _trajectory_rollout_id, session_result_to_samples
from vime_bridge.config import resolve_polar_slime_config, resolve_vllm_router_base_url
from vime_bridge.reward_post_process import post_process_rewards
from vime_bridge.rollout import _build_rollout_benchmark_metrics


def _trace(resp_ids, reward, finish="stop"):
    n = len(resp_ids)
    return wire.Trace(
        prompt_ids=[1, 2, 3],
        response_ids=resp_ids,
        loss_mask=[1] * n,
        prompt_messages=[{"role": "user", "content": "hi"}],
        response_messages=[{"role": "assistant", "content": "ok"}],
        finish_reason=finish,
        response_logprobs=[-0.1] * n,
        reward=reward,
    )


def _session(session_id, traces, status=wire.SessionStatus.COMPLETED, timing=None):
    return wire.SessionResult(
        session_id=session_id,
        task_id="t-" + session_id,
        status=status,
        trajectory=wire.Trajectory(status="COMPLETED", traces=traces),
        timing=timing or wire.SessionTiming(),
    )


def test_session_result_to_samples_sets_per_trajectory_rollout_id():
    # One trajectory (session) with 2 traces (turns) -> 2 Samples that MUST
    # share the same rollout_id so vime's native reducer weights the trajectory
    # as one unit (Option A).
    result = _session("s0", [_trace([10, 11], 1.0), _trace([12, 13, 14], 0.0)])
    samples = session_result_to_samples(
        result, group_index=1, trajectory_index=2, reward_key="score"
    )
    assert len(samples) == 2
    expected_rid = _trajectory_rollout_id(1, 2)
    for s in samples:
        assert s.group_index == 1
        assert s.index == 2
        assert s.rollout_id == expected_rid  # per-trajectory grouping key
    # Distinct trajectory -> distinct rollout_id
    other = session_result_to_samples(
        _session("s1", [_trace([20, 21], 0.5)]),
        group_index=1, trajectory_index=3, reward_key="score",
    )
    assert other[0].rollout_id != expected_rid
    assert other[0].rollout_id == _trajectory_rollout_id(1, 3)


def test_trajectory_rollout_id_is_collision_free():
    # Cantor pairing must be injective over (group_index, index) pairs, including
    # the cross-group case the old *1e6 stride could collide on (index >= stride).
    pairs = {(g, i) for g in range(6) for i in range(6)}
    pairs |= {(1, 1_000_000), (2, 7), (7, 2)}  # big index + order-asymmetry
    ids = [_trajectory_rollout_id(g, i) for g, i in pairs]
    assert len(set(ids)) == len(pairs), "rollout_id collision"


def test_post_process_rewards_returns_one_per_sample():
    args = SimpleNamespace(
        rewards_normalization=True,
        advantage_estimator="grpo",
        grpo_std_normalization=True,
        reward_key="score",
        polar_reward_key=None,
    )
    # group 0: two trajectories with different rewards -> GRPO-normalized in-group
    s = []
    s += session_result_to_samples(_session("a", [_trace([1, 2], 1.0)]),
                                   group_index=0, trajectory_index=0, reward_key="score")
    s += session_result_to_samples(_session("b", [_trace([1, 2], 0.0)]),
                                   group_index=0, trajectory_index=1, reward_key="score")
    raw, rewards = post_process_rewards(args, s)
    assert len(raw) == len(s)
    assert len(rewards) == len(s)


def test_post_process_rewards_std_floor_damps_near_tie_groups():
    # P1-a(slime 2540e19 同款):近全同组 [1,1,1,0.99] 的 std≈0.005,无 floor 时
    # 0.99 那条 advantage ≈ -1.73(因 0.01 分差吃满强度);STD_FLOOR=0.05 后 ÷0.05
    # -> -0.15,比例型。大差异组不受 floor 影响。
    args = SimpleNamespace(
        rewards_normalization=True,
        advantage_estimator="grpo",
        grpo_std_normalization=True,
        reward_key="score",
        polar_reward_key=None,
    )
    s = []
    for i, rw in enumerate([1.0, 1.0, 1.0, 0.99]):
        s += session_result_to_samples(_session(f"s{i}", [_trace([1, 2], rw)]),
                                       group_index=0, trajectory_index=i, reward_key="score")
    _raw, rewards = post_process_rewards(args, s)
    assert abs(rewards[3] - (-0.15)) < 1e-6, rewards
    assert abs(rewards[0] - 0.05) < 1e-6, rewards
    # 对照:组内 std=0.5 >> floor 时按原样白化([1.0, 0.0] -> torch 无偏方差下 ±1/√2)
    s2 = []
    s2 += session_result_to_samples(_session("hi", [_trace([1, 2], 1.0)]),
                                    group_index=0, trajectory_index=0, reward_key="score")
    s2 += session_result_to_samples(_session("lo", [_trace([1, 2], 0.0)]),
                                    group_index=0, trajectory_index=1, reward_key="score")
    _raw2, rewards2 = post_process_rewards(args, s2)
    assert abs(rewards2[0] - 0.7071) < 1e-3 and abs(rewards2[1] + 0.7071) < 1e-3, rewards2


def test_build_rollout_benchmark_metrics_matches_basic_benchmark_semantics():
    result = _session(
        "sbench",
        [_trace([10, 11, 12], 1.0)],
        timing=wire.SessionTiming(
            register_to_init_queue_ms=400.0,
            init_ms=600.0,
            run_ms=1000.0,
            postrun_ms=0.0,
        ),
    )
    samples = session_result_to_samples(result, group_index=0, trajectory_index=0, reward_key="score")
    metrics = _build_rollout_benchmark_metrics(samples)
    assert metrics["rollout_bench/total_input_tokens"] == 3.0
    assert metrics["rollout_bench/total_generated_tokens"] == 3.0
    assert abs(metrics["rollout_bench/request_throughput"] - 0.5) < 1e-9
    assert abs(metrics["rollout_bench/output_throughput"] - 1.5) < 1e-9
    assert abs(metrics["rollout_bench/total_token_throughput"] - 3.0) < 1e-9
    assert metrics["rollout_bench/peak_output_token_throughput"] == 2.0
    assert metrics["rollout_bench/peak_concurrent_requests"] == 1.0
    assert metrics["rollout_bench/ttft_mean_ms"] == 1000.0
    assert metrics["rollout_bench/ttft_median_ms"] == 1000.0
    assert metrics["rollout_bench/ttft_p99_ms"] == 1000.0
    assert metrics["rollout_bench/tpot_mean_ms"] == 500.0
    assert metrics["rollout_bench/tpot_median_ms"] == 500.0
    assert metrics["rollout_bench/tpot_p99_ms"] == 500.0
    assert metrics["rollout_bench/itl_mean_ms"] == 500.0
    assert metrics["rollout_bench/itl_median_ms"] == 500.0
    assert metrics["rollout_bench/itl_p99_ms"] == 500.0


def test_resolve_vllm_router_base_url_and_config():
    # vllm router url resolves from vime's vllm_router_ip/port
    ns = SimpleNamespace(vllm_router_ip="127.0.0.1", vllm_router_port=4077,
                         sglang_router_ip=None, sglang_router_port=None)
    assert resolve_vllm_router_base_url(ns) == "http://127.0.0.1:4077"

    # minimal polar config resolves (mirrors the live-run arg surface)
    cfg_ns = SimpleNamespace(
        polar_url="http://polar:8080", polar_rollout_url=None, polar_run_id="run1",
        polar_reward_key="score", reward_key="score",
        polar_task_id_template="{args.polar_run_id}-op-{rollout_id}-{sample.group_index}",
        operator_tasks_dir="/tmp/op_tasks", polar_tasks_dir=None,
        rollout_scheduler_mode="session_pool", rollout_max_active_sessions=16,
        rollout_release_on_postrun=True, rollout_min_complete_accept_fraction=0.6,
        rollout_max_async_level=1, rollout_request_timeout=4000,
        rollout_batch_size=2, n_samples_per_prompt=8, update_weights_interval=1,
        hf_checkpoint="/tmp/hf",
    )
    cfg = resolve_polar_slime_config(cfg_ns)
    assert cfg.rollout_server_url == "http://polar:8080"
    assert cfg.scheduler_mode == "session_pool"
    assert cfg.max_active_sessions == 16
    assert abs(cfg.min_complete_accept_fraction - 0.6) < 1e-9
