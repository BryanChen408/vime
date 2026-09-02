"""CPU regressions for capacity-aware sticky routing in the VIME DP proxy."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import logging
import os
import sys
import types
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


NUM_GPUS = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
PROXY_PATH = REPO_ROOT / "scripts" / "dp_load_balance_proxy_server.py"


def _load_proxy_module():
    """Load the standalone proxy without requiring a full vLLM installation."""
    module_name = "_vime_test_dp_load_balance_proxy_server"
    if module_name in sys.modules:
        return sys.modules[module_name]

    saved_modules = {name: sys.modules.get(name) for name in ("vllm", "vllm.logger")}
    saved_event_loop_policy = asyncio.get_event_loop_policy()
    vllm_module = types.ModuleType("vllm")
    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = logging.getLogger
    vllm_module.logger = logger_module
    sys.modules["vllm"] = vllm_module
    sys.modules["vllm.logger"] = logger_module
    os.environ.setdefault("POLAR_LB_PROXY_LOG", "/tmp/vime_test_dp_lb_proxy.log")

    try:
        spec = importlib.util.spec_from_file_location(module_name, PROXY_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        asyncio.set_event_loop_policy(saved_event_loop_policy)
        for name, saved in saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved


LB = _load_proxy_module()


async def _close_state(state) -> None:
    await asyncio.gather(*(server.client.aclose() for server in state.dp_servers))


@pytest.fixture
def proxy_state():
    # Capacity ratio 1:2 models the heterogeneous .56/.64 KV pools while keeping
    # deterministic small numbers for the routing assertions.
    state = LB.ProxyState(
        [("127.0.0.1", 8100), ("127.0.0.1", 8101)],
        capacity_units=[100.0, 200.0],
    )
    # Existing routing tests exercise the fail-open estimate path without making
    # network requests. Tests below opt into explicit live metric snapshots.
    state.read_live_server_loads = AsyncMock(return_value=None)
    yield state
    asyncio.run(_close_state(state))


async def _assign_and_finish(state, session_id: str, token_count: float = 10.0) -> int:
    idx = await state.select_server_by_session(session_id, token_count)
    state.release_server(idx, token_count)
    return idx


def test_new_sessions_are_distributed_by_kv_capacity(proxy_state):
    async def _exercise():
        return [await _assign_and_finish(proxy_state, f"session-{i}") for i in range(6)]

    assignments = asyncio.run(_exercise())

    # Equal-length sessions follow the 1:2 KV-capacity ratio instead of splitting
    # equally by raw session count.
    assert Counter(assignments) == {0: 2, 1: 4}
    pressures = [
        server.estimated_session_kv_tokens / server.kv_capacity_units
        for server in proxy_state.dp_servers
    ]
    assert pressures == pytest.approx([0.2, 0.2])
    assert proxy_state.read_live_server_loads.await_count == 6


def test_existing_session_stays_pinned_and_updates_its_kv_estimate(proxy_state):
    proxy_state.read_live_server_loads.return_value = [
        LB.LiveServerLoad(0.1, 0.0, 0.0),
        LB.LiveServerLoad(0.2, 0.0, 0.0),
    ]

    async def _exercise():
        first_idx = await _assign_and_finish(proxy_state, "same-session", token_count=10.0)
        second_idx = await _assign_and_finish(proxy_state, "same-session", token_count=35.0)
        return first_idx, second_idx

    first_idx, second_idx = asyncio.run(_exercise())

    assert second_idx == first_idx
    assert proxy_state.read_live_server_loads.await_count == 1
    assert sum(server.active_sessions for server in proxy_state.dp_servers) == 1
    assert proxy_state.session_map["same-session"].estimated_kv_tokens == 35.0
    assert proxy_state.dp_servers[first_idx].estimated_session_kv_tokens == 35.0


def test_sticky_debt_prevents_deceptively_empty_engine_from_attracting_new_session(proxy_state):
    # Server 0 has just released most physical KV but still owns far more sticky
    # sessions that can return.  Dominant pressure must keep that future obligation
    # in the primary score instead of treating it as an almost-unused tie-breaker.
    proxy_state.dp_servers[0].estimated_session_kv_tokens = 90.0
    proxy_state.dp_servers[1].estimated_session_kv_tokens = 10.0
    proxy_state.read_live_server_loads.return_value = [
        LB.LiveServerLoad(0.05, 0.0, 0.0),
        LB.LiveServerLoad(0.80, 0.0, 0.0),
    ]

    idx = asyncio.run(proxy_state.select_server_by_session("live-kv-session", 10.0))

    assert idx == 1


def test_real_kv_pressure_still_protects_physically_full_engine():
    state = LB.ProxyState(
        [("127.0.0.1", 8150), ("127.0.0.1", 8151)],
        capacity_units=[100.0, 100.0],
    )
    state.dp_servers[0].estimated_session_kv_tokens = 10.0
    state.dp_servers[1].estimated_session_kv_tokens = 50.0
    state.read_live_server_loads = AsyncMock(
        return_value=[
            LB.LiveServerLoad(0.90, 0.0, 0.0),
            LB.LiveServerLoad(0.10, 0.0, 0.0),
        ]
    )

    async def _exercise():
        try:
            return await state.select_server_by_session("physical-kv-session", 10.0)
        finally:
            await _close_state(state)

    assert asyncio.run(_exercise()) == 1


def test_concurrent_new_sessions_reserve_load_before_metrics_catch_up():
    state = LB.ProxyState(
        [("127.0.0.1", 8200), ("127.0.0.1", 8201)],
        capacity_units=[100.0, 100.0],
    )
    state.read_live_server_loads = AsyncMock(
        return_value=[
            LB.LiveServerLoad(0.0, 0.0, 0.0),
            LB.LiveServerLoad(0.0, 0.0, 0.0),
        ]
    )

    async def _exercise():
        try:
            return await asyncio.gather(
                state.select_server_by_session("concurrent-a", 40.0),
                state.select_server_by_session("concurrent-b", 40.0),
            )
        finally:
            await _close_state(state)

    assignments = asyncio.run(_exercise())

    assert assignments == [0, 1]


def test_current_heterogeneous_pool_gets_five_and_six_session_quotas():
    capacities = [1_376_256.0] * 8 + [1_691_648.0] * 4
    state = LB.ProxyState(
        [("127.0.0.1", 8200 + i) for i in range(12)],
        capacity_units=capacities,
        max_active_sessions=64,
    )

    try:
        assert state.session_quotas == [5] * 8 + [6] * 4
        assert sum(state.session_quotas) == 64
        snapshot = state.snapshot()
        assert snapshot["max_active_sessions"] == 64
        assert snapshot["session_quotas"] == [5] * 8 + [6] * 4
        assert [server["session_quota"] for server in snapshot["servers"]] == [5] * 8 + [6] * 4
    finally:
        asyncio.run(_close_state(state))


def test_all_64_max_length_sessions_stay_within_capacity_weighted_quotas():
    capacities = [1_376_256.0] * 8 + [1_691_648.0] * 4
    state = LB.ProxyState(
        [("127.0.0.1", 8400 + i) for i in range(12)],
        capacity_units=capacities,
        max_active_sessions=64,
    )
    # Keep one backend deceptively empty while every other live gauge remains at
    # 40%. Without the quota boundary, completed first turns can repeatedly fall
    # back into that same backend after active_tokens returns to zero.
    state.read_live_server_loads = AsyncMock(
        return_value=[LB.LiveServerLoad(0.0, 0.0, 0.0)]
        + [LB.LiveServerLoad(0.4, 0.0, 0.0)] * 11
    )

    async def _exercise():
        try:
            for i in range(64):
                await _assign_and_finish(state, f"max-length-{i}", token_count=262_144.0)
        finally:
            await _close_state(state)

    asyncio.run(_exercise())

    assert [server.active_sessions for server in state.dp_servers] == [5] * 8 + [6] * 4
    assert state.read_live_server_loads.await_count == 64
    pressures = [
        server.estimated_session_kv_tokens / server.kv_capacity_units
        for server in state.dp_servers
    ]
    assert max(pressures) == pytest.approx(20 / 21)
    assert all(pressure < 1.0 for pressure in pressures)


def test_full_quotas_fail_open_and_terminal_release_reopens_exact_slot():
    state = LB.ProxyState(
        [("127.0.0.1", 8600), ("127.0.0.1", 8601)],
        capacity_units=[100.0, 200.0],
        max_active_sessions=3,
    )
    state.read_live_server_loads = AsyncMock(return_value=None)

    async def _exercise():
        try:
            assignments = [
                await _assign_and_finish(state, f"quota-{i}", token_count=10.0)
                for i in range(3)
            ]
            assert [server.active_sessions for server in state.dp_servers] == [1, 2]

            # A delayed terminal callback must not make the proxy reject session 4.
            overflow_idx = await _assign_and_finish(state, "quota-overflow", token_count=10.0)
            assert sum(server.active_sessions for server in state.dp_servers) == 4

            server_zero_session = next(
                session_id
                for session_id, binding in state.session_map.items()
                if binding.server_idx == 0
            )
            state.release_sticky_session(server_zero_session)
            reopened_idx = await _assign_and_finish(state, "quota-reopened", token_count=10.0)
            return assignments, overflow_idx, reopened_idx
        finally:
            await _close_state(state)

    assignments, overflow_idx, reopened_idx = asyncio.run(_exercise())

    assert Counter(assignments) == {0: 1, 1: 2}
    assert overflow_idx in (0, 1)
    assert reopened_idx == 0


def test_policy_boundary_clear_is_fail_closed_until_requests_are_drained(proxy_state):
    idx = asyncio.run(proxy_state.select_server_by_session("busy-session", 10.0))

    with pytest.raises(RuntimeError, match="active requests"):
        proxy_state.clear_sticky_cache()

    proxy_state.release_server(idx, 10.0)
    result = proxy_state.clear_sticky_cache()

    assert result == {"status": "ok", "cleared_sessions": 1}
    assert proxy_state.session_map == {}
    assert all(server.active_sessions == 0 for server in proxy_state.dp_servers)
    assert all(server.estimated_session_kv_tokens == 0 for server in proxy_state.dp_servers)


def test_terminal_session_release_is_targeted_and_idempotent(proxy_state):
    async def _exercise():
        first_idx = await _assign_and_finish(proxy_state, "finished-session", token_count=35.0)
        other_idx = await _assign_and_finish(proxy_state, "live-session", token_count=15.0)
        return first_idx, other_idx

    first_idx, other_idx = asyncio.run(_exercise())

    first = proxy_state.release_sticky_session("finished-session")
    duplicate = proxy_state.release_sticky_session("finished-session")

    assert first == {
        "status": "ok",
        "released": True,
        "session_id": "finished-session",
        "server_idx": first_idx,
    }
    assert duplicate == {
        "status": "ok",
        "released": False,
        "session_id": "finished-session",
    }
    assert "finished-session" not in proxy_state.session_map
    assert proxy_state.session_map["live-session"].server_idx == other_idx
    assert sum(server.active_sessions for server in proxy_state.dp_servers) == 1
    assert sum(server.estimated_session_kv_tokens for server in proxy_state.dp_servers) == 15.0


def test_server_info_capacity_and_token_id_estimation_use_existing_fields(proxy_state):
    capacity = LB._kv_capacity_units_from_server_info(
        {
            "vllm_config": {
                "cache_config": {
                    "num_gpu_blocks": 1024,
                    "block_size": 16,
                }
            }
        }
    )

    assert capacity == 16384.0
    assert proxy_state.estimate_prompt_tokens({"prompt_token_ids": list(range(37))}, 9999) == 37
    assert proxy_state.estimate_prompt_tokens({"prompt": [[1, 2], [3, 4, 5]]}, 9999) == 5
    assert proxy_state.estimate_prompt_tokens({"messages": [{"content": "fallback"}]}, 400) == 100


def test_live_load_parser_uses_vllm_prometheus_gauges():
    load = LB._live_server_load_from_prometheus(
        """
# HELP vllm:kv_cache_usage_perc KV-cache usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="model"} 0.625
# HELP vllm:num_requests_running Number of running requests.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="model"} 7
# HELP vllm:num_requests_waiting Number of waiting requests.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="model"} 2
"""
    )

    assert load == LB.LiveServerLoad(
        kv_cache_usage=0.625,
        running_requests=7.0,
        waiting_requests=2.0,
    )


def test_live_load_parser_fails_closed_when_required_gauge_is_missing():
    with pytest.raises(ValueError, match="missing required gauges"):
        LB._live_server_load_from_prometheus(
            """
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0"} 0.25
"""
        )


def test_live_metric_collection_is_all_or_nothing_and_backs_off_after_failure():
    metrics = """
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0"} 0.25
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0"} 3
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0"} 0
"""

    class _Response:
        text = metrics

        def raise_for_status(self):
            return None

    class _Client:
        def __init__(self, fail: bool):
            self.fail = fail
            self.calls = 0

        async def get(self, *args, **kwargs):
            self.calls += 1
            if self.fail:
                raise RuntimeError("metrics unavailable")
            return _Response()

        async def aclose(self):
            return None

    state = LB.ProxyState(
        [("127.0.0.1", 8300), ("127.0.0.1", 8301)],
        capacity_units=[100.0, 100.0],
    )

    async def _exercise():
        await _close_state(state)
        healthy = _Client(fail=False)
        failed = _Client(fail=True)
        state.dp_servers[0].client = healthy
        state.dp_servers[1].client = failed
        try:
            assert await state.read_live_server_loads() is None
            # The immediate retry takes the estimate fallback without hitting either
            # endpoint again, so a failed backend cannot serialize a whole burst.
            assert await state.read_live_server_loads() is None
            return healthy.calls, failed.calls
        finally:
            await _close_state(state)

    calls = asyncio.run(_exercise())

    assert calls == (1, 1)


def test_capacity_discovery_is_all_or_nothing(proxy_state):
    class _Response:
        def __init__(self, blocks: int):
            self.blocks = blocks

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "vllm_config": {
                    "cache_config": {
                        "num_gpu_blocks": self.blocks,
                        "block_size": 16,
                    }
                }
            }

    class _Client:
        def __init__(self, blocks: int | None):
            self.blocks = blocks

        async def get(self, *args, **kwargs):
            if self.blocks is None:
                raise RuntimeError("backend unavailable")
            return _Response(self.blocks)

        async def aclose(self):
            return None

    async def _exercise():
        await _close_state(proxy_state)
        proxy_state.dp_servers[0].client = _Client(100)
        proxy_state.dp_servers[1].client = _Client(200)
        assert await proxy_state.discover_kv_capacities() is True
        assert [server.kv_capacity_units for server in proxy_state.dp_servers] == [1600.0, 3200.0]

        proxy_state.dp_servers[1].client = _Client(None)
        assert await proxy_state.discover_kv_capacities() is False
        assert [server.kv_capacity_units for server in proxy_state.dp_servers] == [1.0, 1.0]

    asyncio.run(_exercise())


def test_capacity_discovery_failure_disables_weighted_session_quotas():
    class _Response:
        def __init__(self, blocks: int):
            self.blocks = blocks

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "vllm_config": {
                    "cache_config": {
                        "num_gpu_blocks": self.blocks,
                        "block_size": 10,
                    }
                }
            }

    class _Client:
        def __init__(self, blocks: int | None):
            self.blocks = blocks

        async def get(self, *args, **kwargs):
            if self.blocks is None:
                raise RuntimeError("backend unavailable")
            return _Response(self.blocks)

        async def aclose(self):
            return None

    state = LB.ProxyState(
        [("127.0.0.1", 8700), ("127.0.0.1", 8701)],
        max_active_sessions=3,
    )

    async def _exercise():
        await _close_state(state)
        state.dp_servers[0].client = _Client(10)
        state.dp_servers[1].client = _Client(20)
        assert await state.discover_kv_capacities() is True
        assert state.session_quotas == [1, 2]

        state.dp_servers[1].client = _Client(None)
        assert await state.discover_kv_capacities() is False
        assert state.session_quotas is None
        assert [server.kv_capacity_units for server in state.dp_servers] == [1.0, 1.0]
        await _close_state(state)

    asyncio.run(_exercise())


def test_proxy_lifespan_retries_capacity_discovery_in_background(monkeypatch):
    class _Client:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    class _State:
        def __init__(self):
            self.dp_servers = [types.SimpleNamespace(client=_Client())]
            self.discovery_calls = 0

        async def discover_kv_capacities(self):
            self.discovery_calls += 1
            return self.discovery_calls >= 2

    state = _State()
    monkeypatch.setattr(LB, "ProxyState", lambda *args, **kwargs: state)
    monkeypatch.setattr(
        LB,
        "global_args",
        types.SimpleNamespace(server_instances=[("127.0.0.1", 8800)], max_active_sessions=64),
        raising=False,
    )
    monkeypatch.setattr(LB, "_SESSION_POLICY", "least_load")
    monkeypatch.setattr(LB, "_CAPACITY_RETRY_INTERVAL", 0.0)

    async def _exercise():
        async with LB.lifespan(None):
            for _ in range(10):
                if state.discovery_calls >= 2:
                    break
                await asyncio.sleep(0)

    asyncio.run(_exercise())

    assert state.discovery_calls == 2
    assert state.dp_servers[0].client.closed is True


def test_cancelled_forward_releases_live_load(proxy_state, monkeypatch):
    class _Request:
        headers = {"x-session-id": "cancelled-session"}

        async def json(self):
            return {"prompt_token_ids": list(range(20)), "max_tokens": 10}

        async def body(self):
            return b"request-body"

    async def _cancelled_forward(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(LB, "proxy_state", proxy_state)
    monkeypatch.setattr(LB, "global_args", types.SimpleNamespace(max_retries=3, retry_delay=0.001), raising=False)
    monkeypatch.setattr(LB, "_forward_upstream_with_retry", _cancelled_forward)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(LB._handle_completions("/completions", _Request()))

    assert all(server.active_requests == 0 for server in proxy_state.dp_servers)
    assert all(server.active_tokens == 0 for server in proxy_state.dp_servers)


def test_rollout_manager_clears_affinity_before_resuming_polar():
    """Lock the fail-closed ordering without importing Ray/NPU runtime dependencies."""
    tree = ast.parse((REPO_ROOT / "vime" / "ray" / "rollout.py").read_text())
    manager = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RolloutManager"
    )
    finish = next(
        node
        for node in manager.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "finish_policy_update"
    )
    call_lines = {
        ast.unparse(node.func): node.lineno
        for node in ast.walk(finish)
        if isinstance(node, ast.Call)
    }

    assert call_lines["srv.clear_lb_proxy_sticky_cache"] < call_lines["self._call_rollout_function_hook"]


def test_rollout_manager_passes_existing_active_session_limit_to_lb_proxy():
    tree = ast.parse((REPO_ROOT / "vime" / "ray" / "rollout.py").read_text())
    start_proxy = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_start_lb_proxy"
    )
    source = ast.unparse(start_proxy)

    assert "rollout_max_active_sessions" in source
    assert "--max-active-sessions" in source


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
