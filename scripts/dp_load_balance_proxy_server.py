# Adapted from https://github.com/vllm-project/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py

# SPDX-License-Identifier: Apache-2.0
#
# Tutorial: Using the Load Balance Proxy Server For External DP
#
# This proxy server is designed to distribute requests between multiple
# vLLM servers running in data parallel for large language model inference.
# It is useful for scaling out inference workloads and balancing load across
# multiple vLLM instances.
#
# Features:
# - Load balances requests to multiple vLLM servers.
# - Supports OpenAI-compatible /v1/completions and /v1/chat/completions endpoints.
# - Streams responses from backend servers to clients.
#
# vime 适配(ITEM 1 透传 + DP #4 的 B+ 方案 —— 见 docs/design/router_return_token_ids_passthrough.md §10):
# - 原样 dict 转发请求(req_data = request.json() → json=req_data),不像 Rust router 的 typed 解析
#   会丢 vLLM 扩展字段 return_token_ids → 保 token 保真(这是替 Rust router 的根本原因)。
# - session 亲和:读 x-session-id header(vime consistent_hash 约定);首次按实际 KV 容量配额与
#   max(实时负载,长期 session 负债)选引擎,后续钉住复用 prefix cache;无 session id 则用
#   vLLM 官方 DP proxy 同款 active_tokens。
# - 加 /health 就绪探针(polar 探测)。
#
# Prerequisites:
# - Python 3.10+
# - Install dependencies:
#     pip install fastapi<0.124.0 httpx uvicorn
#
# Step 1: Start Your Backend Servers
# ----------------------------------
# You need to have at least two vLLM servers running in data parallel.
# These can be mock servers or actual vLLM servers.
# Note that this proxy also works with only one vLLM server running, but
# will fall back to direct request forwarding which is meaningless.
#
# For testing, you can use the provided mock server:
#
#   vllm serve --host 0.0.0.0 --port 8100 --data-parallel-rank 0 ... # vLLM DP0
#   vllm serve --host 0.0.0.0 --port 8101 --data-parallel-rank 1 ... # vLLM DP1
#
# Step 2: Start the Proxy Server
# ------------------------------
# Run the proxy server, specifying the host/port for each vLLM DP Instance:
#
#   python dp_load_balance_proxy_server.py \
#     --host 0.0.0.0 --port 9000 \
#     --dp-hosts 127.0.0.1 127.0.0.1 \
#     --dp-ports 8100 8101 \
#
# This will start the proxy on port 9000, load balancing between two vLLM DP servers.
#
# Step 3: Send a Request to the Proxy
# -----------------------------------
# You can now send OpenAI-compatible requests to the proxy. For example:
#
#   curl -X POST http://localhost:9000/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "prompt": "The quick brown fox jumps over the lazy dog",
#           "max_tokens": 16
#         }'
#
# Or for chat completions:
#
#   curl -X POST http://localhost:9000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "messages": [{"role": "user", "content": "Hello!"}],
#           "max_tokens": 16
#         }'
#
# Step 4: Health Check
# --------------------
# To check if the proxy is running and see how many backend instances are
# connected, use:
#
#   curl http://localhost:9000/healthcheck
#
# This will return a JSON object with the status and the number of vLLM DP servers.
#
# Notes:
# - You can scale the number of vLLM data parallel size as needed.
# - The proxy will consider the length of requests to balance load.
# - For production, ensure your backend servers are robust and secure.
#
# For more details, see the code and comments in this file.

import argparse
import asyncio
import functools
import hashlib
import heapq
import json
import math
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from prometheus_client.parser import text_string_to_metric_families
from vllm.logger import init_logger

logger = init_logger(__name__)

# 落盘诊断日志:把 LB proxy 的 502/transport 现场(连不上 engine 的真实 httpx 异常、
# 重试耗尽、上游 HTTP 码)固定写文件,便于定位 ERROR session 里 502/transport 的根因。
# 之前这些 WARNING/ERROR 只进 Ray 汇聚的 train log,常被截断/看不全。路径可用 POLAR_LB_PROXY_LOG 覆盖。
try:
    import logging as _logging
    _lb_log = os.environ.get("POLAR_LB_PROXY_LOG",
                             os.path.join(os.environ.get("POLAR_ENGINE_METRICS_DIR", "/mnt/share/polar_engine_metrics"),
                                          "lb_proxy.log"))
    os.makedirs(os.path.dirname(_lb_log), exist_ok=True)
    _fh = _logging.FileHandler(_lb_log)
    _fh.setLevel(_logging.WARNING)
    _fh.setFormatter(_logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_fh)
    logger.info("LB proxy 诊断日志 → %s", _lb_log)
except Exception as _e:  # noqa: BLE001
    logger.warning("LB proxy 落盘日志初始化失败: %s", _e)

# Add uvloop for faster event loop if available
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


# [2026-08-17] 新 session 的引擎选择策略:least_load(默认,选最闲) | hash(旧行为,md5 取模)
_SESSION_POLICY = os.environ.get("VIME_LB_SESSION_POLICY", "least_load").strip().lower()
# session 映射的存活期与清理间隔。session 实测 54 分钟/场、轮次间可能有分钟级空档,
#   TTL 取 2 小时留足余量(误清只会让该 session 换引擎、丢一次前缀缓存,不会出错)。
_SESSION_TTL = float(os.environ.get("VIME_LB_SESSION_TTL", "7200"))
_SESSION_PRUNE_INTERVAL = 60.0
# Live load is sampled only while binding a previously unseen session.  A short
# timeout keeps a broken metrics endpoint from delaying rollout admission; the
# existing capacity-aware estimate remains the fail-open path.
_LIVE_METRICS_TIMEOUT = float(os.environ.get("VIME_LB_METRICS_TIMEOUT", "2.0"))
_LIVE_METRICS_RETRY_INTERVAL = float(os.environ.get("VIME_LB_METRICS_RETRY_INTERVAL", "5.0"))
_LIVE_METRICS_WARNING_INTERVAL = 60.0
_CAPACITY_RETRY_INTERVAL = float(os.environ.get("VIME_LB_CAPACITY_RETRY_INTERVAL", "30.0"))
_SESSION_QUOTA_WARNING_INTERVAL = 60.0


@dataclass
class SessionBinding:
    server_idx: int
    last_seen: float
    estimated_kv_tokens: float


@dataclass(frozen=True)
class LiveServerLoad:
    kv_cache_usage: float
    running_requests: float
    waiting_requests: float


class ServerState:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/v1"
        self.client = httpx.AsyncClient(
            timeout=None,
            base_url=self.url,
            limits=httpx.Limits(max_connections=100000, max_keepalive_connections=100000),
        )
        self.active_tokens = 0
        self.active_requests = 0
        # ``/server_info`` 在 vLLM engine ready 后回填 num_gpu_blocks 和
        # block_size。两者的乘积不是混合注意力模型的绝对 token 容量，但同一
        # 模型各 DP 引擎间与实际 KV 容量成比例，足够用于归一化负载。
        self.kv_capacity_units = 1.0
        # [2026-08-17] 钉在本引擎的活跃 session 数。active_tokens 只反映"此刻在飞的请求",
        #   而 polar 的 session 是 40+ 轮 / 54 分钟的长驻体、轮次之间有大段空档(编译/跑测试),
        #   那时 active_tokens 归零但 session 仍会回来。给新 session 选引擎要看这个。
        self.active_sessions = 0
        self.estimated_session_kv_tokens = 0.0
        self.aborted_requests = set()  # Track aborted requests


class ProxyState:
    def __init__(
        self,
        server_instances,
        capacity_units: list[float] | None = None,
        max_active_sessions: int | None = None,
    ):
        if max_active_sessions is not None and max_active_sessions <= 0:
            raise ValueError("max_active_sessions must be greater than 0")
        self.dp_servers: list[ServerState] = [ServerState(h, p) for h, p in server_instances]
        self.max_active_sessions = max_active_sessions
        # Capacity-weighted upper bounds for first-time sticky bindings.  They are
        # enabled only after every backend supplied a real /server_info capacity;
        # equal fallback values must never masquerade as authoritative capacity for
        # a heterogeneous colocated+dedicated rollout pool.
        self.session_quotas: list[int] | None = None
        self.req_id_lock = asyncio.Lock()

        # Initialize priority queues for efficient server selection
        # Each entry is (priority_score, server_index, server_reference)
        # Lower priority score = higher priority (less loaded)
        self.lb_heap = [(0, i, server) for i, server in enumerate(self.dp_servers)]
        heapq.heapify(self.lb_heap)
        # session_id -> (server_idx, last_seen_monotonic)。首次出现时按负载选引擎并钉住,
        #   之后同 session 恒定复用 → 前缀 KV 命中(实测 87-95%)靠它。
        self.session_map: dict[str, SessionBinding] = {}
        self._last_prune = time.monotonic()
        # Only first-time session bindings enter this lock. Existing sessions keep
        # their zero-extra-I/O affinity path while concurrent new sessions cannot all
        # observe the same low-KV backend and stampede it.
        self._new_session_lock = asyncio.Lock()
        self._live_metrics_retry_after = 0.0
        self._last_live_metrics_warning = float("-inf")
        self._last_session_quota_warning = float("-inf")
        if capacity_units is not None:
            self.set_capacity_units(capacity_units)

    @staticmethod
    def _allocate_capacity_quotas(capacity_units: list[float], total_sessions: int) -> list[int]:
        """Apportion an exact global session limit by KV-capacity weight.

        Hamilton/largest-remainder apportionment preserves the configured global
        limit while avoiding hard-coded host or gpu_memory_utilization assumptions.
        """
        total_capacity = float(sum(capacity_units))
        exact = [total_sessions * float(capacity) / total_capacity for capacity in capacity_units]
        quotas = [math.floor(value) for value in exact]
        remaining = total_sessions - sum(quotas)
        order = sorted(
            range(len(capacity_units)),
            key=lambda i: (-(exact[i] - quotas[i]), i),
        )
        for i in order[:remaining]:
            quotas[i] += 1
        return quotas

    def set_capacity_units(self, capacity_units: list[float], *, authoritative: bool = True) -> None:
        if len(capacity_units) != len(self.dp_servers):
            raise ValueError(
                f"capacity count ({len(capacity_units)}) does not match server count ({len(self.dp_servers)})"
            )
        if any(not isinstance(value, (int, float)) or value <= 0 for value in capacity_units):
            raise ValueError(f"all KV capacity units must be positive: {capacity_units!r}")
        for server, capacity in zip(self.dp_servers, capacity_units, strict=True):
            server.kv_capacity_units = float(capacity)
        self.lb_heap = [
            (server.active_tokens / server.kv_capacity_units, i, server)
            for i, server in enumerate(self.dp_servers)
        ]
        heapq.heapify(self.lb_heap)
        if authoritative and self.max_active_sessions is not None:
            self.session_quotas = self._allocate_capacity_quotas(
                [server.kv_capacity_units for server in self.dp_servers],
                self.max_active_sessions,
            )
        else:
            self.session_quotas = None

    async def discover_kv_capacities(self) -> bool:
        """Read initialized KV capacity from vLLM's existing ``/server_info``.

        Capacity discovery is deliberately all-or-nothing. Mixing a real block count
        from one backend with the default value from another would create a much worse
        imbalance than equal weighting, so any missing/unhealthy response falls back to
        equal capacity for every backend.
        """

        async def _read(server: ServerState) -> float:
            url = f"http://{server.host}:{server.port}/server_info"
            response = await server.client.get(url, params={"config_format": "json"}, timeout=30.0)
            response.raise_for_status()
            return _kv_capacity_units_from_server_info(response.json())

        results = await asyncio.gather(*(_read(server) for server in self.dp_servers), return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            self.set_capacity_units([1.0] * len(self.dp_servers), authoritative=False)
            logger.warning(
                "KV capacity discovery failed for %d/%d backends; falling back to equal weights "
                "with session quotas disabled: %s",
                len(failures),
                len(self.dp_servers),
                failures,
            )
            return False

        capacities = [float(result) for result in results]
        self.set_capacity_units(capacities)
        logger.info(
            "Discovered backend KV capacity units: %s; max_active_sessions=%s session_quotas=%s",
            capacities,
            self.max_active_sessions,
            self.session_quotas,
        )
        return True

    async def read_live_server_loads(self) -> list[LiveServerLoad] | None:
        """Read current vLLM scheduler/KV gauges for a new-session decision.

        No successful result is cached: every newly bound session gets a fresh
        snapshot.  On a metrics failure, a short circuit-breaker interval prevents
        a backend outage from serially adding the full timeout to a burst of new
        sessions; callers then use the existing estimate-based fallback.
        """

        now = time.monotonic()
        if now < self._live_metrics_retry_after:
            return None

        async def _read(server: ServerState) -> LiveServerLoad:
            url = f"http://{server.host}:{server.port}/metrics"
            response = await server.client.get(url, timeout=_LIVE_METRICS_TIMEOUT)
            response.raise_for_status()
            return _live_server_load_from_prometheus(response.text)

        results = await asyncio.gather(*(_read(server) for server in self.dp_servers), return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            self._live_metrics_retry_after = now + _LIVE_METRICS_RETRY_INTERVAL
            if now - self._last_live_metrics_warning >= _LIVE_METRICS_WARNING_INTERVAL:
                self._last_live_metrics_warning = now
                logger.warning(
                    "Live KV metrics failed for %d/%d backends; using capacity-aware estimate until retry: %s",
                    len(failures),
                    len(self.dp_servers),
                    failures,
                )
            return None

        self._live_metrics_retry_after = 0.0
        return [result for result in results if isinstance(result, LiveServerLoad)]

    def _update_server_priority(self, server_idx: int):
        """Update the priority of a decoder server in the heap."""
        server = self.dp_servers[server_idx]
        priority = server.active_tokens / server.kv_capacity_units
        # Remove old entry and add new one
        self.lb_heap = [(p, i, s) for p, i, s in self.lb_heap if i != server_idx]
        heapq.heappush(self.lb_heap, (priority, server_idx, server))  # type: ignore

    async def next_req_id(self):
        async with self.req_id_lock:
            return str(uuid.uuid4())

    def select_server(self, token_count):  # Changed to synchronous
        # No lock needed - entire function is atomic
        if not self.lb_heap:
            raise RuntimeError("No decoder servers available")

        priority, chosen, server = heapq.heappop(self.lb_heap)

        # Update the chosen server atomically
        self.dp_servers[chosen].active_tokens += token_count
        self.dp_servers[chosen].active_requests += 1

        # Update priority and re-add to heap
        self._update_server_priority(chosen)

        return chosen

    def _maybe_prune_sessions(self, now: float) -> None:
        """惰性清理过期 session 映射(最多每 _SESSION_PRUNE_INTERVAL 秒一次)。
        不清理的话 session_map 会随 run 无界增长,active_sessions 也会只增不减、失去均衡意义。"""
        if now - self._last_prune < _SESSION_PRUNE_INTERVAL:
            return
        self._last_prune = now
        stale = [sid for sid, binding in self.session_map.items() if now - binding.last_seen > _SESSION_TTL]
        for sid in stale:
            binding = self.session_map.pop(sid)
            server = self.dp_servers[binding.server_idx]
            if server.active_sessions > 0:
                server.active_sessions -= 1
            server.estimated_session_kv_tokens = max(
                0.0,
                server.estimated_session_kv_tokens - binding.estimated_kv_tokens,
            )
        if stale:
            logger.debug("Pruned %d stale session mappings", len(stale))

    def _reserve_existing_session(
        self,
        session_id: str,
        token_count: float,
        entry: SessionBinding,
        now: float,
    ) -> int:
        idx = entry.server_idx
        # 每轮请求携带完整增长后的上下文。用本 session 见过的最大请求估算
        # 其可复用 KV footprint；上下文偶尔截断时不立即低估仍留在缓存中的旧块。
        estimated_kv_tokens = max(entry.estimated_kv_tokens, float(token_count))
        server = self.dp_servers[idx]
        server.estimated_session_kv_tokens += estimated_kv_tokens - entry.estimated_kv_tokens
        self.session_map[session_id] = SessionBinding(idx, now, estimated_kv_tokens)
        server.active_tokens += token_count
        server.active_requests += 1
        self._update_server_priority(idx)
        return idx

    def _choose_new_session_server(
        self,
        token_count: float,
        live_loads: list[LiveServerLoad] | None,
    ) -> int:
        candidates = list(range(len(self.dp_servers)))
        if self.session_quotas is not None:
            under_quota = [
                i
                for i, server in enumerate(self.dp_servers)
                if server.active_sessions < self.session_quotas[i]
            ]
            if under_quota:
                candidates = under_quota
            else:
                # Global admission should normally prevent this.  Never reject or
                # deadlock a request if terminal release is delayed: fail open to the
                # pressure score, but leave an actionable diagnostic.
                now = time.monotonic()
                if now - self._last_session_quota_warning >= _SESSION_QUOTA_WARNING_INTERVAL:
                    self._last_session_quota_warning = now
                    logger.warning(
                        "All capacity-weighted session quotas are full; routing new session "
                        "by pressure (active_sessions=%s quotas=%s configured_max=%s)",
                        [server.active_sessions for server in self.dp_servers],
                        self.session_quotas,
                        self.max_active_sessions,
                    )

        if live_loads is not None:
            if len(live_loads) != len(self.dp_servers):
                raise ValueError(
                    f"live load count ({len(live_loads)}) does not match server count ({len(self.dp_servers)})"
                )

            # Avoid adding live and sticky pressure: both can describe the same KV
            # blocks.  Their maximum is the dominant resource pressure. active_tokens
            # remains a short-lived live-metrics-lag reservation, while the sticky
            # estimate prevents an idle long-running session from disappearing from
            # the primary score between turns.
            def _live_key(i: int):
                server = self.dp_servers[i]
                capacity = server.kv_capacity_units
                physical_pressure = live_loads[i].kv_cache_usage + server.active_tokens / capacity
                sticky_pressure = server.estimated_session_kv_tokens / capacity
                projected_pressure = max(physical_pressure, sticky_pressure) + token_count / capacity
                return (
                    projected_pressure,
                    live_loads[i].waiting_requests,
                    live_loads[i].running_requests + server.active_requests,
                    server.active_sessions,
                    i,
                )

            return min(
                candidates,
                key=_live_key,
            )

        # Metrics unavailable: use the same dominant-pressure rule with the
        # capacity-aware proxy counters. Capacity quota protection remains active
        # only when /server_info discovery was authoritative.
        return min(
            candidates,
            key=lambda i: (
                max(
                    self.dp_servers[i].estimated_session_kv_tokens,
                    self.dp_servers[i].active_tokens,
                )
                / self.dp_servers[i].kv_capacity_units
                + token_count / self.dp_servers[i].kv_capacity_units,
                self.dp_servers[i].active_sessions,
                i,
            ),
        )

    async def select_server_by_session(self, session_id: str, token_count: float) -> int:
        """Session 亲和:同一 session_id(vime 每条 rollout sample 一个)恒定钉在同一引擎,
        多轮请求复用前缀 KV —— 对齐 vime/slime 上游 router_policy=consistent_hash 的效果。

        [2026-08-17] 首次分配从"md5 取模"改成"选当前最闲的引擎"。
          原实现一律 md5 %N,完全不看负载,而 _select_instance 只要带 x-session-id 就走这条 →
          下面那个基于最小堆的 select_server()(真正的负载感知)永远执行不到。
          小样本下 md5 分布不均:实测 6 引擎 / 每轮 32 session,累计完成量极差 2.9x、
          变异系数 0.36,且长会话(54 分钟/40+ 轮)会把一次偏斜放大成整场偏斜。
          亲和性不受影响 —— 只改"第一次选谁",选定后照样钉死。
          置 VIME_LB_SESSION_POLICY=hash 可回退到旧的纯哈希行为。

        [2026-09-01] least_load 首次分配读取各 vLLM /metrics 的真实 KV 使用率，
          同时保留 active_tokens 作为尚未反映到指标中的请求预留量。指标失败时回退到
          原有的容量归一化 session 估算；已经绑定的 session 不读取指标也不迁移。

        [2026-09-02] 将长期 session 负债提升为主压力，并按 /server_info 的实际 KV
          容量分摊全局 active-session 上限。防止当前 KV 刚释放但仍绑定大量长 session
          的引擎继续吸入新会话；配额全满时只告警并按压力 fail-open，不拒绝请求。
        """
        now = time.monotonic()
        self._maybe_prune_sessions(now)
        entry = self.session_map.get(session_id)
        if entry is not None:
            # Existing sessions never query /metrics and never change engines.
            return self._reserve_existing_session(session_id, token_count, entry, now)

        # Serialize only first-time bindings. This avoids duplicate metric reads for
        # concurrent turns of the same new session and makes each reservation visible
        # before the next new session chooses an engine. Existing sessions bypass it.
        async with self._new_session_lock:
            now = time.monotonic()
            self._maybe_prune_sessions(now)
            entry = self.session_map.get(session_id)
            if entry is not None:
                return self._reserve_existing_session(session_id, token_count, entry, now)

            live_loads = None
            if _SESSION_POLICY == "hash":
                idx = int(hashlib.md5(session_id.encode("utf-8")).hexdigest(), 16) % len(self.dp_servers)
            else:
                live_loads = await self.read_live_server_loads()
                idx = self._choose_new_session_server(token_count, live_loads)

            server = self.dp_servers[idx]
            estimated_kv_tokens = float(token_count)
            server.active_sessions += 1
            server.estimated_session_kv_tokens += estimated_kv_tokens
            self.session_map[session_id] = SessionBinding(idx, now, estimated_kv_tokens)
            server.active_tokens += token_count
            server.active_requests += 1
            self._update_server_priority(idx)
            logger.debug(
                "New session %s -> server %d (policy=%s, live_kv=%s, sessions=%s, quotas=%s, "
                "estimated_kv_pressure=%s)",
                session_id,
                idx,
                _SESSION_POLICY,
                None if live_loads is None else [round(load.kv_cache_usage, 4) for load in live_loads],
                [sv.active_sessions for sv in self.dp_servers],
                self.session_quotas,
                [round(sv.estimated_session_kv_tokens / sv.kv_capacity_units, 4) for sv in self.dp_servers],
            )
        return idx

    def release_server(self, idx: int, token_count):  # Changed to synchronous
        # No lock needed - atomic operation. Clamp defensively so cancellation/error
        # races cannot poison all future load-balancing decisions with a negative load.
        server = self.dp_servers[idx]
        server.active_tokens = max(0.0, server.active_tokens - token_count)
        server.active_requests = max(0, server.active_requests - 1)
        # Update priority queue after releasing
        self._update_server_priority(idx)

    def clear_sticky_cache(self) -> dict[str, Any]:
        """Clear only proxy routing bookkeeping at a drained policy boundary."""
        busy = [i for i, server in enumerate(self.dp_servers) if server.active_requests]
        if busy:
            raise RuntimeError(f"cannot clear sticky routing while backends still have active requests: {busy}")

        cleared = len(self.session_map)
        self.session_map.clear()
        for server in self.dp_servers:
            server.active_sessions = 0
            server.estimated_session_kv_tokens = 0.0
        self._last_prune = time.monotonic()
        return {"status": "ok", "cleared_sessions": cleared}

    def release_sticky_session(self, session_id: str) -> dict[str, Any]:
        """Forget one terminal session without disturbing other affinities.

        Polar calls this only after the session has entered its terminal cleanup
        path and all future generation for the ID has been closed.  The operation
        is deliberately idempotent so callback retries and policy-boundary cleanup
        can race safely.
        """
        binding = self.session_map.pop(session_id, None)
        if binding is None:
            return {"status": "ok", "released": False, "session_id": session_id}

        server = self.dp_servers[binding.server_idx]
        server.active_sessions = max(0, server.active_sessions - 1)
        server.estimated_session_kv_tokens = max(
            0.0,
            server.estimated_session_kv_tokens - binding.estimated_kv_tokens,
        )
        return {
            "status": "ok",
            "released": True,
            "session_id": session_id,
            "server_idx": binding.server_idx,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "sessions": len(self.session_map),
            "max_active_sessions": self.max_active_sessions,
            "session_quotas": self.session_quotas,
            "servers": [
                {
                    "host": server.host,
                    "port": server.port,
                    "kv_capacity_units": server.kv_capacity_units,
                    "session_quota": None if self.session_quotas is None else self.session_quotas[i],
                    "active_sessions": server.active_sessions,
                    "estimated_session_kv_tokens": server.estimated_session_kv_tokens,
                    "active_requests": server.active_requests,
                    "active_tokens": server.active_tokens,
                }
                for i, server in enumerate(self.dp_servers)
            ],
        }

    @staticmethod
    def estimate_prompt_tokens(req_data: Any, request_length: int) -> int:
        """Use token-id payloads when available, with a byte-size fallback."""

        def _count_token_ids(value: Any) -> int | None:
            if not isinstance(value, list):
                return None
            if all(isinstance(item, int) for item in value):
                return len(value)
            nested = [_count_token_ids(item) for item in value]
            if nested and all(count is not None for count in nested):
                return sum(int(count) for count in nested)
            return None

        if isinstance(req_data, dict):
            for key in ("prompt_token_ids", "token_ids", "prompt"):
                count = _count_token_ids(req_data.get(key))
                if count is not None:
                    return max(1, count)
        # The proxy intentionally owns no tokenizer. Four UTF-8/JSON bytes per token
        # is only a fallback estimate; payloads carrying token ids take the exact path.
        return max(1, request_length // 4)

    def calculate_request_score(self, request_length: int, max_tokens: int = 16, ignore_eos: bool = False) -> float:
        if ignore_eos:
            return request_length + max_tokens
        else:
            # Note that 0.5 is an empirical value here because we don't know
            # the actual number of tokens generated before EOS.
            return request_length + 0.5 * max_tokens


proxy_state = None


def _live_server_load_from_prometheus(payload: str) -> LiveServerLoad:
    """Parse vLLM's existing Prometheus gauges without depending on log text."""

    metric_names = {
        "vllm:kv_cache_usage_perc",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
    }
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    for family in text_string_to_metric_families(payload):
        for sample in family.samples:
            if sample.name in values:
                values[sample.name].append(float(sample.value))

    missing = [name for name, samples in values.items() if not samples]
    if missing:
        raise ValueError(f"vLLM /metrics is missing required gauges: {sorted(missing)!r}")

    # An endpoint normally exposes one engine. max(KV) and sum(requests) also give
    # safe semantics if an endpoint exposes several engine-labelled samples.
    kv_cache_usage = max(values["vllm:kv_cache_usage_perc"])
    running_requests = sum(values["vllm:num_requests_running"])
    waiting_requests = sum(values["vllm:num_requests_waiting"])
    if not all(math.isfinite(value) for value in (kv_cache_usage, running_requests, waiting_requests)):
        raise ValueError("vLLM /metrics returned a non-finite live load gauge")
    if not 0.0 <= kv_cache_usage <= 1.0:
        raise ValueError(f"vLLM /metrics returned invalid KV usage: {kv_cache_usage}")
    if running_requests < 0.0 or waiting_requests < 0.0:
        raise ValueError(
            f"vLLM /metrics returned negative request counts: running={running_requests}, waiting={waiting_requests}"
        )
    return LiveServerLoad(
        kv_cache_usage=kv_cache_usage,
        running_requests=running_requests,
        waiting_requests=waiting_requests,
    )


def _kv_capacity_units_from_server_info(payload: Any) -> float:
    try:
        cache_config = payload["vllm_config"]["cache_config"]
        num_gpu_blocks = int(cache_config["num_gpu_blocks"])
        block_size = int(cache_config["block_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("/server_info is missing initialized cache_config capacity") from exc
    if num_gpu_blocks <= 0 or block_size <= 0:
        raise ValueError(
            f"/server_info returned invalid KV capacity: num_gpu_blocks={num_gpu_blocks}, block_size={block_size}"
        )
    return float(num_gpu_blocks * block_size)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--dp-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--dp-ports", type=int, nargs="+", default=[8001])
    parser.add_argument(
        "--max-active-sessions",
        type=int,
        default=None,
        help="Global active-session limit used for capacity-weighted sticky quotas",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for HTTP requests")
    parser.add_argument(
        "--retry-delay", type=float, default=0.001, help="Base delay (seconds) for exponential backoff retries"
    )
    args = parser.parse_args()
    if len(args.dp_hosts) != len(args.dp_ports):
        raise ValueError("Number of dp hosts must match number of dp ports")
    if args.max_active_sessions is not None and args.max_active_sessions <= 0:
        raise ValueError("max-active-sessions must be greater than 0")
    args.server_instances = list(zip(args.dp_hosts, args.dp_ports))
    return args


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy_state
    proxy_state = ProxyState(
        global_args.server_instances,
        max_active_sessions=global_args.max_active_sessions,
    )
    capacity_retry_task = None
    if _SESSION_POLICY != "hash":
        capacity_ready = await proxy_state.discover_kv_capacities()
        if not capacity_ready:
            async def _retry_capacity_discovery():
                while True:
                    await asyncio.sleep(_CAPACITY_RETRY_INTERVAL)
                    if await proxy_state.discover_kv_capacities():
                        return

            # Keep requests fail-open on equal weights while capacity discovery
            # retries independently.  A later success only steers future bindings;
            # existing sticky sessions are never migrated.
            capacity_retry_task = asyncio.create_task(_retry_capacity_discovery())
    print(f"Initialized {len(proxy_state.dp_servers)} dp server clients.")
    try:
        yield
    finally:
        if capacity_retry_task is not None:
            capacity_retry_task.cancel()
            await asyncio.gather(capacity_retry_task, return_exceptions=True)
        for p in proxy_state.dp_servers:
            await p.client.aclose()


async def listen_for_disconnect(request: Request) -> None:
    """Return if a disconnect message is received"""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


def with_cancellation(handler_func):
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        request = kwargs["request"]
        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))
        done, pending = await asyncio.wait([handler_task, cancellation_task], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper


app = FastAPI(lifespan=lifespan)


async def _forward_upstream_with_retry(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
    extra_headers: dict | None = None,
) -> tuple[httpx.Response | None, tuple[int, bytes] | None]:
    """向上游 vLLM 转发并 retry;用 client.send(stream=True) 先拿响应头,据 status 决定:
      - 2xx: 返回 (response, None) —— 已确认可流式,body 由调用方 aiter_bytes 消费后 aclose。
      - 最终非 2xx / 连不上: 返回 (None, (status_code, body)) —— 调用方原样回传真实 status。

    ⚠️ 关键(对齐无 proxy 的 sglang router 行为): 上游 4xx/5xx 绝不能被吞成 200+空 body。
    否则 operator gateway 的 completion() 对空 body 做 resp.json() 抛 JSONDecodeError,而
    gateway 只 except UpstreamError → 失败 session 无法干净 errored → session_pool 达不到
    min_complete → rollout hang(agentic 长跑撞 context 上限时必现)。原样透传 status 后,
    gateway 的 _raise_for_status 会抛 UpstreamHTTPError → session 秒级 errored → drop-and-continue。
    """
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "X-Request-Id": request_id}
    if extra_headers:  # 透传 polar telemetry trace 头(x-polar-trace-id)等,供引擎侧 join
        headers.update({str(k): str(v) for k, v in extra_headers.items() if v})
    for attempt in range(1, max_retries + 1):
        req = client.build_request("POST", endpoint, json=req_data, headers=headers)
        try:
            response = await client.send(req, stream=True)
        except httpx.RequestError as e:
            if attempt < max_retries:
                logger.warning("Attempt %s failed connecting %s: %s", attempt, endpoint, e)
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
                continue
            logger.error("All %s attempts failed connecting %s: %s", max_retries, endpoint, e)
            body = json.dumps(
                {"error": {"message": f"upstream connect failed after {max_retries} attempts: {e}",
                           "type": "upstream_transport_error"}}
            ).encode()
            return None, (502, body)
        if response.status_code < 400:
            return response, None
        # 上游 4xx/5xx: 读错误体、关闭连接(下面据 attempt 决定 retry 还是原样透传)
        body = await response.aread()
        await response.aclose()
        if attempt < max_retries:
            logger.warning("Attempt %s got HTTP %s from %s", attempt, response.status_code, endpoint)
            await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            continue
        logger.error(
            "All %s attempts got HTTP %s from %s; propagating upstream status (not swallowing).",
            max_retries, response.status_code, endpoint,
        )
        return None, (response.status_code, body)
    return None, (502, json.dumps(
        {"error": {"message": "upstream retry exhausted", "type": "upstream_error"}}).encode())


async def _select_instance(api: str, req_data: Any, request_length: int, session_id: str | None = None):
    # refer to vLLM sampling_params: max_token default value
    req_dict = req_data if isinstance(req_data, dict) else {}
    sampling_params = req_dict.get("sampling_params", {})
    if not isinstance(sampling_params, dict):
        sampling_params = {}
    max_tokens = req_dict.get("max_tokens", req_dict.get("max_completion_tokens"))
    if max_tokens is None:
        max_tokens = sampling_params.get("max_tokens", sampling_params.get("max_new_tokens"))
    try:
        max_tokens = int(max_tokens) if max_tokens is not None else 16
    except (TypeError, ValueError):
        max_tokens = 16
    ignore_eos = req_dict.get("ignore_eos", sampling_params.get("ignore_eos", False))
    prompt_tokens = proxy_state.estimate_prompt_tokens(req_data, request_length)
    priority_score = proxy_state.calculate_request_score(
        prompt_tokens,
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
    )
    logger.debug(
        "Request bytes: %s, estimated prompt tokens: %s, max tokens: %s, ignore_eos: %s, "
        "session: %s, Priority score: %s",
        request_length,
        prompt_tokens,
        max_tokens,
        ignore_eos,
        session_id,
        priority_score,
    )
    request_id = await proxy_state.next_req_id()
    if session_id:
        # session 亲和(对齐上游 consistent_hash 默认):同一 sample 的多轮钉同一引擎、复用前缀 KV
        server_idx = await proxy_state.select_server_by_session(session_id, priority_score)
    else:
        # 无 x-session-id → 退回 active_tokens 最小负载(= vLLM 官方 DP proxy 同款)
        server_idx = proxy_state.select_server(priority_score)
    chosen_server = proxy_state.dp_servers[server_idx]
    logger.debug("Choose server %s to process request %s", chosen_server.url, request_id)
    return InstanceInfo(
        request_id=request_id, server_idx=server_idx, priority_score=priority_score, server_state=chosen_server
    )


@dataclass
class InstanceInfo:
    request_id: str
    server_idx: int
    priority_score: float
    server_state: ServerState


async def _handle_completions(api: str, request: Request):
    instance_info = None
    response = None
    release_in_handler = True
    try:
        req_data = await request.json()
        req_body = await request.body()
        request_length = len(req_body)
        # vime 约定:consistent_hash 路由用 x-session-id header(vllm_rollout / agent adapters 发)。
        # 有则 session 亲和,无则负载均衡。ITEM 1 的 token 保真靠下方 req_data 原样 dict 转发保证
        # (不像 Rust router 的 typed 解析会丢 return_token_ids)。
        session_id = request.headers.get("x-session-id")
        instance_info = await _select_instance(api, req_data, request_length, session_id=session_id)

        response, error = await _forward_upstream_with_retry(
            instance_info.server_state.client,
            api,
            req_data,
            request_id=instance_info.request_id,
            max_retries=global_args.max_retries,
            base_delay=global_args.retry_delay,
            extra_headers={"x-polar-trace-id": request.headers.get("x-polar-trace-id")},
        )
        if error is not None:
            # 上游最终非 2xx: 原样回传真实 status + 错误体,绝不吞成 200+空 body(见
            # _forward_upstream_with_retry)。令 operator gateway 走 UpstreamError → session
            # 秒级 errored → drop-and-continue,而不是 resp.json() 崩 JSONDecodeError → hang。
            status_code, body = error
            return Response(content=body, status_code=status_code, media_type="application/json")

        async def generate_stream():
            # 2xx 已确认;流式透传 body(stream=False 时上游返完整 JSON,一样字节透传)。
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            except Exception as e:
                logger.error(
                    "Streaming interrupted after response started from %s: %s, request %s.",
                    instance_info.server_state.url,
                    e,
                    instance_info.request_id,
                )
            finally:
                await response.aclose()
                # After streaming done, release tokens
                proxy_state.release_server(instance_info.server_idx, instance_info.priority_score)

        streaming_response = StreamingResponse(generate_stream(), media_type="application/json")
        # From here the stream generator owns response close + load release.
        release_in_handler = False
        return streaming_response
    except Exception as e:
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in external dp proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise
    finally:
        # The client can disconnect while we are still connecting/waiting for upstream
        # headers. The original handler leaked active_tokens in that path forever.
        if release_in_handler and instance_info is not None:
            if response is not None:
                await response.aclose()
            proxy_state.release_server(instance_info.server_idx, instance_info.priority_score)


@app.post("/v1/completions")
@with_cancellation
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
@with_cancellation
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "dp_instances": len(proxy_state.dp_servers),
    }


@app.get("/health")
async def health():
    # vllm 标准就绪探针路径。polar(sglang_router_url)按 vllm worker 语义探 /health;原 example 只有
    # /healthcheck,单机直连无碍,接 polar 必须补 /health 返 200,否则端点被判 down → 不发生成请求 →
    # rollout 饿死(与 PD proxy 同一坑,见 pd_mooncake_proxy_server.py 的 /health)。
    return {"status": "ok", "dp_instances": len(proxy_state.dp_servers)}


@app.get("/vime/lb_state")
async def lb_state():
    """Expose routing state for load-distribution diagnostics."""
    return proxy_state.snapshot()


@app.post("/vime/release_sticky_session")
async def release_sticky_session(request: Request):
    """Release one Polar session's sticky bookkeeping after terminal cleanup."""
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="request body must be valid JSON") from exc
    session_id = payload.get("session_id") if isinstance(payload, dict) else None
    if not isinstance(session_id, str) or not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id must be a non-empty string")
    session_id = session_id.strip()
    if len(session_id) > 4096:
        raise HTTPException(status_code=400, detail="session_id is too long")

    result = proxy_state.release_sticky_session(session_id)
    logger.info(
        "Released terminal sticky session %s: released=%s server_idx=%s",
        session_id,
        result["released"],
        result.get("server_idx"),
    )
    return result


@app.post("/vime/clear_sticky_cache")
async def clear_sticky_cache(policy_version: int | None = None):
    """Clear stale session affinity after the serving policy has advanced."""
    try:
        result = proxy_state.clear_sticky_cache()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    result["policy_version"] = policy_version
    logger.info(
        "Cleared %d sticky sessions at policy_version=%s",
        result["cleared_sessions"],
        policy_version,
    )
    return result


if __name__ == "__main__":
    global global_args
    global_args = parse_args()
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
