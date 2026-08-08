"""metrics 面板端口发现：不得骚扰 Mooncake 握手端口，也不得漏掉晚起的 engine。

背景（实跑 train_qwen36_polar_pd_20260808-103225.log）：面板扫描区间
15000-15200 和 Mooncake KV 握手端口重叠，往握手端口发 ``GET /metrics`` 会让
Mooncake 把 HTTP 头当二进制长度前缀读，在 vllm 日志里刷
``readString: too large length from socket: 8387229930220700999``
（小端解码即 ASCII ``GET /met``）+ ``SocketHandShakePlugin: malformed json``，
每 ~31 秒一轮。
"""

import errno
import importlib.util
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import requests


def _load_monitor_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "vllm_metrics_monitor_v2.py"
    )
    module_name = "test_vllm_metrics_monitor_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler 接口
        body = b"vllm:num_requests_running 2.0\nvllm:generation_tokens_total 99\n"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


def _reset(mod, *, lo, hi, internal=0, skip=frozenset()):
    mod.engines.clear()
    mod.engine_history.clear()
    mod._probe_strikes.clear()
    mod._probe_quarantine.clear()
    mod.config["discover_host"] = "127.0.0.1"
    mod.config["discover_ports"] = (lo, hi)
    mod.config["engine_internal_ports"] = internal
    mod.config["skip_ports"] = skip


@pytest.fixture
def mod():
    return _load_monitor_module()


@pytest.fixture
def metrics_server():
    """真 HTTP /metrics 端点。"""
    port = _free_port()
    httpd = HTTPServer(("127.0.0.1", port), _MetricsHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield port
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def mooncake_like():
    """裸 socket：accept、读字节、不回 HTTP —— Mooncake 握手端口的行为。

    实测真握手端口不会关连接，所以这里也保持连接开着，让 requests 读超时。
    """
    hits = {"n": 0}
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.listen(8)
    held = []
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            try:
                conn, _ = sock.accept()
            except OSError:
                return
            hits["n"] += 1
            held.append(conn)  # 持有不关，模拟握手端口

    threading.Thread(target=serve, daemon=True).start()
    yield port, hits
    stop.set()
    sock.close()
    for c in held:
        c.close()


@pytest.mark.unit
def test_engine_internal_window_quarantines_mooncake_port_permanently(mod, monkeypatch):
    """engine 的派生内部端口窗口内，非 metrics 端口应被永久隔离。"""
    api_port = 15000
    mc_port = 15002  # bootstrap = port+2，落在窗口 [+1, +5] 内
    _reset(mod, lo=api_port, hi=api_port + 6, internal=5)

    def fake_get(url, **kwargs):
        assert kwargs.get("proxies") == {"http": None, "https": None}
        if f":{api_port}/" in url:
            resp = requests.Response()
            resp.status_code = 200
            resp._content = b"vllm:num_requests_running 1.0\n"
            return resp
        if f":{mc_port}/" in url:
            raise requests.exceptions.ReadTimeout("read timed out")
        raise requests.exceptions.ConnectionError(
            OSError(errno.ECONNREFUSED, "Connection refused")
        )

    monkeypatch.setattr(mod.requests, "get", fake_get)

    found = mod.probe_engines()
    assert found == [f"http://127.0.0.1:{api_port}/metrics"]
    # 同一轮内即生效：端口升序扫描，API port 先于自己的内部端口被看到
    assert mod._probe_quarantine[mc_port] == float("inf")


@pytest.mark.unit
def test_refused_port_never_quarantined_so_late_engine_is_found(mod, metrics_server):
    """空端口不记 strike —— engine 是训练启动后才陆续 bind 的，必须每轮重试。"""
    port = metrics_server
    _reset(mod, lo=port, hi=port, internal=5)

    # engine 尚未起来：占住端口前先确认「未监听」路径不隔离
    _reset(mod, lo=_free_port(), hi=0, internal=5)
    empty_port = mod.config["discover_ports"][0]
    mod.config["discover_ports"] = (empty_port, empty_port)
    for _ in range(mod._PROBE_MAX_STRIKES + 3):
        assert mod.probe_engines() == []
    assert empty_port not in mod._probe_quarantine
    assert empty_port not in mod._probe_strikes

    # engine 起来后立刻能发现
    _reset(mod, lo=port, hi=port, internal=5)
    assert mod.probe_engines() == [f"http://127.0.0.1:{port}/metrics"]


@pytest.mark.unit
def test_listening_non_http_port_converges_to_quarantine(mod, mooncake_like):
    """listen 着但不返回 metrics 的端口，最多探 _PROBE_MAX_STRIKES 次就隔离。"""
    port, hits = mooncake_like
    _reset(mod, lo=port, hi=port, internal=0)  # 关闭窗口推导，只测负缓存

    for _ in range(mod._PROBE_MAX_STRIKES + 5):
        assert mod.probe_engines() == []

    assert hits["n"] == mod._PROBE_MAX_STRIKES, (
        f"应只被探 {mod._PROBE_MAX_STRIKES} 次，实际 {hits['n']} 次"
    )
    assert port in mod._probe_quarantine
    assert mod._probe_quarantine[port] != float("inf")  # 有 TTL，engine 重启可复用


@pytest.mark.unit
def test_quarantine_expiry_allows_rediscovery(mod, metrics_server):
    """隔离到期后重新给机会 —— engine 重启可能复用同一端口。"""
    port = metrics_server
    _reset(mod, lo=port, hi=port, internal=5)
    mod._probe_quarantine[port] = time.time() - 1.0  # 已到期
    assert mod.probe_engines() == [f"http://127.0.0.1:{port}/metrics"]
    assert port not in mod._probe_quarantine

    # 永久隔离不该到期
    mod.engines.clear()
    mod._probe_quarantine[port] = float("inf")
    assert mod.probe_engines() == []


@pytest.mark.unit
def test_skip_ports_are_never_contacted(mod, mooncake_like):
    port, hits = mooncake_like
    _reset(mod, lo=port, hi=port, internal=0, skip=frozenset({port}))
    mod.probe_engines()
    assert hits["n"] == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "spec,expected",
    [
        ("15002-15005,15076", {15002, 15003, 15004, 15005, 15076}),
        ("", set()),
        ("15001", {15001}),
        ("bad,15001", {15001}),  # 非法片段忽略，旁路能力不该因此挂掉
        ("15005-15002", {15002, 15003, 15004, 15005}),  # 反序区间
    ],
)
def test_parse_port_set(mod, spec, expected):
    assert mod._parse_port_set(spec) == frozenset(expected)


@pytest.mark.unit
def test_nobody_listening_classification(mod):
    """区分「没人 listen」(不记 strike) 和「有人但不说 HTTP」(记 strike)。"""
    assert mod._nobody_listening(requests.exceptions.ConnectTimeout()) is True
    # 连上却不应答：Mooncake 握手端口与卡死的 engine 同形，交给窗口/限次逻辑处理
    assert mod._nobody_listening(requests.exceptions.ReadTimeout()) is False

    # OSError 挂在 args 上（urllib3 的一种包装方式）
    wrapped = requests.exceptions.ConnectionError(
        OSError(errno.ECONNREFUSED, "Connection refused")
    )
    assert mod._nobody_listening(wrapped) is True

    # 真实 refused 异常链
    with pytest.raises(Exception) as ei:  # noqa: PT011 - 只为取到真实异常链
        requests.get(
            f"http://127.0.0.1:{_free_port()}/metrics",
            timeout=0.25,
            proxies={"http": None, "https": None},
        )
    assert mod._nobody_listening(ei.value) is True

    # 环形异常链不得死循环
    a, b = OSError("a"), OSError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert mod._nobody_listening(a) is False


@pytest.mark.unit
def test_real_run_port_layout_windows_do_not_swallow_next_engine(mod):
    """用实跑布局验证窗口边界。

    日志 rollout.py:1056 实际分配（tp4 → 窗口宽度 1+4=5）：
      engine 0: 15000, nccl 15001, bootstrap 15002-15005
      engine 1: 15074, nccl 15075, bootstrap 15076-15079
      engine 2: 15080, nccl 15081, bootstrap 15082-15085
    engine 1 的窗口必须止步 15079，不能吃掉 engine 2 的 API port 15080。
    """
    internal = 5
    api_ports = [15000, 15074, 15080]
    internal_ports = (
        list(range(15002, 15006))
        + list(range(15076, 15080))
        + list(range(15082, 15086))
        + [15001, 15075, 15081]
    )

    def in_window(port):
        return any(base < port <= base + internal for base in api_ports)

    for p in api_ports:
        assert not in_window(p), f"engine API port {p} 落入他人窗口，会被误杀"
    for p in internal_ports:
        assert in_window(p), f"内部端口 {p} 未被窗口覆盖"
    for p in (15006, 15073, 15086, 15200):
        assert not in_window(p), f"无关端口 {p} 被窗口覆盖"


@pytest.mark.unit
def test_fetch_metrics_bypasses_proxy(mod, monkeypatch):
    """engine 在本机/局域网，必须直连。继承 http_proxy 会让所有探测变成统一超时。"""
    seen = {}

    def fake_get(url, **kwargs):
        seen.update(kwargs)
        resp = requests.Response()
        resp.status_code = 200
        resp._content = b"vllm:num_requests_running 1.0\n"
        return resp

    monkeypatch.setattr(mod.requests, "get", fake_get)
    out = mod.fetch_metrics("http://127.0.0.1:15000/metrics")
    assert out == {"vllm:num_requests_running": 1.0}
    assert seen.get("proxies") == {"http": None, "https": None}
