"""Unit tests for ``vime.ray.lb_proxy``."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from vime.ray.lb_proxy import ProxyState, build_app, parse_args

ENGINES = [("10.0.0.1", 8100), ("10.0.0.2", 8101)]


class _FakeUpstream:
    """Answers every engine request from a scripted list of replies."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.requests = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        if isinstance(reply, Exception):
            raise reply
        status, payload = reply
        return httpx.Response(status, content=payload)


@pytest.fixture
def proxy(monkeypatch):
    """A client over the app, with every engine client swapped for a scripted transport."""

    def _build(replies, **kwargs):
        upstream = _FakeUpstream(replies)
        app = build_app(ENGINES, **{"retry_delay": 0.0, **kwargs})
        client = TestClient(app)
        client.__enter__()
        for server in app.state.proxy.dp_servers:
            server.client = httpx.AsyncClient(
                transport=httpx.MockTransport(upstream.handler), base_url=server.url
            )
        return client, app, upstream

    built = []

    def _factory(replies, **kwargs):
        result = _build(replies, **kwargs)
        built.append(result)
        return result

    yield _factory
    for client, _, _ in built:
        client.__exit__(None, None, None)


@pytest.mark.unit
def test_successful_reply_is_streamed_back(proxy):
    client, _, upstream = proxy([(200, b'{"choices": []}')])
    response = client.post("/v1/completions", json={"prompt": "hi"})
    assert response.status_code == 200
    assert response.content == b'{"choices": []}'
    assert len(upstream.requests) == 1


@pytest.mark.unit
def test_request_body_is_forwarded_verbatim(proxy):
    # The reason this proxy exists: a typed router would drop vLLM extensions such as
    # return_token_ids on the way through.
    client, _, upstream = proxy([(200, b"{}")])
    payload = {"prompt": "hi", "return_token_ids": True, "some_future_field": [1, 2]}
    client.post("/v1/completions", json=payload)
    assert json.loads(upstream.requests[0].content) == payload


@pytest.mark.unit
@pytest.mark.parametrize("status", [400, 404, 422, 500, 503])
def test_upstream_failure_status_is_propagated(proxy, status):
    # Never swallow a failure into an empty 200: the caller parses the body as JSON, and a
    # session that cannot be marked failed stalls the rollout waiting on it.
    client, _, _ = proxy([(status, b'{"error": "upstream said no"}')])
    response = client.post("/v1/completions", json={"prompt": "hi"})
    assert response.status_code == status
    assert response.json() == {"error": "upstream said no"}


@pytest.mark.unit
def test_failures_are_retried_before_being_propagated(proxy):
    client, _, upstream = proxy([(503, b"{}"), (503, b"{}"), (200, b'{"ok": true}')], max_retries=3)
    response = client.post("/v1/completions", json={"prompt": "hi"})
    assert response.status_code == 200
    assert len(upstream.requests) == 3


@pytest.mark.unit
def test_transport_errors_become_502(proxy):
    client, _, _ = proxy([httpx.ConnectError("refused")], max_retries=2)
    response = client.post("/v1/completions", json={"prompt": "hi"})
    assert response.status_code == 502
    assert "upstream connect failed" in response.json()["error"]["message"]


@pytest.mark.unit
def test_session_header_pins_requests_to_one_engine(proxy):
    client, app, _ = proxy([(200, b"{}")])
    seen = set()
    for _ in range(6):
        client.post("/v1/completions", json={"prompt": "hi"}, headers={"x-session-id": "sample-42"})
        seen.add(_busiest_or_none(app))
    assert seen == {None}, "load must be released after each turn"

    idx = app.state.proxy.select_server_by_session("sample-42", 0)
    for _ in range(5):
        assert app.state.proxy.select_server_by_session("sample-42", 0) == idx


@pytest.mark.unit
def test_load_is_released_on_every_path(proxy):
    client, app, _ = proxy([(200, b"{}"), (500, b"{}"), (200, b"{}")])
    for _ in range(3):
        client.post("/v1/completions", json={"prompt": "hello world"})
    assert [server.active_tokens for server in app.state.proxy.dp_servers] == [0, 0]
    assert [server.active_requests for server in app.state.proxy.dp_servers] == [0, 0]


@pytest.mark.unit
def test_requests_without_a_session_go_to_the_least_loaded_engine():
    state = ProxyState(ENGINES)
    assert state.select_server(10) == 0
    assert state.select_server(1) == 1
    assert state.select_server(1) == 1
    assert [server.active_tokens for server in state.dp_servers] == [10, 2]


@pytest.mark.unit
def test_new_sessions_prefer_fewer_active_requests_then_fewer_tokens():
    state = ProxyState(ENGINES)
    state.dp_servers[0].active_requests = 2
    state.dp_servers[0].active_tokens = 10
    state.dp_servers[1].active_requests = 1
    state.dp_servers[1].active_tokens = 100
    assert state.select_server_by_session("session-a", 1) == 1

    state.dp_servers[0].active_requests = 2
    state.dp_servers[0].active_tokens = 10
    state.dp_servers[1].active_requests = 2
    state.dp_servers[1].active_tokens = 100
    assert state.select_server_by_session("session-b", 1) == 0


@pytest.mark.unit
def test_ignore_eos_charges_the_whole_budget():
    state = ProxyState(ENGINES)
    assert state.score_request(100, max_tokens=40, ignore_eos=True) == 140
    assert state.score_request(100, max_tokens=40, ignore_eos=False) == 120


@pytest.mark.unit
@pytest.mark.parametrize("path", ["/health", "/healthcheck"])
def test_readiness_probes(proxy, path):
    client, _, _ = proxy([(200, b"{}")])
    response = client.get(path)
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "dp_instances": len(ENGINES)}


@pytest.mark.unit
def test_mismatched_host_and_port_counts_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        parse_args(["--dp-hosts", "a", "b", "--dp-ports", "1"])


def _busiest_or_none(app):
    busy = [i for i, server in enumerate(app.state.proxy.dp_servers) if server.active_tokens]
    return busy[0] if busy else None
