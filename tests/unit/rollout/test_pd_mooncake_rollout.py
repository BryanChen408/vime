import types

import pytest

from vime.ray import rollout as mod


class _FakeEngineHandle:
    def __init__(self, url: str | None = None, pd_endpoint: tuple[str, int] | None = None):
        self._url = url
        self._pd_endpoint = pd_endpoint
        self.get_url = types.SimpleNamespace(remote=lambda: self._url)
        self.get_pd_endpoint = types.SimpleNamespace(remote=lambda: self._pd_endpoint)


class _FakeServerGroup:
    def __init__(self, worker_type: str, engines):
        self.worker_type = worker_type
        self.engines = engines


@pytest.mark.unit
def test_mooncake_proxy_launch_uses_prefill_http_urls_only(monkeypatch):
    captured = {}

    def fake_start_mooncake_pd_proxy(_args, router_ip, router_port, prefill_urls, decode_urls):
        captured["router_ip"] = router_ip
        captured["router_port"] = router_port
        captured["prefill_urls"] = prefill_urls
        captured["decode_urls"] = decode_urls
        return "fake-proc"

    monkeypatch.setattr(mod, "_start_mooncake_pd_proxy", fake_start_mooncake_pd_proxy)
    monkeypatch.setattr(mod, "_MOONCAKE_PD_PROXY_PROCS", [])
    monkeypatch.setattr(
        mod,
        "ray",
        types.SimpleNamespace(get=lambda value: value),
    )

    router_ip = "80.48.5.56"
    router_port = 18000
    prefill_urls = []
    prefill_backend_urls = []
    decode_urls = []
    server_groups = [
        _FakeServerGroup(
            "prefill",
            [_FakeEngineHandle(pd_endpoint=("http://80.48.5.56:18100", 20001))],
        ),
        _FakeServerGroup(
            "decode",
            [_FakeEngineHandle(url="http://80.48.5.56:18200")],
        ),
    ]

    for g in server_groups:
        for e in g.engines:
            if e is None:
                continue
            if g.worker_type == "prefill":
                endpoint = mod.ray.get(e.get_pd_endpoint.remote())
                if endpoint is not None:
                    prefill_urls.append(endpoint)
                    prefill_backend_urls.append(endpoint[0])
            elif g.worker_type == "decode":
                url = mod.ray.get(e.get_url.remote())
                if url:
                    decode_urls.append(url)

    mod._MOONCAKE_PD_PROXY_PROCS.append(
        mod._start_mooncake_pd_proxy(None, router_ip, router_port, prefill_backend_urls, decode_urls)
    )

    assert captured["router_ip"] == "80.48.5.56"
    assert captured["router_port"] == 18000
    assert captured["prefill_urls"] == ["http://80.48.5.56:18100"]
    assert captured["decode_urls"] == ["http://80.48.5.56:18200"]
    assert mod._MOONCAKE_PD_PROXY_PROCS == ["fake-proc"]
