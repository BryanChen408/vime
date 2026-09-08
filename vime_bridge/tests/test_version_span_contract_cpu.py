from __future__ import annotations

from types import SimpleNamespace

import httpx

import vime_bridge.version_span as version_span


class _Response:
    def __init__(self, payload) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def _install(monkeypatch, payload) -> None:
    class _Client:
        def __init__(self, *, timeout) -> None:
            del timeout

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url, params=None):
            del url, params
            return _Response(payload)

    monkeypatch.setattr(httpx, "Client", _Client)


def test_rollout_policy_version_requires_all_gateways(monkeypatch) -> None:
    args = SimpleNamespace(
        polar_url="http://polar-rollout:8180",
        polar_rollout_url=None,
        polar_gateway_url=None,
        polar_gateway_control_timeout=30.0,
    )
    _install(monkeypatch, {"all_updated": False, "policy_version": 3})

    assert version_span.push_policy_version_to_gateway(args, 3) is False
    _install(monkeypatch, {"all_updated": True, "policy_version": 3})
    assert version_span.push_policy_version_to_gateway(args, 3) is True


def test_direct_policy_version_requires_exact_version(monkeypatch) -> None:
    args = SimpleNamespace(
        polar_url=None,
        polar_rollout_url=None,
        polar_gateway_url="http://polar-gateway:8100",
        polar_gateway_control_timeout=30.0,
    )
    _install(monkeypatch, {"policy_version": 2})

    assert version_span.push_policy_version_to_gateway(args, 3) is False
