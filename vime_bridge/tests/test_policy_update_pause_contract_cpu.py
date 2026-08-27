from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

import vime_bridge.rollout as rollout


def _args(**overrides):
    values = {
        "polar_url": "http://polar-rollout:8180",
        "polar_rollout_url": None,
        "polar_gateway_url": None,
        "polar_weight_update_pause_timeout": 300.0,
        "polar_gateway_control_timeout": 30.0,
        "polar_weight_update_drain_sessions": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _Response:
    def __init__(self, payload, *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.request = httpx.Request("POST", "http://polar/control")
        self.text = str(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "control failure",
                request=self.request,
                response=httpx.Response(
                    self.status_code,
                    request=self.request,
                    text=self.text,
                ),
            )

    def json(self):
        return self._payload


def _install_response(monkeypatch, response: _Response, calls: list) -> None:
    class _Client:
        def __init__(self, *, timeout) -> None:
            calls.append(("timeout", timeout))

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url, params=None):
            calls.append((url, params))
            return response

    monkeypatch.setattr(rollout.httpx, "Client", _Client)


def test_pause_accepts_closed_admission_before_engine_abort(monkeypatch) -> None:
    calls = []
    _install_response(
        monkeypatch,
        _Response({"all_paused": True, "all_drained": False, "inflight": 18}),
        calls,
    )

    status = rollout._pause_gateway_generation(_args(), require_drained=False)

    assert status["all_paused"] is True
    assert status["all_drained"] is False
    assert status["inflight"] == 18
    assert calls[-1] == (
        "http://polar-rollout:8180/rollout/admin/inference/pause",
        {"timeout_seconds": 300.0},
    )


def test_pause_requires_drained_after_engine_abort(monkeypatch) -> None:
    _install_response(
        monkeypatch,
        _Response({"all_paused": True, "all_drained": False, "inflight": 2}),
        [],
    )

    with pytest.raises(rollout.PolarRolloutSchedulerError, match="remained non-drained"):
        rollout._pause_gateway_generation(
            _args(),
            timeout_seconds=30.0,
            require_drained=True,
        )


def test_httpx_status_error_is_converted_before_crossing_ray(monkeypatch) -> None:
    _install_response(
        monkeypatch,
        _Response({"detail": "gateway unavailable"}, status_code=502),
        [],
    )

    with pytest.raises(rollout.PolarRolloutSchedulerError, match="status=502") as exc_info:
        rollout._pause_gateway_generation(_args(), require_drained=False)

    assert not isinstance(exc_info.value, httpx.HTTPStatusError)


class _Worker:
    def __init__(self) -> None:
        self.config = SimpleNamespace(scheduler_mode="session_pool")
        self.draining = False
        self.finished = False

    def begin_policy_update_drain(self, policy_version: int) -> None:
        assert policy_version == 1
        self.draining = True

    def finish_policy_update_drain(self) -> None:
        self.finished = True
        self.draining = False

    def update_policy_version(self, policy_version: int) -> None:
        assert policy_version == 1


def test_prepare_failure_does_not_resume_local_or_remote_admission(monkeypatch) -> None:
    worker = _Worker()
    resumed = []
    monkeypatch.setattr(rollout, "_global_async_worker", worker)
    monkeypatch.setattr(
        rollout,
        "_pause_gateway_generation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            rollout.PolarRolloutSchedulerError("pause failed")
        ),
    )
    monkeypatch.setattr(rollout, "_resume_gateway_generation", lambda args: resumed.append(True))

    with pytest.raises(rollout.PolarRolloutSchedulerError, match="pause failed"):
        rollout.prepare_policy_update(_args(), 1)

    assert worker.draining is True
    assert worker.finished is False
    assert resumed == []


def test_commit_requires_version_publish_success(monkeypatch) -> None:
    monkeypatch.setattr(
        rollout,
        "_pause_gateway_generation",
        lambda *args, **kwargs: {
            "all_paused": True,
            "all_drained": True,
            "inflight": 0,
        },
    )
    monkeypatch.setattr(rollout, "push_policy_version_to_gateway", lambda args, version: False)

    with pytest.raises(rollout.PolarRolloutSchedulerError, match="Failed to publish"):
        rollout.commit_policy_update_boundary(_args(), 1)


def test_finish_resume_failure_keeps_local_admission_closed(monkeypatch) -> None:
    worker = _Worker()
    worker.draining = True
    monkeypatch.setattr(rollout, "_global_async_worker", worker)
    monkeypatch.setattr(
        rollout,
        "_resume_gateway_generation",
        lambda args: (_ for _ in ()).throw(rollout.PolarRolloutSchedulerError("resume failed")),
    )

    with pytest.raises(rollout.PolarRolloutSchedulerError, match="resume failed"):
        rollout.finish_policy_update(_args(), 1)

    assert worker.draining is True
    assert worker.finished is False
