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
        "polar_policy_transition_enabled": False,
        "polar_policy_control_timeout": 45.0,
        "polar_run_id": "test-run",
        "rollout_scheduler_mode": "session_pool",
        "rollout_function_path": "vime_bridge.rollout.generate_rollout_polar_async",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _transition_context(*, engines=None, kind="update"):
    args = _args(polar_policy_transition_enabled=True)
    return rollout._PolicyTransitionContext(
        transition_id=(
            "update-test-0-to-1" if kind == "update" else "bootstrap-test-0-to-0"
        ),
        policy_namespace=rollout._policy_namespace(args),
        from_epoch=0,
        to_epoch=1 if kind == "update" else 0,
        from_engine_versions=dict(engines or {}),
        kind=kind,
    )


def _transition_payload(context, phase, *, engine_versions=None):
    return {
        "transition_id": context.transition_id,
        "policy_namespace": context.policy_namespace,
        "kind": context.kind,
        "from_epoch": context.from_epoch,
        "to_epoch": context.to_epoch,
        "from_engine_versions": dict(context.from_engine_versions),
        "engine_versions": dict(engine_versions or {}),
        "phase": phase,
        "active_namespace": context.policy_namespace,
        "active_epoch": context.to_epoch if phase == "serving" else context.from_epoch,
    }


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
        {"timeout_seconds": 300.0, "wait_for_drain": False},
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


def test_rollout_resume_requires_every_gateway_ack(monkeypatch) -> None:
    _install_response(
        monkeypatch,
        _Response({"all_resumed": False, "nodes": [{"status": "error"}]}),
        [],
    )

    with pytest.raises(rollout.PolarRolloutSchedulerError, match="every gateway"):
        rollout._resume_gateway_generation(_args())


def test_direct_gateway_resume_requires_unpaused_state(monkeypatch) -> None:
    _install_response(monkeypatch, _Response({"paused": True}), [])

    with pytest.raises(rollout.PolarRolloutSchedulerError, match="did not confirm"):
        rollout._resume_gateway_generation(
            _args(polar_url=None, polar_gateway_url="http://polar-gateway:8100")
        )


class _Worker:
    def __init__(self) -> None:
        self.config = SimpleNamespace(scheduler_mode="session_pool")
        self.draining = False
        self.finished = False
        self.abandoned_before = None
        self.requested_before = None
        self.waited_before = None
        self.policy_version = 0

    def begin_policy_update_drain(self, policy_version: int) -> None:
        assert policy_version == 1
        self.draining = True

    def finish_policy_update_drain(self) -> None:
        self.finished = True
        self.draining = False

    def update_policy_version(self, policy_version: int) -> None:
        assert policy_version == 1
        self.policy_version = policy_version

    def current_policy_version(self) -> int:
        return self.policy_version

    def abandon_policy_versions_before(self, policy_version: int, *, timeout: float) -> None:
        assert timeout == 30.0
        self.abandoned_before = policy_version

    def request_policy_versions_before(self, policy_version: int) -> None:
        self.requested_before = policy_version

    def wait_policy_versions_before(self, policy_version: int, *, timeout: float) -> None:
        self.waited_before = (policy_version, timeout)


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
    worker = _Worker()
    monkeypatch.setattr(rollout, "_global_async_worker", worker)
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

    assert worker.abandoned_before == 1


def test_commit_applies_local_cutoff_before_version_publish(monkeypatch) -> None:
    calls = []
    worker = _Worker()
    monkeypatch.setattr(rollout, "_global_async_worker", worker)
    monkeypatch.setattr(
        rollout,
        "_pause_gateway_generation",
        lambda *args, **kwargs: calls.append(("drained", kwargs["require_drained"]))
        or {
            "all_paused": True,
            "all_drained": True,
            "inflight": 0,
        },
    )

    def abandon(policy_version: int, *, timeout: float) -> None:
        calls.append(("cutoff", policy_version, timeout))

    worker.abandon_policy_versions_before = abandon
    monkeypatch.setattr(
        rollout,
        "push_policy_version_to_gateway",
        lambda args, version: calls.append(("publish", version)) or True,
    )

    rollout.commit_policy_update_boundary(_args(), 1)

    assert calls == [
        ("cutoff", 1, 30.0),
        ("drained", True),
        ("publish", 1),
    ]


def test_commit_drain_mode_still_cuts_off_any_remaining_old_work(monkeypatch) -> None:
    worker = _Worker()
    monkeypatch.setattr(rollout, "_global_async_worker", worker)
    monkeypatch.setattr(
        rollout,
        "_pause_gateway_generation",
        lambda *args, **kwargs: {
            "all_paused": True,
            "all_drained": True,
            "inflight": 0,
        },
    )
    monkeypatch.setattr(rollout, "push_policy_version_to_gateway", lambda args, version: True)

    rollout.commit_policy_update_boundary(
        _args(polar_weight_update_drain_sessions=True),
        1,
    )

    assert worker.abandoned_before == 1


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


def test_transaction_prepare_only_closes_admission_before_engine_abort(monkeypatch) -> None:
    worker = _Worker()
    calls = []
    monkeypatch.setattr(rollout, "_global_async_worker", worker)
    monkeypatch.setattr(rollout, "_process_policy_transition", None)

    def post(args, path, *, json_payload, transition_id):
        calls.append((path, json_payload, transition_id))
        context = rollout._current_policy_transition()
        assert context is not None
        return _transition_payload(context, "admission_closed")

    monkeypatch.setattr(rollout, "_post_policy_control", post)
    status = rollout.prepare_policy_update(
        _args(polar_policy_transition_enabled=True),
        1,
        {"engine-000": "7", "engine-001": "7"},
    )

    assert status["all_paused"] is True
    assert status["all_drained"] is False
    assert worker.draining is True
    assert calls[0][0] == "/rollout/admin/policy-transitions/begin"
    assert calls[0][1]["engine_versions"] == {"engine-000": "7", "engine-001": "7"}
    assert calls[0][1]["policy_namespace"] == rollout._policy_namespace(
        _args(polar_policy_transition_enabled=True)
    )


def test_transaction_post_abort_boundary_does_not_wait_for_local_detach(monkeypatch) -> None:
    worker = _Worker()
    worker.draining = True
    monkeypatch.setattr(rollout, "_global_async_worker", worker)
    monkeypatch.setattr(
        rollout,
        "_process_policy_transition",
        _transition_context(engines={"engine-000": "7"}),
    )
    monkeypatch.setattr(
        rollout,
        "_post_policy_control",
        lambda *args, **kwargs: _transition_payload(
            rollout._current_policy_transition(),
            "ready_for_training",
        ),
    )

    status = rollout.commit_policy_update_boundary(
        _args(polar_policy_transition_enabled=True),
        1,
        True,
    )

    assert status["all_drained"] is True
    assert worker.requested_before == 1
    assert worker.waited_before is None
    assert worker.abandoned_before is None


def test_transaction_boundary_rejects_missing_engine_abort_proof(monkeypatch) -> None:
    monkeypatch.setattr(rollout, "_global_async_worker", None)
    monkeypatch.setattr(
        rollout,
        "_process_policy_transition",
        _transition_context(engines={"engine-000": "7"}),
    )
    with pytest.raises(
        rollout.PolarRolloutSchedulerError,
        match="all-engine abort acknowledgement",
    ):
        rollout.commit_policy_update_boundary(
            _args(polar_policy_transition_enabled=True),
            1,
        )


def test_transaction_finish_waits_local_fence_and_commits_all_engine_evidence(monkeypatch) -> None:
    worker = _Worker()
    worker.draining = True
    calls = []
    monkeypatch.setattr(rollout, "_global_async_worker", worker)
    monkeypatch.setattr(
        rollout,
        "_process_policy_transition",
        _transition_context(
            engines={"engine-000": "7", "engine-001": "7"}
        ),
    )

    def post(args, path, *, json_payload, transition_id):
        calls.append((path, json_payload, transition_id))
        context = rollout._current_policy_transition()
        assert context is not None
        return _transition_payload(
            context,
            "serving",
            engine_versions=json_payload["engine_versions"],
        )

    monkeypatch.setattr(rollout, "_post_policy_control", post)
    rollout.finish_policy_update(
        _args(polar_policy_transition_enabled=True),
        1,
        {"engine-000": "8", "engine-001": "8"},
    )

    assert worker.waited_before == (1, 45.0)
    assert worker.finished is True
    assert worker.policy_version == 1
    assert calls[0][1] == {
        "verified_policy_epoch": 1,
        "policy_namespace": rollout._policy_namespace(
            _args(polar_policy_transition_enabled=True)
        ),
        "engine_versions": {"engine-000": "8", "engine-001": "8"},
    }
    assert rollout._process_policy_transition is None


def test_transaction_rejects_mixed_engine_versions_before_control_post(monkeypatch) -> None:
    called = []
    monkeypatch.setattr(rollout, "_process_policy_transition", None)
    monkeypatch.setattr(rollout, "_post_policy_control", lambda *args, **kwargs: called.append(True))

    with pytest.raises(rollout.PolarRolloutSchedulerError, match="mixed weight versions"):
        rollout.prepare_policy_update(
            _args(polar_policy_transition_enabled=True),
            1,
            {"engine-000": "7", "engine-001": "8"},
        )

    assert called == []


def test_transaction_rejects_group_scheduler_without_local_epoch_cutoff() -> None:
    with pytest.raises(
        rollout.PolarRolloutSchedulerError,
        match="session_pool",
    ):
        rollout.prepare_initial_policy(
            _args(
                polar_policy_transition_enabled=True,
                rollout_scheduler_mode="group",
            ),
            0,
        )


def test_sync_transaction_uses_zero_inflight_instead_of_worker_cutoff(
    monkeypatch,
) -> None:
    args = _args(
        polar_policy_transition_enabled=True,
        rollout_scheduler_mode="group",
        rollout_function_path="vime_bridge.rollout.generate_rollout_polar_sync",
    )
    calls = []
    monkeypatch.setattr(rollout, "_global_async_worker", None)
    monkeypatch.setattr(rollout, "_process_policy_transition", None)

    def post(_args, path, *, json_payload, transition_id):
        context = rollout._current_policy_transition()
        assert context is not None
        calls.append((path, dict(json_payload), transition_id))
        return _transition_payload(context, "admission_closed")

    monkeypatch.setattr(rollout, "_post_policy_control", post)
    status = rollout.prepare_policy_update(
        args,
        6,
        {"engine-000": "5", "engine-001": "5"},
    )

    assert status["all_paused"] is True
    assert status["all_drained"] is False
    assert calls[0][0] == "/rollout/admin/policy-transitions/begin"
    assert calls[0][1] == {
        "transition_id": calls[0][2],
        "policy_namespace": rollout._policy_namespace(args),
        "from_epoch": 5,
        "to_epoch": 6,
        "engine_versions": {"engine-000": "5", "engine-001": "5"},
    }


def test_bootstrap_closes_before_first_weight_evidence_and_commits_namespace(monkeypatch) -> None:
    args = _args(polar_policy_transition_enabled=True)
    calls = []
    monkeypatch.setattr(rollout, "_global_async_worker", None)
    monkeypatch.setattr(rollout, "_process_policy_transition", None)
    monkeypatch.setattr(rollout, "_last_committed_policy", None)

    def post(_args, path, *, json_payload, transition_id):
        context = rollout._current_policy_transition()
        assert context is not None
        calls.append((path, dict(json_payload), transition_id))
        if path.endswith("/bootstrap/begin"):
            assert context.from_engine_versions == {}
            return _transition_payload(context, "admission_closed")
        if path.endswith("/confirm-drained"):
            return _transition_payload(context, "ready_for_training")
        if path.endswith("/commit"):
            return _transition_payload(
                context,
                "serving",
                engine_versions=json_payload["engine_versions"],
            )
        raise AssertionError(path)

    monkeypatch.setattr(rollout, "_post_policy_control", post)
    prepared = rollout.prepare_initial_policy(args, 0)
    assert prepared["all_paused"] is True
    rollout.commit_policy_update_boundary(args, 0, True)
    rollout.finish_initial_policy(args, 0, {"engine-000": "1"})

    namespace = rollout._policy_namespace(args)
    assert calls[0][0] == "/rollout/admin/policy/bootstrap/begin"
    assert calls[0][1] == {
        "transition_id": calls[0][2],
        "policy_namespace": namespace,
        "epoch": 0,
    }
    assert rollout._last_committed_policy == (calls[0][2], namespace, 0)
    assert rollout._process_policy_transition is None


def test_explicit_policy_control_4xx_is_not_reconciled(monkeypatch) -> None:
    class Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, *args, **kwargs):  # noqa: ANN201
            return _Response({"detail": "namespace mismatch"}, status_code=409)

    monkeypatch.setattr(rollout.httpx, "Client", Client)
    monkeypatch.setattr(
        rollout,
        "_get_policy_transition",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("explicit 4xx must not be reconciled")
        ),
    )
    with pytest.raises(rollout.PolarRolloutSchedulerError, match="rejected"):
        rollout._post_policy_control(
            _args(polar_policy_transition_enabled=True),
            "/rollout/admin/policy-transitions/begin",
            json_payload={},
            transition_id="t",
        )


def test_transactional_task_metadata_carries_run_namespace() -> None:
    payload = {}
    args = _args(polar_policy_transition_enabled=True)
    rollout._attach_scheduler_metadata(
        payload,
        group_id=3,
        policy_version=9,
        rollout_step=8,
        policy_namespace=rollout._task_policy_namespace(args),
    )
    assert payload["metadata"]["policy_namespace"] == rollout._policy_namespace(args)
    assert payload["metadata"]["policy_version"] == 9


def test_dispose_retries_ambiguous_quiesce_and_always_stops_local_worker(monkeypatch) -> None:
    args = _args(polar_policy_transition_enabled=True)
    context = _transition_context(engines={"engine-000": "7"})
    monkeypatch.setattr(
        rollout,
        "_last_committed_policy",
        (context.transition_id, context.policy_namespace, context.to_epoch),
    )
    monkeypatch.setattr(rollout, "_global_async_worker", None)
    replies = [
        _transition_payload(context, "serving", engine_versions={"engine-000": "8"}),
        _transition_payload(context, "quiesced", engine_versions={"engine-000": "8"}),
    ]
    stopped = []
    monkeypatch.setattr(
        rollout,
        "_post_policy_control",
        lambda *args, **kwargs: replies.pop(0),
    )
    monkeypatch.setattr(rollout, "stop_global_worker", lambda: stopped.append(True))

    rollout.dispose_rollout(args)

    assert replies == []
    assert stopped == [True]
