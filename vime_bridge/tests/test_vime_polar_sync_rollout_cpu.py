"""CPU-only unit tests for the synchronous one-shot Polar rollout path.

Covers ``generate_rollout_polar_sync`` / ``_run_sync_train_rollout`` /
``_run_sync_train_group`` / ``_abort_inflight``, the module-level
``_task_rejection_reason`` lift, and the isolation invariant that the sync path
never reaches the async worker. No NPU, no network, no training.

Run::

    export PYTHONPATH="/usr/local/lib/python3.11/site-packages:/workspace/Megatron-LM:$PWD"
    python -m pytest vime_bridge/tests/test_vime_polar_sync_rollout_cpu.py -q -o addopts=""

``-o addopts=""`` is required: pyproject sets ``--pyargs``, which makes pytest
resolve the argument as an importable module rather than as this path.
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import ast
import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

import vime_bridge.rollout as R


NUM_GPUS = 0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _args(**overrides):
    base = dict(
        polar_url="http://polar.invalid:8080",
        polar_run_id="unit",
        polar_reward_key="score",
        polar_task_id_template="unit-{rollout_id}-{sample.group_index}",
        operator_tasks_dir=None,
        rollout_request_timeout=5.0,
        rollout_min_complete_accept_fraction=0.8,
        rollout_sync_oversubscribe_factor=1.0,
        polar_policy_transition_enabled=False,
        rollout_function_path="vime_bridge.rollout.generate_rollout_polar_sync",
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        hf_checkpoint="/tmp/ckpt",
        reward_key="score",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class FakeDataSource:
    """Hands out one fresh group per requested slot and records the demand."""

    def __init__(self, exhaust_after: int | None = None):
        self.calls: list[int] = []
        self.served = 0
        self.exhaust_after = exhaust_after

    def get_samples(self, num_samples: int):
        self.calls.append(num_samples)
        if self.exhaust_after is not None and self.served >= self.exhaust_after:
            return []
        out = []
        for _ in range(num_samples):
            self.served += 1
            out.append([SimpleNamespace(group_index=self.served)])
        return out


class RequeueDataSource(FakeDataSource):
    def __init__(self, exhaust_after: int | None = None):
        super().__init__(exhaust_after=exhaust_after)
        self.requeued: list[list[object]] = []

    def add_samples(self, samples):
        self.requeued.extend(samples)


@pytest.fixture
def stub_output(monkeypatch):
    """Stub the training-output type and metric helpers (they need slime types)."""
    monkeypatch.setattr(R, "_load_rollout_train_output_type", lambda: (lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setattr(R, "_polar_extra_metrics", lambda *a, **k: {})
    monkeypatch.setattr(R, "_extract_sample_reward", lambda s, key: 1.0)


def _script_groups(monkeypatch, script, seconds=None):
    """Drive sync groups from scripted verdicts and optional fake latencies."""
    seq = iter(script)
    durations = iter(seconds) if seconds is not None else None

    async def fake_group(*, client, args, config, rollout_id, group, group_id, handle=None):
        verdict = next(seq)
        elapsed = next(durations) if durations is not None else 0.0
        await asyncio.sleep(0)
        if handle is not None:
            handle.task_ids.append(f"task-{group_id}")
        if verdict == "raise":
            raise RuntimeError("injected transport failure")
        if verdict:
            return R._SyncGroupOutcome(
                group=group,
                accepted=True,
                samples=[SimpleNamespace(i=group_id)],
                elapsed=elapsed,
            )
        return R._SyncGroupOutcome(
            group=group,
            accepted=False,
            rejection_reason="injected reject",
            elapsed=elapsed,
        )

    monkeypatch.setattr(R, "_run_sync_train_group", fake_group)


# --------------------------------------------------------------------------
# 1. the _task_rejection_reason lift is behaviour-preserving
# --------------------------------------------------------------------------
def test_task_rejection_reason_module_and_method_agree():
    """The worker method must be a pure delegation to the module-level function."""
    group = [object(), object()]

    class _Res:
        def __init__(self, status, results):
            self.task_id = "t"
            self.status = status
            self.results = results

    cases = [
        _Res("completed", [1, 2]),  # ok
        _Res("failed", [1, 2]),  # bad status
        _Res("completed", []),  # empty results
        _Res("completed", [1]),  # count mismatch
    ]
    worker = R.AsyncPolarRolloutWorker.__new__(R.AsyncPolarRolloutWorker)
    for res in cases:
        assert R._task_rejection_reason(res, group) == worker._task_rejection_reason(res, group)


def test_task_rejection_reason_method_body_is_a_delegation():
    src = inspect.getsource(R.AsyncPolarRolloutWorker._task_rejection_reason)
    assert "return _task_rejection_reason(task_result, group)" in src


# --------------------------------------------------------------------------
# 2-3. collection loop: exact batch size, and top-up on rejection/failure
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("script", "expect_submitted", "expect_rejected", "expect_topups"),
    [
        ([True, True], 2, 0, 0),  # clean path
        ([True, False, True], 3, 1, 1),  # one rejected -> top up
        (["raise", True, True], 3, 1, 1),  # one hard failure -> top up
        ([False, False, True, True], 4, 2, 1),  # all rejected -> top up 2
    ],
)
def test_returns_exactly_batch_size(
    monkeypatch, stub_output, script, expect_submitted, expect_rejected, expect_topups
):
    _script_groups(monkeypatch, script)
    src = FakeDataSource()
    out = asyncio.run(R._run_sync_train_rollout(_args(), 7, src))

    assert len(out.samples) == 2, "must return exactly rollout_batch_size groups"
    assert out.metrics["polar/sync/submitted_groups"] == expect_submitted
    assert out.metrics["polar/sync/rejected_groups"] == expect_rejected
    assert out.metrics["polar/sync/topup_rounds"] == expect_topups
    assert src.calls[0] == 2, "first pull must ask for exactly rollout_batch_size at factor 1.0"


def test_strict_sync_reaches_return_with_no_pending_groups(monkeypatch, stub_output):
    _script_groups(monkeypatch, [True, True])
    pending_counts: list[int] = []

    async def capture_abort(pending, args, *, data_source):
        del args, data_source
        pending_counts.append(len(pending))
        return R._AbortStats()

    monkeypatch.setattr(R, "_abort_inflight", capture_abort)

    output = asyncio.run(R._run_sync_train_rollout(_args(), 7, FakeDataSource()))

    assert len(output.samples) == 2
    assert pending_counts == [0]


# --------------------------------------------------------------------------
# 4. under-supply is loud, never a silently undersized batch
# --------------------------------------------------------------------------
def test_systematic_rejection_is_bounded(monkeypatch, stub_output):
    """A never-exhausting data source plus an always-failing group must not spin forever.

    This is the shape of a real misconfiguration (malformed payload, wrong polar
    endpoint): every group is rejected, so top-up never converges. The loop has to
    give up loudly instead of resubmitting for the rest of time.
    """
    _script_groups(monkeypatch, [False] * 10_000)
    src = FakeDataSource()  # inexhaustible on purpose
    with pytest.raises(R.PolarRolloutSchedulerError, match="rather than resubmitting indefinitely"):
        asyncio.run(R._run_sync_train_rollout(_args(), 0, src))


def test_exhausted_data_source_raises(monkeypatch, stub_output):
    _script_groups(monkeypatch, [False] * 8)
    src = FakeDataSource(exhaust_after=2)
    with pytest.raises(R.PolarRolloutSchedulerError, match="data source exhausted"):
        asyncio.run(R._run_sync_train_rollout(_args(), 0, src))


# --------------------------------------------------------------------------
# 5. staleness is 0 by construction: policy_version is always the current step
# --------------------------------------------------------------------------
@pytest.mark.parametrize("durable", [False, True])
def test_policy_metadata_matches_sync_rollout_epoch(
    monkeypatch,
    stub_output,
    durable,
):
    seen: list[dict] = []

    async def capture(payload, *, max_sessions_per_task, submit_one):
        seen.append(payload["metadata"])
        return SimpleNamespace(task_id=payload["task_id"], status="completed", results=[1])

    # Payload *construction* needs real operator samples (op_name, task dirs); the
    # behaviour under test is the metadata stamping that happens after it, so stub
    # the builder and let the real `_attach_scheduler_metadata` run.
    monkeypatch.setattr(
        R,
        "_build_submission_payload",
        lambda **kw: {"task_id": f"t-{kw['task_position']}", "metadata": {}},
    )
    monkeypatch.setattr(R, "_submit_payload_in_chunks", capture)
    monkeypatch.setattr(R, "_task_rejection_reason", lambda tr, g: None)
    monkeypatch.setattr(R, "_convert_task_result_to_samples", lambda *a, **k: [SimpleNamespace(x=1)])
    monkeypatch.setattr(R, "_has_trainable_tokens", lambda s: True)
    monkeypatch.setattr(R, "_low_complete_accept_fraction_rejection_reason", lambda *a, **k: None)

    rollout_id = 41
    args = _args(polar_policy_transition_enabled=durable)
    asyncio.run(R._run_sync_train_rollout(args, rollout_id, FakeDataSource()))

    assert seen, "no payload was submitted"
    for metadata in seen:
        assert metadata["policy_version"] == rollout_id
        assert metadata["rollout_step"] == rollout_id
        if durable:
            assert metadata["policy_namespace"] == R._policy_namespace(args)
        else:
            assert "policy_namespace" not in metadata


def test_submit_reports_server_task_id_before_terminal_result():
    events = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Client:
        async def post(self, *args, **kwargs):
            del args, kwargs
            return Response({"task_id": "server-task-id"})

        async def get(self, *args, **kwargs):
            del args, kwargs
            events.append("poll")
            return Response(
                {
                    "task_id": "server-task-id",
                    "status": "completed",
                    "total_sessions": 0,
                    "completed_sessions": 0,
                    "results": [],
                }
            )

    result = asyncio.run(
        R._submit_and_wait_for_task(
            Client(),
            "http://polar",
            {"task_id": "planned-task-id"},
            poll_interval=0,
            on_task_id=lambda task_id: events.append(("task_id", task_id)),
        )
    )

    assert result.task_id == "server-task-id"
    assert events == [("task_id", "server-task-id"), "poll"]


# --------------------------------------------------------------------------
# 6. oversubscribe cancellation and group requeue
# --------------------------------------------------------------------------
def test_oversubscribe_requires_group_requeue_capability(stub_output):
    with pytest.raises(R.PolarRolloutSchedulerError, match="add_samples"):
        asyncio.run(
            R._run_sync_train_rollout(
                _args(rollout_sync_oversubscribe_factor=1.5),
                0,
                FakeDataSource(),
            )
        )


def test_read_only_rollout_data_source_is_not_treated_as_requeue_capable():
    def inherited_read_only_add_samples(self, samples):
        raise RuntimeError("read only")

    inherited_read_only_add_samples.__module__ = "vime.rollout.data_source"
    inherited_read_only_add_samples.__qualname__ = "RolloutDataSource.add_samples"
    source_type = type(
        "RolloutDataSourceChild",
        (),
        {"add_samples": inherited_read_only_add_samples},
    )

    assert not R._supports_sync_requeue(source_type())


def test_oversubscribe_factor_is_capped(stub_output):
    with pytest.raises(ValueError, match="<= 1.50"):
        asyncio.run(
            R._run_sync_train_rollout(
                _args(rollout_sync_oversubscribe_factor=1.51),
                0,
                RequeueDataSource(),
            )
        )


def test_oversubscribe_below_one_is_rejected(stub_output):
    with pytest.raises(ValueError, match=">= 1.0"):
        asyncio.run(R._run_sync_train_rollout(_args(rollout_sync_oversubscribe_factor=0.5), 0, FakeDataSource()))


def test_abort_inflight_is_noop_when_nothing_pending():
    stats = asyncio.run(R._abort_inflight(set(), _args(), data_source=None))
    assert (stats.aborted_groups, stats.aborted_sessions, stats.requeued_groups) == (0, 0, 0)


def test_abort_inflight_refuses_to_drop_live_work():
    """A non-empty pending set must fail loudly rather than leak into an engine sleep."""

    async def _run():
        task = asyncio.create_task(asyncio.sleep(60))
        try:
            with pytest.raises(NotImplementedError):
                await R._abort_inflight({task}, _args(), data_source=None)
        finally:
            task.cancel()

    asyncio.run(_run())


class _CancelResponse:
    def __init__(self, payload, error: Exception | None = None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    def json(self):
        return self.payload


class _CancelClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def post(self, url, params=None):
        self.calls.append((url, params))
        return self.response


def test_abort_inflight_cancels_tasks_and_requeues_nonselected_groups():
    async def _run():
        surplus = R._SyncGroupHandle(group=["surplus"], group_id=1)
        surplus.outcome = R._SyncGroupOutcome(group=surplus.group, accepted=True)
        pending = R._SyncGroupHandle(
            group=["pending"],
            group_id=2,
            task_ids=["task-pending", "task-pending"],
        )
        surplus_task = asyncio.create_task(asyncio.sleep(0))
        await surplus_task
        pending_task = asyncio.create_task(asyncio.sleep(60))
        source = RequeueDataSource()
        client = _CancelClient(
            _CancelResponse({"all_cancelled": True, "cancelled_sessions": 2})
        )
        try:
            stats = await R._abort_inflight(
                {pending_task},
                _args(),
                data_source=source,
                handles={surplus_task: surplus, pending_task: pending},
                client=client,
                rollout_server_url="http://polar",
            )
        finally:
            pending_task.cancel()
            await asyncio.gather(pending_task, return_exceptions=True)

        assert stats == R._AbortStats(
            aborted_groups=1,
            aborted_sessions=2,
            requeued_groups=2,
        )
        assert source.requeued == [["surplus"], ["pending"]]
        assert client.calls == [
            (
                "http://polar/rollout/task/task-pending/cancel",
                {"reason": "sync_oversubscribe_abort"},
            )
        ]

    asyncio.run(_run())


@pytest.mark.parametrize(
    "response",
    [
        _CancelResponse({"all_cancelled": False, "cancelled_sessions": 0}),
        _CancelResponse({}, RuntimeError("gateway cancellation unavailable")),
    ],
)
def test_abort_inflight_requires_positive_cancel_ack_before_requeue(response):
    async def _run():
        handle = R._SyncGroupHandle(
            group=["pending"],
            group_id=0,
            task_ids=["task-pending"],
        )
        task = asyncio.create_task(asyncio.sleep(60))
        source = RequeueDataSource()
        try:
            with pytest.raises(R.PolarRolloutSchedulerError, match="did not converge"):
                await R._abort_inflight(
                    {task},
                    _args(),
                    data_source=source,
                    handles={task: handle},
                    client=_CancelClient(response),
                    rollout_server_url="http://polar",
                )
            assert source.requeued == []
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(_run())


def test_abort_inflight_fails_closed_when_requeue_fails():
    class FailingRequeueSource:
        def add_samples(self, groups):
            raise RuntimeError(f"cannot requeue {len(groups)} group")

    async def _run():
        handle = R._SyncGroupHandle(
            group=["pending"],
            group_id=0,
            task_ids=["task-pending"],
        )
        task = asyncio.create_task(asyncio.sleep(60))
        try:
            with pytest.raises(R.PolarRolloutSchedulerError, match="requeue failed"):
                await R._abort_inflight(
                    {task},
                    _args(),
                    data_source=FailingRequeueSource(),
                    handles={task: handle},
                    client=_CancelClient(
                        _CancelResponse(
                            {"all_cancelled": True, "cancelled_sessions": 1}
                        )
                    ),
                    rollout_server_url="http://polar",
                )
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(_run())


def test_oversubscribe_selects_first_groups_and_requeues_surplus(monkeypatch, stub_output):
    _script_groups(monkeypatch, [True, True, True])
    source = RequeueDataSource()
    captured = {}

    async def fake_abort(
        pending,
        args,
        *,
        data_source,
        handles=None,
        client=None,
        rollout_server_url=None,
    ):
        del pending, args, client, rollout_server_url
        captured["handles"] = handles
        for handle in handles.values():
            data_source.add_samples([handle.group])
        return R._AbortStats(aborted_groups=1, aborted_sessions=2, requeued_groups=1)

    monkeypatch.setattr(R, "_abort_inflight", fake_abort)
    output = asyncio.run(
        R._run_sync_train_rollout(
            _args(rollout_batch_size=2, rollout_sync_oversubscribe_factor=1.5),
            0,
            source,
        )
    )

    assert len(output.samples) == 2
    assert output.metrics["polar/sync/submitted_groups"] == 3
    assert output.metrics["polar/sync/requeued_groups"] == 1
    assert len(source.requeued) == 1
    assert source.requeued[0][0].group_index == 3
    assert len(captured["handles"]) == 1


@pytest.mark.parametrize("durable", [False, True])
def test_oversubscribe_cancels_live_group_before_return(
    monkeypatch,
    stub_output,
    durable,
):
    clients = []
    locally_cancelled = []

    class Client(_CancelClient):
        def __init__(self, *, timeout):
            del timeout
            super().__init__(
                _CancelResponse({"all_cancelled": True, "cancelled_sessions": 1})
            )
            clients.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            del args
            return None

    async def fake_group(*, group, group_id, handle, **kwargs):
        del kwargs
        handle.task_ids.append(f"task-{group_id}")
        if group_id < 2:
            await asyncio.sleep(0)
            return R._SyncGroupOutcome(
                group=group,
                accepted=True,
                samples=[SimpleNamespace(i=group_id)],
                task_result=SimpleNamespace(status="completed"),
            )
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            locally_cancelled.append(group_id)
            raise

    monkeypatch.setattr(R.httpx, "AsyncClient", Client)
    monkeypatch.setattr(R, "_run_sync_train_group", fake_group)
    source = RequeueDataSource()

    output = asyncio.run(
        R._run_sync_train_rollout(
            _args(
                rollout_batch_size=2,
                rollout_sync_oversubscribe_factor=1.5,
                polar_policy_transition_enabled=durable,
            ),
            0,
            source,
        )
    )

    assert len(output.samples) == 2
    assert output.metrics["polar/sync/aborted_groups"] == 1
    assert output.metrics["polar/sync/aborted_sessions"] == 1
    assert output.metrics["polar/sync/requeued_groups"] == 1
    assert locally_cancelled == [2]
    assert source.requeued[0][0].group_index == 3
    assert clients[0].calls == [
        (
            "http://polar.invalid:8080/rollout/task/task-2/cancel",
            {"reason": "sync_oversubscribe_abort"},
        )
    ]


def test_oversubscribe_cleans_up_before_reporting_insufficient_batch(
    monkeypatch,
    stub_output,
):
    _script_groups(monkeypatch, [True, False, False])
    source = RequeueDataSource()
    cleanup_calls = []

    async def fake_abort(*args, **kwargs):
        cleanup_calls.append((args, kwargs))
        return R._AbortStats(requeued_groups=2)

    monkeypatch.setattr(R, "_abort_inflight", fake_abort)

    with pytest.raises(R.PolarRolloutSchedulerError, match="only 1/2"):
        asyncio.run(
            R._run_sync_train_rollout(
                _args(rollout_batch_size=2, rollout_sync_oversubscribe_factor=1.5),
                0,
                source,
            )
        )

    assert len(cleanup_calls) == 1


def test_durable_transition_uses_zero_inflight_sync_collector(monkeypatch, stub_output):
    _script_groups(monkeypatch, [True, True])
    source = FakeDataSource()

    output = asyncio.run(
        R._run_sync_train_rollout(
            _args(polar_policy_transition_enabled=True),
            0,
            source,
        )
    )

    assert len(output.samples) == 2
    assert output.metrics["polar/sync/accepted_groups"] == 2
    assert source.calls == [2]


# --------------------------------------------------------------------------
# 7. isolation: the sync call graph must never reach the async worker
# --------------------------------------------------------------------------
FORBIDDEN = {
    "AsyncPolarRolloutWorker",
    "get_global_async_worker",
    "stop_global_worker",
    "_global_async_worker",
    "drain_completed",
    "_ready_groups",
    "deferred_queue",
    "prepare_policy_update",
    "finish_policy_update",
    "update_policy_version",
    "_pause_gateway_generation",
    "_resume_gateway_generation",
    "begin_policy_update_drain",
    "finish_policy_update_drain",
}


def _module_level_functions(tree: ast.Module) -> dict[str, ast.AST]:
    """Module-level functions (and their nested defs), excluding class methods.

    Methods must stay out of the traversable set: they are keyed by bare name,
    so including them lets the walk hop from a module function into an unrelated
    same-named method and report hits that the sync path never reaches. Any
    genuine reference from the sync path *into* the worker is still caught --
    the forbidden names are checked on every node regardless.
    """
    out: dict[str, ast.AST] = {}

    def _collect(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[child.name] = child
                _collect(child)
            elif not isinstance(child, ast.ClassDef):
                _collect(child)

    _collect(tree)
    return out


def test_sync_path_never_reaches_async_worker():
    """The isolation is what makes "returns with zero in flight" true.

    Reaching any of the async worker's machinery would reintroduce the
    speculative group opening that leaves sessions generating past the end of
    generate() -- the exact thing that makes an engine sleep unsafe.
    """
    tree = ast.parse(Path(R.__file__).read_text(encoding="utf-8"))
    fns = _module_level_functions(tree)
    assert "generate_rollout_polar_sync" in fns
    assert "_async_session_pool_loop" not in fns, "worker methods must not be traversable"

    seen: set[str] = set()
    stack = [
        "generate_rollout_polar_sync",
        "_run_sync_train_rollout",
        "_run_sync_train_group",
        "_abort_inflight",
    ]
    hits: list[tuple[str, str]] = []
    while stack:
        fn = stack.pop()
        if fn in seen or fn not in fns:
            continue
        seen.add(fn)
        for node in ast.walk(fns[fn]):
            name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
            if name in FORBIDDEN:
                hits.append((fn, name))
            if name in fns:
                stack.append(name)

    assert not hits, f"sync path reached async-only machinery: {hits}"
    assert len(seen) > 10, "call-graph walk collapsed; the check would be vacuous"


def test_sync_entrypoint_delegates_eval_to_the_existing_batch():
    src = inspect.getsource(R.generate_rollout_polar_sync)
    assert "_run_eval_rollout" in src
    assert "_run_sync_train_rollout" in src


# --------------------------------------------------------------------------
# 8. async non-regression: the worker path is untouched by the lift
# --------------------------------------------------------------------------
def test_async_entrypoint_still_uses_the_worker():
    src = inspect.getsource(R.generate_rollout_polar_async)
    assert "async_worker" in src, "the async path must still be driven by the background worker"


# --------------------------------------------------------------------------
# 8b. latency spread: the cost side of strict synchronous collection
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("seconds", "expect_max", "expect_median", "expect_tail"),
    [
        ([10.0, 10.0, 10.0, 10.0], 10.0, 10.0, 1.0),
        ([5.0, 5.0, 5.0, 60.0], 60.0, 5.0, 12.0),
    ],
)
def test_group_latency_spread_is_reported(
    monkeypatch,
    stub_output,
    seconds,
    expect_max,
    expect_median,
    expect_tail,
):
    _script_groups(monkeypatch, [True] * len(seconds), seconds=seconds)
    output = asyncio.run(
        R._run_sync_train_rollout(
            _args(rollout_batch_size=len(seconds)),
            3,
            FakeDataSource(),
        )
    )

    assert output.metrics["polar/sync/group_seconds_max"] == expect_max
    assert output.metrics["polar/sync/group_seconds_median"] == expect_median
    assert output.metrics["polar/sync/tail_ratio"] == pytest.approx(expect_tail)


def test_rejected_groups_count_toward_the_latency_spread():
    metrics = R._sync_group_latency_metrics([1.0, 2.0, 99.0])
    assert metrics["polar/sync/group_seconds_max"] == 99.0
    assert metrics["polar/sync/group_seconds_min"] == 1.0


def test_latency_metrics_are_absent_when_no_group_completed():
    assert R._sync_group_latency_metrics([]) == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-o", "addopts="]))
