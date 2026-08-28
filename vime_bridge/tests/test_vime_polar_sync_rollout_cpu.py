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


@pytest.fixture
def stub_output(monkeypatch):
    """Stub the training-output type and metric helpers (they need slime types)."""
    monkeypatch.setattr(R, "_load_rollout_train_output_type", lambda: (lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setattr(R, "_polar_extra_metrics", lambda *a, **k: {})
    monkeypatch.setattr(R, "_extract_sample_reward", lambda s, key: 1.0)


def _script_groups(monkeypatch, script, seconds=None):
    """Drive ``_run_sync_train_group`` from a list: True=accept, False=reject, 'raise'=error.

    ``seconds`` optionally assigns each scripted group a fake wall-clock, so the
    latency-spread metrics can be asserted without real waiting.
    """
    seq = iter(script)
    durations = iter(seconds) if seconds is not None else None

    async def fake_group(*, client, args, config, rollout_id, group, group_id):
        verdict = next(seq)
        elapsed = next(durations) if durations is not None else 0.0
        await asyncio.sleep(0)
        if verdict == "raise":
            raise RuntimeError("injected transport failure")
        if verdict:
            return R._SyncGroupOutcome(
                group=group, accepted=True, samples=[SimpleNamespace(i=group_id)], elapsed=elapsed
            )
        return R._SyncGroupOutcome(
            group=group, accepted=False, rejection_reason="injected reject", elapsed=elapsed
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
def test_policy_version_always_equals_rollout_id(monkeypatch, stub_output):
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
    asyncio.run(R._run_sync_train_rollout(_args(), rollout_id, FakeDataSource()))

    assert seen, "no payload was submitted"
    for metadata in seen:
        assert metadata["policy_version"] == rollout_id
        assert metadata["rollout_step"] == rollout_id


# --------------------------------------------------------------------------
# 6. the oversubscribe seam refuses to silently leak in-flight sessions
# --------------------------------------------------------------------------
def test_oversubscribe_above_one_is_rejected(stub_output):
    with pytest.raises(NotImplementedError, match="oversubscribe"):
        asyncio.run(R._run_sync_train_rollout(_args(rollout_sync_oversubscribe_factor=1.5), 0, FakeDataSource()))


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
        ([10.0, 10.0, 10.0, 10.0], 10.0, 10.0, 1.0),  # flat: nothing to win
        ([5.0, 5.0, 5.0, 60.0], 60.0, 5.0, 12.0),  # one straggler owns the step
    ],
)
def test_group_latency_spread_is_reported(
    monkeypatch, stub_output, seconds, expect_max, expect_median, expect_tail
):
    """tail_ratio is what decides whether oversubscribe+abort is worth building.

    A synchronous step lasts as long as its slowest group; every group that
    finished earlier leaves its share of the pool idle for the rest of the
    window. Without this number, "should we implement the abort seam" stays a
    judgement call instead of a measurement.
    """
    _script_groups(monkeypatch, [True] * len(seconds), seconds=seconds)
    out = asyncio.run(R._run_sync_train_rollout(_args(rollout_batch_size=len(seconds)), 3, FakeDataSource()))

    assert out.metrics["polar/sync/group_seconds_max"] == expect_max
    assert out.metrics["polar/sync/group_seconds_median"] == expect_median
    assert out.metrics["polar/sync/tail_ratio"] == pytest.approx(expect_tail)


def test_rejected_groups_count_toward_the_latency_spread():
    """A rejected group still consumed a slot for its whole duration."""
    metrics = R._sync_group_latency_metrics([1.0, 2.0, 99.0])
    assert metrics["polar/sync/group_seconds_max"] == 99.0
    assert metrics["polar/sync/group_seconds_min"] == 1.0


def test_latency_metrics_are_absent_when_no_group_completed():
    assert R._sync_group_latency_metrics([]) == {}


# --------------------------------------------------------------------------
# 9. launch-script contract: the runner is shared, so the switch must be gated
# --------------------------------------------------------------------------
REPO_ROOT = Path(R.__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run-qwen36-35b-polar-multi-pd.sh"
SYNC_START = REPO_ROOT / "scripts" / "start_sync_hybrid.sh"


def _eval_rollout_contract(feat_sync: str) -> dict[str, str]:
    """Evaluate the runner's FEAT_SYNC_ROLLOUT block for one setting."""
    import subprocess

    text = RUNNER.read_text(encoding="utf-8")
    start = text.index('if [ "${FEAT_SYNC_ROLLOUT:-0}" = "1" ]; then')
    end = text.index("\nfi\n", start) + len("\nfi\n")
    block = text[start:end]
    script = block + '\necho "FN=$ROLLOUT_FN"\necho "SCHED=${SCHED_ARGS[*]}"\necho "TIS=${TIS_ARGS[*]}"\n'
    out = subprocess.run(
        ["bash", "-c", script],
        env={"FEAT_SYNC_ROLLOUT": feat_sync, "PATH": os.environ.get("PATH", "")},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return dict(line.split("=", 1) for line in out.strip().splitlines())


def test_runner_sync_branch_drops_the_async_only_knobs():
    """staleness is 0 by construction, so off-policy correction must be off.

    Leaving --use-tis or the session_pool knobs in place would apply an
    importance-sampling correction against a ratio that is identically 1, and
    would configure an admission pool the sync path never consults.
    """
    got = _eval_rollout_contract("1")
    assert got["FN"] == "vime_bridge.rollout.generate_rollout_polar_sync"
    assert got["TIS"] == "", "--use-tis has no meaning when staleness is 0"
    assert "session_pool" not in got["SCHED"]
    assert "--rollout-sync-oversubscribe-factor" in got["SCHED"]


def test_runner_async_branch_is_unchanged():
    """The runner is shared with start_pd.sh; the default must stay async."""
    got = _eval_rollout_contract("0")
    assert got["FN"] == "vime_bridge.rollout.generate_rollout_polar_async"
    assert got["TIS"] == "--use-tis"
    assert "--rollout-scheduler-mode session_pool" in got["SCHED"]
    assert "--rollout-release-on-postrun" in got["SCHED"]


def test_sync_hybrid_start_script_enables_the_sync_contract():
    text = SYNC_START.read_text(encoding="utf-8")
    assert "FEAT_SYNC_ROLLOUT=1" in text
    # These configure the async worker's admission/staleness bookkeeping; the
    # sync path has none, so leaving them here would misdescribe the run.
    for stale_knob in ("POLAR_MAX_ACTIVE_SESSIONS=", "POLAR_DRAIN_SESSIONS=", "POLAR_MAX_OFF_POLICY_STEPS="):
        assert stale_knob not in text, f"{stale_knob} is async-only and must not be set for a sync run"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-o", "addopts="]))
