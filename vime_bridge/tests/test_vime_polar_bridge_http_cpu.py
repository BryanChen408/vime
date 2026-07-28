"""CPU-only HTTP integration test for the vime+polar bridge.

Drives the bridge's real submit -> poll -> parse -> convert path against an
in-process FastAPI stub-polar mounted via httpx ASGITransport. No NPU, no real
polar server, no training, no network socket.

Run: TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=<vime repo root> \
     python -m pytest vime_bridge/tests/test_vime_polar_bridge_http_cpu.py -q
"""
from __future__ import annotations

import asyncio
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import httpx
from fastapi import FastAPI

from vime_bridge import wire
from vime_bridge.adapter import session_result_to_samples
from vime_bridge.rollout import _submit_and_wait_for_task


def _trace(response_ids, reward):
    n = len(response_ids)
    return wire.Trace(
        prompt_ids=[1, 2, 3],
        response_ids=response_ids,
        loss_mask=[1] * n,
        prompt_messages=[{"role": "user", "content": "hi"}],
        response_messages=[{"role": "assistant", "content": "ok"}],
        finish_reason="stop",
        response_logprobs=[-0.1] * n,
        reward=reward,
    )


def _session_result():
    return wire.SessionResult(
        session_id="s0",
        task_id="t1",
        status=wire.SessionStatus.COMPLETED,
        trajectory=wire.Trajectory(
            status="COMPLETED",
            traces=[_trace([10, 11], 1.0), _trace([12, 13, 14], 0.0)],
        ),
    )


def _make_stub_app(poll_before_ready: int = 1):
    """Stub polar: submit returns a task_id; the task reports RUNNING for the
    first ``poll_before_ready`` GETs, then COMPLETED with one session result —
    so the bridge's real polling loop is actually exercised."""
    app = FastAPI()
    state = {"polls": 0}

    @app.post("/rollout/task/submit")
    async def submit():
        return {"task_id": "t1"}

    @app.get("/rollout/task/{task_id}")
    async def task_status(task_id: str):
        state["polls"] += 1
        if state["polls"] <= poll_before_ready:
            ts = wire.TaskStatus(task_id=task_id, status="running",
                                 total_sessions=1, completed_sessions=0, results=[])
        else:
            ts = wire.TaskStatus(task_id=task_id, status="completed",
                                 total_sessions=1, completed_sessions=1,
                                 results=[_session_result()])
        return ts.model_dump(mode="json")

    return app


def test_submit_poll_convert_against_stub_polar():
    app = _make_stub_app(poll_before_ready=2)

    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://polar") as client:
            return await _submit_and_wait_for_task(
                client,
                "http://polar",
                {"task_id": "t1", "instruction": "do x", "num_samples": 1},
                poll_interval=0.01,
            )

    result = asyncio.run(_run())

    # HTTP submit + polling loop + TaskStatus parse (wire.py) all round-tripped.
    assert result.status == "completed"
    assert len(result.results) == 1

    # Full conversion: SessionResult -> vime Samples with per-trajectory rollout_id.
    samples = session_result_to_samples(
        result.results[0], group_index=0, trajectory_index=0, reward_key="score"
    )
    assert len(samples) == 2                       # 2 traces -> 2 samples
    assert samples[0].rollout_id == samples[1].rollout_id  # one trajectory, one rollout_id
    assert samples[0].reward == {"score": 1.0}
    assert samples[1].reward == {"score": 0.0}
