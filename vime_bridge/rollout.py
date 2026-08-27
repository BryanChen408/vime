"""Slime rollout bridge for Polar-managed agent sessions.

Single entrypoint ``generate_rollout_polar_async`` routes training to a
persistent background worker and evaluation to a one-shot submit+poll batch.
Both paths speak Polar's async-only HTTP surface (``/rollout/task/submit`` +
``/rollout/task/{task_id}``).
"""

from __future__ import annotations

import asyncio
import atexit
import copy
import hashlib
import json
import logging
import math
import queue
import statistics
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request

from vime_bridge._messages import prompt_to_instruction_text
from vime_bridge.adapter import RolloutLogprobError, session_result_to_samples
from vime_bridge.config import (
    PolarSlimeConfig,
    render_instruction,
    render_task_payload,
    resolve_polar_slime_config,
)
from vime_bridge.wire import SessionStatus, TaskResult, TaskStatus
from vime_bridge.version_span import push_policy_version_to_gateway

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 2.0  # seconds between task-status polls (eval / no-callback path)
_CALLBACK_FALLBACK_POLL_SECONDS = 60.0  # defensive backstop for dropped callbacks
_SESSION_POOL_RUN_RELEASE_POLL_SECONDS = 2.0
_SESSION_POOL_RUN_RELEASE_POLL_TIMEOUT_SECONDS = 1.0
_LONGEST_TRACE_ARTIFACT_INTERVAL = 5  # dump longest trace every N rollouts
_SESSION_POOL_RUN_RELEASE_STATUSES = frozenset({
    str(SessionStatus.POST_RUN),
    str(SessionStatus.BUILDING),
    str(SessionStatus.EVALUATING),
    str(SessionStatus.COMPLETED),
    str(SessionStatus.ERROR),
    str(SessionStatus.TIMEOUT),
})


class PolarRolloutSchedulerError(RuntimeError):
    """Raised when the async Polar scheduler cannot safely make progress."""


class PolarLowCompleteAcceptFractionError(PolarRolloutSchedulerError):
    """Raised when a completed task has too few trainable completed sessions."""


class _NoopCallbackServer:
    should_exit = False


@dataclass(slots=True)
class _DeferredGroup:
    group: list[Any]


@dataclass(slots=True)
class _PendingGroup:
    group_id: int
    group: list[Any]
    submitted_rollout_id: int
    policy_version: int
    session_cost: int


@dataclass(slots=True)
class _CompletedGroup:
    group_id: int
    group: list[Any]
    samples: list[Any]
    task_id: str
    submitted_rollout_id: int
    policy_version: int
    session_count: int
    completed_at: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class _ReadyGroup:
    completed: _CompletedGroup | None = None
    dropped: bool = False


@dataclass(slots=True)
class _PendingSessionUnit:
    group_id: int
    group_pos: int
    sample_pos: int
    sample: Any
    parent_group: list[Any]
    parent_task_id: str
    task_id: str
    submitted_rollout_id: int
    policy_version: int


@dataclass(slots=True)
class _SessionSlotResult:
    task_result: TaskResult
    samples: list[Any]


@dataclass(slots=True)
class _SessionGroupAccumulator:
    group_id: int
    group_pos: int
    group: list[Any]
    submitted_rollout_id: int
    policy_version: int
    parent_task_id: str
    next_submit_pos: int = 0
    slots: dict[int, _SessionSlotResult] = field(default_factory=dict)
    result_paths: list[str] = field(default_factory=list)
    rejected_reason: str | None = None
    cancelled_by_pause_policy: bool = False

    @property
    def group_size(self) -> int:
        return len(self.group)

    @property
    def submitted_count(self) -> int:
        return self.next_submit_pos

    @property
    def completed_count(self) -> int:
        return len(self.slots)

    @property
    def fully_submitted(self) -> bool:
        return self.next_submit_pos >= self.group_size

    @property
    def complete(self) -> bool:
        return self.completed_count >= self.group_size

    @property
    def partial(self) -> bool:
        return not self.fully_submitted

    @property
    def terminal(self) -> bool:
        return self.complete or self.rejected_reason is not None

# ---------------------------------------------------------------------------
# Global worker singleton
# ---------------------------------------------------------------------------
_global_async_worker: "AsyncPolarRolloutWorker | None" = None
_worker_lock = threading.Lock()


def get_global_async_worker(args: Any, data_source: Any) -> "AsyncPolarRolloutWorker":
    global _global_async_worker
    with _worker_lock:
        if _global_async_worker is None or not _global_async_worker.is_alive():
            logger.info("Creating new async Polar rollout worker")
            _global_async_worker = AsyncPolarRolloutWorker(args, data_source)
            _global_async_worker.start()
        return _global_async_worker


def stop_global_worker() -> None:
    global _global_async_worker
    with _worker_lock:
        if _global_async_worker is not None:
            _global_async_worker.stop()
            _global_async_worker = None


def update_policy_version(args: Any, policy_version: int) -> None:
    """Optional hook called by Slime after serving weights are updated."""
    del args
    with _worker_lock:
        if _global_async_worker is not None:
            _global_async_worker.update_policy_version(policy_version)


def prepare_policy_update(args: Any, policy_version: int) -> dict[str, Any]:
    """Close Polar admission before VIME aborts serving-engine requests.

    This hook deliberately does not resume on failure and does not advance the gateway
    version yet.  ``RolloutManager.prepare_policy_update`` first calls this hook, aborts
    every serving engine, and then calls ``commit_policy_update_boundary`` below.  That
    ordering prevents an old-weight completion from being stamped with the new version.
    """
    logger.info("Preparing Polar bridge for policy_version=%s weight update", policy_version)
    with _worker_lock:
        worker = _global_async_worker
        if worker is not None:
            if worker.config.scheduler_mode == "session_pool":
                worker.begin_policy_update_drain(policy_version)
            else:
                worker.pause_admission()

    if worker is not None and worker.config.scheduler_mode == "session_pool":
        # The session-level drain waits for whole agent sessions to finish, which can take
        # far longer than any sane timeout. Two things matter here:
        #   * It must not be the reason the weight update is skipped. The gateway admission
        #     pause below and the post-abort drain/version commit are what protect the weight
        #     boundary; letting a slow session drain raise before them would leave the boundary
        #     unprotected exactly when there is most in-flight work to protect it from.
        #   * Synchronous (colocate) training has no overlap window at all: the engines sleep
        #     for the entire training step, so waiting for sessions only stalls training.
        #     --no-polar-weight-update-drain-sessions skips the wait and abandons whatever is
        #     in flight to the version-span guard (the gateway rejects a continuation whose
        #     turn would cross the update), which comes back as an ERROR trajectory that
        #     adapter.py excludes from training.
        if bool(getattr(args, "polar_weight_update_drain_sessions", True)):
            timeout_seconds = float(getattr(args, "polar_weight_update_pause_timeout", 300.0))
            try:
                worker.wait_for_policy_update_drain(timeout=timeout_seconds)
            except PolarRolloutSchedulerError:
                logger.warning(
                    "Polar session drain did not finish within %.0fs; proceeding with the "
                    "weight update anyway and leaving the stragglers to the version-span guard.",
                    timeout_seconds,
                )
        else:
            logger.info(
                "Polar session drain skipped (--no-polar-weight-update-drain-sessions); "
                "in-flight sessions are abandoned at the weight boundary."
            )

    return _pause_gateway_generation(args, require_drained=False)


def commit_policy_update_boundary(args: Any, policy_version: int) -> dict[str, Any]:
    """Confirm engine abort propagation, then publish the next policy version.

    Called only after VIME's rollout-engine ``/pause?mode=abort`` fanout succeeds.
    Failure is fatal and leaves both Polar and the local scheduler closed.
    """
    timeout_seconds = float(getattr(args, "polar_gateway_control_timeout", 30.0))
    status = _pause_gateway_generation(
        args,
        timeout_seconds=timeout_seconds,
        require_drained=True,
    )
    if not push_policy_version_to_gateway(args, policy_version):
        raise PolarRolloutSchedulerError(
            f"Failed to publish Polar policy_version={policy_version} while admission "
            "was paused"
        )
    return status


def finish_policy_update(args: Any, policy_version: int) -> None:
    """Optional hook called by Slime after overlapping inference weight sync."""
    with _worker_lock:
        worker = _global_async_worker
        if worker is not None and worker.config.scheduler_mode == "session_pool":
            worker.update_policy_version(policy_version)
    _resume_gateway_generation(args)
    with _worker_lock:
        worker = _global_async_worker
        if worker is not None:
            if worker.config.scheduler_mode == "session_pool":
                worker.finish_policy_update_drain()
            else:
                worker.resume_admission()
    logger.info("Finished Polar bridge policy_version=%s weight update", policy_version)


def _resolve_gateway_url(args: Any) -> str | None:
    gateway_url = getattr(args, "polar_gateway_url", None)
    if gateway_url:
        return str(gateway_url).rstrip("/")
    return None


def _resolve_rollout_url(args: Any) -> str | None:
    rollout_url = getattr(args, "polar_url", None) or getattr(args, "polar_rollout_url", None)
    if rollout_url:
        return str(rollout_url).rstrip("/")
    return None


def _control_post_json(
    url: str,
    *,
    action: str,
    timeout: float,
    params: dict[str, object] | None = None,
) -> dict[str, Any]:
    """POST one control request without leaking non-pickleable httpx exceptions to Ray."""
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, params=params)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        try:
            detail = response.text if response is not None else str(exc)
        except Exception:
            detail = str(exc)
        raise PolarRolloutSchedulerError(
            f"Polar {action} failed at {url}: status={status} detail={detail}"
        ) from None
    except (TypeError, ValueError) as exc:
        raise PolarRolloutSchedulerError(
            f"Polar {action} returned invalid JSON at {url}: {exc}"
        ) from None

    if not isinstance(payload, dict):
        raise PolarRolloutSchedulerError(
            f"Polar {action} returned non-object JSON at {url}: {type(payload).__name__}"
        )
    return payload


def _pause_gateway_generation(
    args: Any,
    *,
    timeout_seconds: float | None = None,
    require_drained: bool,
) -> dict[str, Any]:
    rollout_url = _resolve_rollout_url(args)
    effective_timeout = (
        float(timeout_seconds)
        if timeout_seconds is not None
        else float(getattr(args, "polar_weight_update_pause_timeout", 300.0))
    )
    if rollout_url:
        url = f"{rollout_url}/rollout/admin/inference/pause"
        payload = _control_post_json(
            url,
            action="rollout gateway pause",
            timeout=max(effective_timeout + 15.0, 20.0),
            params={"timeout_seconds": effective_timeout},
        )
        all_paused = payload.get("all_paused") is True
        all_drained = payload.get("all_drained") is True
        inflight = int(payload.get("inflight", 0) or 0)
    else:
        gateway_url = _resolve_gateway_url(args)
        if not gateway_url:
            raise PolarRolloutSchedulerError(
                "polar_url, polar_rollout_url, or polar_gateway_url is required when "
                "polar_allow_weight_update_overlap is enabled"
            )

        url = f"{gateway_url}/admin/inference/pause"
        payload = _control_post_json(
            url,
            action="gateway pause",
            timeout=max(effective_timeout + 5.0, 10.0),
            params={"timeout_seconds": effective_timeout},
        )
        all_paused = payload.get("paused") is True
        all_drained = payload.get("drained") is True
        inflight = int(payload.get("inflight", 0) or 0)

    if not all_paused:
        raise PolarRolloutSchedulerError(
            f"Polar admission pause was not acknowledged: {payload}"
        )
    if require_drained and not all_drained:
        raise PolarRolloutSchedulerError(
            f"Polar remained non-drained after VIME aborted rollout engines: inflight={inflight} "
            f"payload={payload}"
        )

    status = {
        "all_paused": all_paused,
        "all_drained": all_drained,
        "inflight": inflight,
        "payload": payload,
    }
    logger.info(
        "Polar gateway pause acknowledged: paused=%s drained=%s inflight=%s",
        all_paused,
        all_drained,
        inflight,
    )
    return status


def _resume_gateway_generation(args: Any) -> None:
    rollout_url = _resolve_rollout_url(args)
    if rollout_url:
        request_timeout = float(getattr(args, "polar_gateway_control_timeout", 30.0))
        payload = _control_post_json(
            f"{rollout_url}/rollout/admin/inference/resume",
            action="rollout gateway resume",
            timeout=max(request_timeout, 5.0),
        )
        logger.info("Resumed Polar gateway generation via rollout server: %s", payload)
        return

    gateway_url = _resolve_gateway_url(args)
    if not gateway_url:
        return

    request_timeout = float(getattr(args, "polar_gateway_control_timeout", 30.0))
    payload = _control_post_json(
        f"{gateway_url}/admin/inference/resume",
        action="gateway resume",
        timeout=max(request_timeout, 5.0),
    )
    logger.info("Resumed Polar gateway generation after inference weight update: %s", payload)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _build_task_payload(
    *,
    args: Any,
    config: PolarSlimeConfig,
    group: list[Any],
    rollout_id: int,
    task_position: int,
) -> dict[str, Any]:
    first_sample = group[0]
    prompt_text = prompt_to_instruction_text(getattr(first_sample, "prompt", ""))
    instruction = render_instruction(
        args=args,
        config=config,
        sample=first_sample,
        prompt_text=prompt_text,
        rollout_id=rollout_id,
        task_position=task_position,
        num_rollouts=len(group),
    )
    return render_task_payload(
        args=args,
        config=config,
        sample=first_sample,
        instruction=instruction,
        rollout_id=rollout_id,
        task_position=task_position,
        num_rollouts=len(group),
    )


def _build_submission_payload(
    *,
    args: Any,
    config: PolarSlimeConfig,
    group: list[Any],
    rollout_id: int,
    task_position: int,
) -> dict[str, Any]:
    payload = _build_task_payload(
        args=args,
        config=config,
        group=group,
        rollout_id=rollout_id,
        task_position=task_position,
    )
    if config.submit_mode == "task_request":
        return payload

    first_sample = group[0]
    sample_metadata = copy.deepcopy(getattr(first_sample, "metadata", None) or {})
    op_name = sample_metadata.get("op_name") or getattr(first_sample, "op_name", None)
    if not op_name:
        raise PolarRolloutSchedulerError(
            "operator_samples submit mode requires sample.metadata.op_name"
        )

    metadata = payload.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("polar operator-sample metadata must be a mapping when provided")

    thin: dict[str, Any] = {
        "task_id": str(payload["task_id"]),
        "instruction": str(payload["instruction"]),
        "num_samples": int(payload.get("num_samples") or len(group)),
        "sample": {
            "op_name": str(op_name),
            "group_index": getattr(first_sample, "group_index", None),
            "index": getattr(first_sample, "index", None),
            "metadata": sample_metadata,
        },
        "metadata": metadata,
    }
    if config.operator_profile:
        thin["profile"] = config.operator_profile
    if payload.get("timeout_seconds") is not None:
        thin["timeout_seconds"] = payload["timeout_seconds"]
    _attach_operator_task_source(thin, config)
    return thin


def _attach_operator_task_source(payload: dict[str, Any], config: PolarSlimeConfig) -> None:
    tasks_dir = config.operator_tasks_dir
    if not tasks_dir:
        return
    sample = payload.get("sample")
    if not isinstance(sample, dict):
        raise ValueError("operator_samples payload must include a sample mapping")
    op_name = sample.get("op_name")
    if not isinstance(op_name, str) or not op_name:
        raise PolarRolloutSchedulerError(
            "operator_samples submit mode requires sample.op_name to attach task source"
        )
    if "/" in op_name or "\\" in op_name or op_name in {".", ".."}:
        raise PolarRolloutSchedulerError(
            "operator_samples sample.op_name must be a file stem, not a path"
        )
    task_path = Path(tasks_dir) / f"{op_name}.py"
    if task_path.name != f"{op_name}.py" or not task_path.is_file():
        raise PolarRolloutSchedulerError(f"missing operator task source: {task_path}")
    source = task_path.read_text(encoding="utf-8")
    sample["task_source"] = source
    sample["task_source_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()


def _chunk_task_payloads(
    payload: dict[str, Any],
    *,
    max_sessions_per_task: int | None,
) -> list[dict[str, Any]]:
    total_sessions = int(payload.get("num_samples") or 1)
    if max_sessions_per_task is None or total_sessions <= max_sessions_per_task:
        return [payload]
    if max_sessions_per_task <= 0:
        raise ValueError("max_sessions_per_task must be greater than 0")

    base_task_id = str(payload["task_id"])
    chunks: list[dict[str, Any]] = []
    chunk_count = math.ceil(total_sessions / max_sessions_per_task)
    for chunk_index, chunk_start in enumerate(range(0, total_sessions, max_sessions_per_task)):
        chunk_size = min(max_sessions_per_task, total_sessions - chunk_start)
        child = copy.deepcopy(payload)
        child["task_id"] = f"{base_task_id}--part{chunk_index:03d}"
        child["num_samples"] = chunk_size
        metadata = child.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("polar task metadata must be a mapping when provided")
        child["metadata"] = {
            **metadata,
            "parent_task_id": base_task_id,
            "chunk_index": chunk_index,
            "chunk_start": chunk_start,
            "chunk_size": chunk_size,
            "chunk_count": chunk_count,
        }
        chunks.append(child)
    return chunks


def _merge_task_results(parent_task_id: str, child_results: list[TaskResult]) -> TaskResult:
    if not child_results:
        return TaskResult(task_id=parent_task_id, status="failed", results=[])
    if len(child_results) == 1 and child_results[0].task_id == parent_task_id:
        return child_results[0]

    merged_results = [
        result
        for child in child_results
        for result in child.results
    ]
    result_paths = [
        path
        for child in child_results
        for path in child.result_paths
    ]
    status = "completed" if all(child.status == "completed" for child in child_results) else "failed"
    return TaskResult(
        task_id=parent_task_id,
        status=status,
        results=merged_results,
        result_paths=result_paths,
    )


def _attach_scheduler_metadata(
    payload: dict[str, Any],
    *,
    group_id: int,
    policy_version: int,
    rollout_step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    metadata = payload.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("polar task metadata must be a mapping when provided")
    scheduler_metadata = {
        "group_id": group_id,
        "policy_version": policy_version,
        "rollout_step": rollout_step,
    }
    if extra:
        scheduler_metadata.update(extra)
    payload["metadata"] = {
        **metadata,
        **scheduler_metadata,
    }


def _make_session_pool_task_id(base_task_id: str, *, group_id: int, sample_pos: int) -> str:
    return f"{base_task_id}--g{group_id:06d}-sp{sample_pos:03d}"


def _new_session_group_accumulator(
    *,
    args: Any,
    config: PolarSlimeConfig,
    group_id: int,
    group_pos: int,
    group: list[Any],
    submitted_rollout_id: int,
    policy_version: int,
) -> _SessionGroupAccumulator:
    payload = _build_submission_payload(
        args=args,
        config=config,
        group=group,
        rollout_id=group_id,
        task_position=0,
    )
    return _SessionGroupAccumulator(
        group_id=group_id,
        group_pos=group_pos,
        group=group,
        submitted_rollout_id=submitted_rollout_id,
        policy_version=policy_version,
        parent_task_id=str(payload["task_id"]),
    )


def _next_session_pool_unit(accumulator: _SessionGroupAccumulator) -> _PendingSessionUnit | None:
    if accumulator.fully_submitted:
        return None
    sample_pos = accumulator.next_submit_pos
    accumulator.next_submit_pos += 1
    return _PendingSessionUnit(
        group_id=accumulator.group_id,
        group_pos=accumulator.group_pos,
        sample_pos=sample_pos,
        sample=accumulator.group[sample_pos],
        parent_group=accumulator.group,
        parent_task_id=accumulator.parent_task_id,
        task_id=_make_session_pool_task_id(
            accumulator.parent_task_id,
            group_id=accumulator.group_id,
            sample_pos=sample_pos,
        ),
        submitted_rollout_id=accumulator.submitted_rollout_id,
        policy_version=accumulator.policy_version,
    )


def _build_session_unit_payload(
    *,
    args: Any,
    config: PolarSlimeConfig,
    unit: _PendingSessionUnit,
) -> dict[str, Any]:
    payload = _build_submission_payload(
        args=args,
        config=config,
        group=[unit.sample],
        rollout_id=unit.group_id,
        task_position=unit.sample_pos,
    )
    payload["num_samples"] = 1
    payload["task_id"] = unit.task_id
    _attach_scheduler_metadata(
        payload,
        group_id=unit.group_id,
        policy_version=unit.policy_version,
        rollout_step=unit.submitted_rollout_id,
        extra={
            "session_pool": True,
            "parent_task_id": unit.parent_task_id,
            "sample_pos": unit.sample_pos,
            "group_size": len(unit.parent_group),
        },
    )
    return payload


def _flatten_session_pool_units(
    *,
    args: Any,
    config: PolarSlimeConfig,
    groups: list[list[Any]],
    first_group_id: int,
    submitted_rollout_id: int,
    policy_version: int,
) -> list[_PendingSessionUnit]:
    units: list[_PendingSessionUnit] = []
    for group_pos, group in enumerate(groups):
        accumulator = _new_session_group_accumulator(
            args=args,
            config=config,
            group_id=first_group_id + group_pos,
            group_pos=group_pos,
            group=group,
            submitted_rollout_id=submitted_rollout_id,
            policy_version=policy_version,
        )
        while True:
            unit = _next_session_pool_unit(accumulator)
            if unit is None:
                break
            units.append(unit)
    return units


def _record_session_unit_result(
    *,
    config: PolarSlimeConfig,
    accumulator: _SessionGroupAccumulator,
    unit: _PendingSessionUnit,
    task_result: TaskResult,
    max_tokens: int | None = None,
) -> None:
    if task_result.status != "completed":
        raise PolarRolloutSchedulerError(
            f"Task {task_result.task_id} cannot be accepted: task status={task_result.status}"
        )
    if len(task_result.results) != 1:
        raise PolarRolloutSchedulerError(
            f"Task {task_result.task_id} cannot be accepted: session count "
            f"{len(task_result.results)} != expected 1"
        )
    samples = _convert_task_result_to_samples(
        config,
        task_result,
        [unit.sample],
        max_tokens=max_tokens,
    )
    if not samples:
        raise PolarRolloutSchedulerError(f"Task {task_result.task_id} converted to zero samples")
    accumulator.slots[unit.sample_pos] = _SessionSlotResult(
        task_result=task_result,
        samples=samples,
    )
    accumulator.result_paths.extend(task_result.result_paths)


def _synthetic_session_pool_task_result(accumulator: _SessionGroupAccumulator) -> TaskResult:
    ordered_slots = [accumulator.slots[pos] for pos in range(accumulator.group_size)]
    results = [
        result
        for slot in ordered_slots
        for result in slot.task_result.results
    ]
    result_paths = [
        path
        for slot in ordered_slots
        for path in slot.task_result.result_paths
    ]
    status = "completed" if all(slot.task_result.status == "completed" for slot in ordered_slots) else "failed"
    return TaskResult(
        task_id=accumulator.parent_task_id,
        status=status,
        results=results,
        result_paths=result_paths,
    )


def _completed_group_from_session_accumulator(
    config: PolarSlimeConfig,
    accumulator: _SessionGroupAccumulator,
) -> _CompletedGroup:
    if not accumulator.complete:
        raise PolarRolloutSchedulerError(
            f"Session pool group {accumulator.group_id} is not complete"
        )
    task_result = _synthetic_session_pool_task_result(accumulator)
    group_samples = [
        sample
        for pos in range(accumulator.group_size)
        for sample in accumulator.slots[pos].samples
    ]
    if not group_samples:
        raise PolarRolloutSchedulerError(f"Task {task_result.task_id} converted to zero samples")
    if not _has_trainable_tokens(group_samples):
        raise PolarRolloutSchedulerError(
            f"Task {task_result.task_id} produced zero trainable tokens"
        )
    rejection_reason = _low_complete_accept_fraction_rejection_reason(
        config,
        task_result,
        group_samples,
    )
    if rejection_reason is not None:
        raise PolarLowCompleteAcceptFractionError(
            f"Task {task_result.task_id} cannot be accepted: {rejection_reason}"
        )
    return _CompletedGroup(
        group_id=accumulator.group_id,
        group=accumulator.group,
        samples=group_samples,
        task_id=task_result.task_id,
        submitted_rollout_id=accumulator.submitted_rollout_id,
        policy_version=accumulator.policy_version,
        session_count=len(task_result.results),
    )


async def _submit_and_wait_for_task(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, Any],
    *,
    poll_interval: float = _POLL_INTERVAL,
    submit_path: str = "/rollout/task/submit",
) -> TaskResult:
    """Submit one task via the async endpoint and poll until terminal."""
    resp = await client.post(
        f"{base_url}{submit_path}",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    task_id = resp.json()["task_id"]

    while True:
        await asyncio.sleep(poll_interval)
        try:
            status_resp = await client.get(f"{base_url}/rollout/task/{task_id}")
            status_resp.raise_for_status()
        except (
            httpx.HTTPStatusError,
            httpx.TimeoutException,
            httpx.TransportError,
        ) as exc:
            logger.warning("Polling Polar task %s failed; continuing: %s", task_id, exc)
            continue
        status = TaskStatus.model_validate(status_resp.json())
        if status.status in ("completed", "failed"):
            break

    return TaskResult(
        task_id=task_id,
        status=status.status,
        results=status.results,
        result_paths=status.result_paths,
    )


async def _submit_payload_in_chunks(
    payload: dict[str, Any],
    *,
    max_sessions_per_task: int | None,
    submit_one: Any,
) -> TaskResult:
    """Submit one logical task, splitting session fanout into sequential child tasks."""
    chunks = _chunk_task_payloads(
        payload,
        max_sessions_per_task=max_sessions_per_task,
    )
    if len(chunks) == 1:
        return await submit_one(chunks[0])

    child_results: list[TaskResult] = []
    for chunk in chunks:
        child_results.append(await submit_one(chunk))
    return _merge_task_results(str(payload["task_id"]), child_results)


def _resolve_max_tokens(args: Any) -> int | None:
    """Per-sample token cap Slime's dynamic batcher can fit on one GPU.

    Megatron asserts every sample length <= max_tokens_per_gpu * cp_size.
    Deep agent trajectories can exceed this (24-turn sessions → 80k+ tokens)
    and must be dropped before they reach the batcher.
    """
    mtpg = getattr(args, "max_tokens_per_gpu", None)
    if not mtpg:
        return None
    cp_size = int(getattr(args, "context_parallel_size", 1) or 1)
    return int(mtpg) * cp_size


def _convert_task_result_to_samples(
    config: PolarSlimeConfig,
    task_result: TaskResult,
    group: list[Any],
    *,
    max_tokens: int | None = None,
) -> list[Any]:
    """Convert one task's session results into flat Slime samples.

    Each session → one trajectory → N traces → N samples, all tagged
    with the same ``Sample.index`` so the reward post-processor groups
    them as one trajectory.  The index is taken from the originating
    group sample at matching position, falling back to the position
    within the task result.
    """
    group_index = _group_index_for(group)
    group_samples: list[Any] = []
    for pos, session_result in enumerate(task_result.results):
        source = group[pos] if pos < len(group) else None
        traj_idx = int(getattr(source, "index", pos) if source is not None else pos)
        group_samples.extend(
            session_result_to_samples(
                session_result,
                group_index,
                trajectory_index=traj_idx,
                reward_key=config.reward_key,
                max_tokens=max_tokens,
            )
        )
    return group_samples


def _trainable_token_count(sample: Any) -> int:
    if bool(getattr(sample, "remove_sample", False)):
        return 0
    loss_mask = getattr(sample, "loss_mask", None)
    if loss_mask is None:
        return int(getattr(sample, "response_length", 0) or 0)
    return sum(1 for value in loss_mask if int(value) != 0)


def _has_trainable_tokens(samples: list[Any]) -> bool:
    return any(_trainable_token_count(sample) > 0 for sample in samples)


def _low_complete_accept_fraction_rejection_reason(
    config: PolarSlimeConfig,
    task_result: TaskResult,
    samples: list[Any],
) -> str | None:
    threshold = config.min_complete_accept_fraction
    if threshold <= 0.0:
        return None

    total_sessions = len(task_result.results)
    if total_sessions <= 0:
        return "empty task results"

    completed_trainable = _completed_trainable_session_count(task_result, samples)
    required = math.ceil(total_sessions * threshold)
    if completed_trainable >= required:
        return None

    fraction = completed_trainable / total_sessions
    return (
        f"completed trainable sessions {completed_trainable}/{total_sessions} "
        f"({fraction:.3f}) below polar_min_complete_accept_fraction={threshold:g} "
        f"(requires >= {required})"
    )


def _completed_trainable_session_count(
    task_result: TaskResult,
    samples: list[Any],
) -> int:
    trainable_session_ids: set[str] = set()
    for sample in samples:
        if _trainable_token_count(sample) <= 0:
            continue
        session_id = _sample_session_id(sample)
        if session_id:
            trainable_session_ids.add(session_id)

    count = 0
    for result in task_result.results:
        if (
            _status_value(result.status) == "COMPLETED"
            and result.session_id in trainable_session_ids
        ):
            count += 1
    return count


def _sample_session_id(sample: Any) -> str | None:
    polar_meta = (getattr(sample, "metadata", {}) or {}).get("polar", {})
    session_id = polar_meta.get("session_id") or getattr(sample, "session_id", None)
    return str(session_id) if session_id else None


def _status_value(status: Any) -> str:
    return str(getattr(status, "value", status))


def _is_zero_trainable_error(exc: BaseException) -> bool:
    return "zero trainable tokens" in str(exc)


def _annotate_accepted_samples(
    samples: list[Any],
    *,
    accepted_rollout_id: int,
    staleness: int,
    policy_version: int,
    scheduler_group_id: int,
) -> None:
    for sample in samples:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            sample.metadata = metadata
        polar_meta = metadata.setdefault("polar", {})
        if not isinstance(polar_meta, dict):
            polar_meta = {}
            metadata["polar"] = polar_meta
        polar_meta.update(
            {
                "accepted_rollout_id": int(accepted_rollout_id),
                "policy_staleness": int(staleness),
                "policy_version": int(policy_version),
                "scheduler_group_id": int(scheduler_group_id),
            }
        )
        train_metadata = getattr(sample, "train_metadata", None)
        if train_metadata is None:
            train_metadata = {}
            sample.train_metadata = train_metadata
        train_metadata.update(
            {
                "policy_staleness": int(staleness),
                "policy_version": int(policy_version),
            }
        )


# ---------------------------------------------------------------------------
# Persistent training worker
# ---------------------------------------------------------------------------
class AsyncPolarRolloutWorker:
    """Persistent background worker that continuously submits Polar tasks.

    Runs in its own thread with a dedicated asyncio event loop.  Pulls
    sample groups from ``data_source``, submits them to the async
    ``/rollout/task/submit`` endpoint, polls until completion, converts
    results, and stores ready groups in an ordered handoff buffer. Training
    loops call ``drain_completed()`` to collect finished groups.
    """

    def __init__(self, args: Any, data_source: Any) -> None:
        self.args = args
        self.data_source = data_source
        self.config = resolve_polar_slime_config(args)
        batch_size = int(getattr(args, "rollout_batch_size", 1) or 1)
        self._ready_groups: dict[int, _ReadyGroup] = {}
        self.deferred_queue: queue.Queue[_DeferredGroup] = queue.Queue()
        self._running = True
        self._thread: threading.Thread | None = None
        self._group_counter = 0
        self._batch_size = batch_size
        self._current_rollout_id = int(getattr(args, "start_rollout_id", 0) or 0)
        self._policy_version = self._current_rollout_id
        self._fatal_error: BaseException | None = None
        self._state_lock = threading.RLock()
        self._metrics: dict[str, float] = {}
        self._active_groups = 0
        self._active_sessions = 0
        self._ready_group_count = 0
        self._admission_paused = False
        self._policy_update_draining = False
        self._policy_update_target_version: int | None = None
        self._policy_update_drain_started_at: float | None = None
        self._policy_update_drain_complete = threading.Event()
        self._policy_update_drain_complete.set()
        self._session_pool_open_groups = 0
        self._session_pool_partial_open_groups = 0
        self._session_pool_pending_sessions = 0
        # Per-task callback plumbing: event fires when the rollout server POSTs
        # the terminal TaskResult to our local listener.
        self._task_events: dict[str, asyncio.Event] = {}
        self._task_results: dict[str, TaskResult] = {}
        self._callback_url: str | None = None

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="polar-async-rollout")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- results ---------------------------------------------------------------

    def set_rollout_context(self, rollout_id: int) -> None:
        with self._state_lock:
            self._current_rollout_id = max(self._current_rollout_id, int(rollout_id))

    def update_policy_version(self, policy_version: int) -> None:
        with self._state_lock:
            self._policy_version = max(self._policy_version, int(policy_version))

    def pause_admission(self) -> None:
        with self._state_lock:
            self._admission_paused = True
            self._metrics["polar/scheduler/admission_pauses"] = (
                self._metrics.get("polar/scheduler/admission_pauses", 0.0) + 1.0
            )

    def resume_admission(self) -> None:
        with self._state_lock:
            self._admission_paused = False

    def begin_policy_update_drain(self, policy_version: int) -> None:
        with self._state_lock:
            self._policy_update_draining = True
            self._policy_update_target_version = int(policy_version)
            self._policy_update_drain_started_at = time.monotonic()
            self._policy_update_drain_complete.clear()
            self._metrics["polar/session_pool/policy_update_drains"] = (
                self._metrics.get("polar/session_pool/policy_update_drains", 0.0) + 1.0
            )
            self._metrics["polar/scheduler/admission_pauses"] = (
                self._metrics.get("polar/scheduler/admission_pauses", 0.0) + 1.0
            )

    def wait_for_policy_update_drain(self, *, timeout: float | None) -> None:
        if not self._policy_update_drain_complete.wait(timeout=timeout):
            raise PolarRolloutSchedulerError(
                "Timed out waiting for Polar session_pool scheduler to drain "
                "open groups before policy update"
            )

    def finish_policy_update_drain(self) -> None:
        with self._state_lock:
            self._policy_update_draining = False
            self._policy_update_target_version = None
            self._policy_update_drain_started_at = None
            self._policy_update_drain_complete.set()

    def raise_if_failed(self) -> None:
        if self._fatal_error is not None:
            raise PolarRolloutSchedulerError(str(self._fatal_error)) from self._fatal_error

    def drain_completed(
        self,
        *,
        max_groups: int,
        rollout_id: int,
    ) -> list[_CompletedGroup]:
        self.raise_if_failed()

        accepted: list[_CompletedGroup] = []
        while len(accepted) < max_groups:
            with self._state_lock:
                if not self._ready_groups:
                    self._ready_group_count = len(self._ready_groups)
                    break
                group_id = next(iter(self._ready_groups))
                ready = self._ready_groups.pop(group_id)
                self._ready_group_count = len(self._ready_groups)

            if ready.dropped:
                continue
            completed = ready.completed
            if completed is None:
                continue
            staleness = max(0, int(rollout_id) - completed.policy_version)
            if staleness > self.config.max_off_policy_steps:
                self._inc_metric("polar/stale_groups")
                reason = (
                    f"staleness {staleness} exceeded max_off_policy_steps="
                    f"{self.config.max_off_policy_steps}"
                )
                self._inc_metric("polar/dropped_groups")
                self._inc_metric("polar/dropped_stale_groups")
                self._inc_metric("polar/dropped_sessions", completed.session_count)
                logger.warning(
                    "Dropping stale Polar group %s task=%s: %s",
                    completed.group_id,
                    completed.task_id,
                    reason,
                )
                continue

            _annotate_accepted_samples(
                completed.samples,
                accepted_rollout_id=rollout_id,
                staleness=staleness,
                policy_version=completed.policy_version,
                scheduler_group_id=completed.group_id,
            )
            accepted.append(completed)
        return accepted

    def queue_size(self) -> int:
        with self._state_lock:
            return self._ready_group_count + self.deferred_queue.qsize()

    def snapshot_metrics(self) -> dict[str, float]:
        with self._state_lock:
            out = dict(self._metrics)
            out["polar/scheduler/active_groups"] = float(self._active_groups)
            out["polar/scheduler/active_sessions"] = float(self._active_sessions)
            out["polar/scheduler/ready_groups"] = float(self._ready_group_count)
            out["polar/scheduler/deferred_queue"] = float(self.deferred_queue.qsize())
            out["polar/scheduler/policy_version"] = float(self._policy_version)
            out["polar/scheduler/admission_paused"] = float(self._admission_paused)
            if self.config.scheduler_mode == "session_pool":
                out["polar/session_pool/active_sessions"] = float(self._active_sessions)
                out["polar/session_pool/open_groups"] = float(self._session_pool_open_groups)
                out["polar/session_pool/partial_open_groups"] = float(self._session_pool_partial_open_groups)
                out["polar/session_pool/pending_sessions"] = float(self._session_pool_pending_sessions)
                out.setdefault("polar/session_pool/submitted_sessions", 0.0)
                out.setdefault("polar/session_pool/completed_sessions", 0.0)
            return out

    # -- internal --------------------------------------------------------------

    def _run_loop(self) -> None:
        asyncio.run(self._async_loop())

    async def _async_loop(self) -> None:
        if self.config.scheduler_mode == "session_pool":
            await self._async_session_pool_loop()
            return
        await self._async_group_loop()

    async def _async_group_loop(self) -> None:
        logger.info("Async Polar rollout worker started")
        active: dict[asyncio.Task[None], _PendingGroup] = {}
        active_session_cost = 0
        wakeup = asyncio.Event()

        callback_server, callback_task = await self._start_callback_listener()
        timeout = None if self.config.request_timeout is None else httpx.Timeout(self.config.request_timeout)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                while self._running:
                    done = [t for t in active if t.done()]
                    for t in done:
                        pending = active.pop(t)
                        active_session_cost -= pending.session_cost
                        try:
                            t.result()
                        except Exception as exc:
                            logger.exception("Polar async task failed")
                            self._set_fatal(exc)
                            self._running = False
                    self._record_active_counts(active, active_session_cost)

                    while self._running and self._can_admit_group(active, active_session_cost):
                        try:
                            next_group = self._next_group_for_submission()
                        except Exception as exc:
                            self._set_fatal(exc)
                            self._running = False
                            break
                        if next_group is None:
                            break
                        session_cost = len(next_group.group)
                        if session_cost > self.config.max_session_concurrency:
                            self._set_fatal(
                                PolarRolloutSchedulerError(
                                    f"Prompt group needs {session_cost} sessions but "
                                    f"derived max_session_concurrency is "
                                    f"{self.config.max_session_concurrency}"
                                )
                            )
                            self._running = False
                            break
                        if active_session_cost + session_cost > self.config.max_session_concurrency:
                            self.deferred_queue.put(next_group)
                            break

                        gid = self._group_counter
                        self._group_counter += 1
                        submitted_rollout_id, policy_version = self._rollout_context()
                        pending = _PendingGroup(
                            group_id=gid,
                            group=next_group.group,
                            submitted_rollout_id=submitted_rollout_id,
                            policy_version=policy_version,
                            session_cost=session_cost,
                        )
                        task = asyncio.create_task(
                            self._submit_and_collect(client, pending),
                            name=f"polar-rollout-task-{gid}",
                        )
                        task.add_done_callback(lambda _: wakeup.set())
                        active[task] = pending
                        active_session_cost += session_cost
                        self._record_active_counts(active, active_session_cost)

                    if self._running:
                        try:
                            await asyncio.wait_for(wakeup.wait(), timeout=0.5)
                        except asyncio.TimeoutError:
                            pass
                        wakeup.clear()

            if active:
                logger.info("Waiting for %d in-flight Polar tasks", len(active))
                await asyncio.gather(*active.keys(), return_exceptions=True)
        finally:
            callback_server.should_exit = True
            try:
                await asyncio.wait_for(callback_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Callback listener did not shut down within 5s")
        logger.info("Async Polar rollout worker stopped")

    async def _async_session_pool_loop(self) -> None:
        logger.info("Async Polar session-pool rollout worker started")
        active: dict[asyncio.Task[TaskResult], _PendingSessionUnit] = {}
        run_pending: set[str] = set()
        next_status_poll_at: dict[str, float] = {}
        open_groups: dict[int, _SessionGroupAccumulator] = {}
        wakeup = asyncio.Event()

        callback_server, callback_task = await self._start_callback_listener()
        timeout = None if self.config.request_timeout is None else httpx.Timeout(self.config.request_timeout)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                while self._running:
                    done = [t for t in active if t.done()]
                    for task in done:
                        unit = active.pop(task)
                        self._release_session_pool_run_pending(
                            run_pending,
                            next_status_poll_at,
                            unit.task_id,
                        )
                        accumulator = open_groups.get(unit.group_id)
                        try:
                            task_result = task.result()
                            logger.info(
                                "PD-DIAG unit_result task=%s status=%s sessions=%d result_paths=%d first_session_id=%s "
                                "first_result_status=%s first_trajectory_status=%s first_trace_count=%s first_result_error=%r first_trajectory_error=%r",
                                task_result.task_id,
                                task_result.status,
                                len(task_result.results),
                                len(task_result.result_paths),
                                (task_result.results[0].session_id if task_result.results else None),
                                (task_result.results[0].status if task_result.results else None),
                                (getattr(task_result.results[0].trajectory, "status", None) if task_result.results else None),
                                (len(task_result.results[0].trajectory.traces) if task_result.results else None),
                                (task_result.results[0].error if task_result.results else None),
                                (getattr(task_result.results[0].trajectory, "error", None) if task_result.results else None),
                            )
                            if accumulator is not None and accumulator.rejected_reason is None:
                                _record_session_unit_result(
                                    config=self.config,
                                    accumulator=accumulator,
                                    unit=unit,
                                    task_result=task_result,
                                    max_tokens=_resolve_max_tokens(self.args),
                                )
                                self._inc_metric("polar/session_pool/completed_sessions")
                        except Exception as exc:
                            logger.warning(
                                "Dropping Polar session_pool group %s after unit %s failed: %s",
                                unit.group_id,
                                unit.task_id,
                                exc,
                            )
                            if accumulator is not None and accumulator.rejected_reason is None:
                                accumulator.rejected_reason = str(exc)
                                self._drop_session_pool_group(
                                    accumulator,
                                    "hard unit failure",
                                    category_metric="polar/dropped_failed_groups",
                                )

                    await self._poll_session_pool_run_release(
                        client,
                        run_pending,
                        next_status_poll_at,
                    )
                    self._finish_terminal_session_pool_groups(open_groups)
                    self._record_session_pool_counts(active, run_pending, open_groups)
                    self._maybe_mark_policy_update_drain_complete(
                        open_groups,
                        active,
                        run_pending,
                    )

                    while self._running and self._can_admit_session_pool_unit(
                        active,
                        run_pending,
                    ):
                        accumulator = self._current_partial_group(open_groups)
                        if accumulator is None:
                            if self._session_pool_draining():
                                break
                            try:
                                next_group = self._next_group_for_submission()
                            except Exception as exc:
                                self._set_fatal(exc)
                                self._running = False
                                break
                            if next_group is None:
                                break
                            with self._state_lock:
                                if self._policy_update_draining:
                                    self.deferred_queue.put(next_group)
                                    break
                                gid = self._group_counter
                                self._group_counter += 1
                                submitted_rollout_id = self._current_rollout_id
                                policy_version = self._policy_version
                            accumulator = _new_session_group_accumulator(
                                args=self.args,
                                config=self.config,
                                group_id=gid,
                                group_pos=gid,
                                group=next_group.group,
                                submitted_rollout_id=submitted_rollout_id,
                                policy_version=policy_version,
                            )
                            open_groups[gid] = accumulator

                        unit = _next_session_pool_unit(accumulator)
                        if unit is None:
                            continue
                        task = asyncio.create_task(
                            self._submit_session_unit(client, unit),
                            name=f"polar-session-unit-{unit.group_id}-{unit.sample_pos}",
                        )
                        task.add_done_callback(lambda _: wakeup.set())
                        active[task] = unit
                        run_pending.add(unit.task_id)
                        self._inc_metric("polar/session_pool/submitted_sessions")
                        self._record_session_pool_counts(active, run_pending, open_groups)
                        if (
                            self._session_pool_admission_count(active, run_pending)
                            >= self.config.max_active_sessions
                        ):
                            break

                    self._record_session_pool_counts(active, run_pending, open_groups)
                    self._maybe_mark_policy_update_drain_complete(
                        open_groups,
                        active,
                        run_pending,
                    )

                    if self._running:
                        try:
                            await asyncio.wait_for(wakeup.wait(), timeout=0.5)
                        except asyncio.TimeoutError:
                            pass
                        wakeup.clear()

            if active:
                logger.info("Waiting for %d in-flight Polar session_pool tasks", len(active))
                await asyncio.gather(*active.keys(), return_exceptions=True)
        finally:
            callback_server.should_exit = True
            try:
                await asyncio.wait_for(callback_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Callback listener did not shut down within 5s")
        logger.info("Async Polar session-pool rollout worker stopped")

    async def _start_callback_listener(self) -> tuple[uvicorn.Server, asyncio.Task[None]]:
        """Bind a FastAPI listener for TaskResult callbacks."""
        if self.config.submit_mode == "operator_samples":
            server = _NoopCallbackServer()

            async def idle() -> None:
                while not server.should_exit:
                    await asyncio.sleep(0.05)

            return server, asyncio.create_task(idle(), name="polar-noop-callback-listener")

        app = FastAPI()

        @app.post("/callback/task_result")
        async def on_task_result(request: Request) -> dict[str, Any]:
            payload = await request.json()
            task_id = payload.get("task_id") if isinstance(payload, dict) else None
            if not task_id:
                return {"ok": False, "reason": "missing task_id"}
            try:
                result = TaskResult.model_validate(payload)
            except Exception:
                logger.exception("Invalid callback payload for task %s", task_id)
                return {"ok": False, "reason": "invalid payload"}
            self._task_results[task_id] = result
            event = self._task_events.get(task_id)
            if event is not None:
                event.set()
            return {"ok": True}

        config = uvicorn.Config(
            app=app, host=self.config.callback_host, port=0,
            log_level="warning", lifespan="on",
        )
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve(), name="polar-callback-listener")
        while not server.started:
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        self._callback_url = f"http://{self.config.callback_host}:{port}/callback/task_result"
        logger.info("Polar trainer callback listener bound to %s", self._callback_url)
        return server, task

    async def _submit_and_collect(
        self, client: httpx.AsyncClient, pending: _PendingGroup
    ) -> None:
        last_error: BaseException | None = None

        if self._running:
            try:
                completed = await self._submit_attempt(client, pending)
                await self._emit_completed(completed)
                return
            except Exception as exc:
                last_error = exc

        if last_error is None:
            return

        if _is_zero_trainable_error(last_error):
            category_metric = "polar/dropped_zero_trainable_groups"
            reason = "zero trainable tokens"
        elif isinstance(last_error, PolarLowCompleteAcceptFractionError):
            category_metric = "polar/dropped_low_complete_fraction_groups"
            reason = "low complete accept fraction"
        elif isinstance(last_error, RolloutLogprobError):
            category_metric = "polar/dropped_logprob_error_groups"
            reason = "rollout logprob error"
        else:
            category_metric = "polar/dropped_failed_groups"
            reason = "task failure"

        self._inc_metric("polar/dropped_groups")
        self._inc_metric(category_metric)
        self._inc_metric("polar/dropped_sessions", pending.session_cost)
        logger.warning(
            "Dropping Polar group %s because of %s: %s",
            pending.group_id,
            reason,
            last_error,
        )
        return

    async def _submit_session_unit(
        self,
        client: httpx.AsyncClient,
        unit: _PendingSessionUnit,
    ) -> TaskResult:
        payload = _build_session_unit_payload(
            args=self.args,
            config=self.config,
            unit=unit,
        )
        return await self._submit_payload(client, payload)

    async def _poll_session_pool_run_release(
        self,
        client: httpx.AsyncClient,
        run_pending: set[str],
        next_status_poll_at: dict[str, float],
    ) -> None:
        if not self.config.session_pool_release_on_postrun or not run_pending:
            return
        now = time.monotonic()
        due_task_ids: list[str] = []
        for task_id in list(run_pending):
            if next_status_poll_at.get(task_id, 0.0) > now:
                continue
            next_status_poll_at[task_id] = now + _SESSION_POOL_RUN_RELEASE_POLL_SECONDS
            due_task_ids.append(task_id)
        if not due_task_ids:
            return
        statuses = await asyncio.gather(
            *(self._session_pool_task_status(client, task_id) for task_id in due_task_ids),
            return_exceptions=True,
        )
        for task_id, status in zip(due_task_ids, statuses, strict=True):
            if isinstance(status, BaseException):
                self._inc_metric("polar/session_pool/run_release_poll_failures")
                logger.debug(
                    "Failed to poll Polar session_pool live status for task %s",
                    task_id,
                    exc_info=True,
                )
                continue
            if status in _SESSION_POOL_RUN_RELEASE_STATUSES:
                self._release_session_pool_run_pending(
                    run_pending,
                    next_status_poll_at,
                    task_id,
                )
                self._inc_metric("polar/session_pool/run_released_sessions")

    async def _session_pool_task_status(
        self,
        client: httpx.AsyncClient,
        task_id: str,
    ) -> str | None:
        base_url = self.config.rollout_server_url
        response = await client.get(
            f"{base_url}/tasks/{task_id}/sessions",
            timeout=_SESSION_POOL_RUN_RELEASE_POLL_TIMEOUT_SECONDS,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.json()
        sessions = payload.get("sessions") if isinstance(payload, dict) else None
        if not isinstance(sessions, list) or not sessions:
            return None
        statuses = [
            str(session.get("status"))
            for session in sessions
            if isinstance(session, dict) and session.get("status") is not None
        ]
        if not statuses:
            return None
        if all(status in _SESSION_POOL_RUN_RELEASE_STATUSES for status in statuses):
            return statuses[-1]
        return None

    def _release_session_pool_run_pending(
        self,
        run_pending: set[str],
        next_status_poll_at: dict[str, float],
        task_id: str,
    ) -> None:
        run_pending.discard(task_id)
        next_status_poll_at.pop(task_id, None)

    def _finish_terminal_session_pool_groups(
        self,
        open_groups: dict[int, _SessionGroupAccumulator],
    ) -> None:
        for group_id, accumulator in list(open_groups.items()):
            if accumulator.rejected_reason is not None:
                open_groups.pop(group_id, None)
                self._mark_group_dropped(group_id)
                continue
            if not accumulator.complete:
                continue
            try:
                completed = _completed_group_from_session_accumulator(
                    self.config,
                    accumulator,
                )
                self._mark_group_ready(completed)
                self._inc_metric("polar/session_pool/completed_groups")
            except Exception as exc:
                if _is_zero_trainable_error(exc):
                    category_metric = "polar/dropped_zero_trainable_groups"
                    reason = "zero trainable tokens"
                elif isinstance(exc, PolarLowCompleteAcceptFractionError):
                    category_metric = "polar/dropped_low_complete_fraction_groups"
                    reason = "low complete accept fraction"
                elif isinstance(exc, RolloutLogprobError):
                    category_metric = "polar/dropped_logprob_error_groups"
                    reason = "rollout logprob error"
                else:
                    category_metric = "polar/dropped_failed_groups"
                    reason = "task failure"
                self._drop_session_pool_group(
                    accumulator,
                    reason,
                    category_metric=category_metric,
                )
            finally:
                open_groups.pop(group_id, None)

    def _drop_session_pool_group(
        self,
        accumulator: _SessionGroupAccumulator,
        reason: str,
        *,
        category_metric: str,
    ) -> None:
        if accumulator.rejected_reason is None:
            accumulator.rejected_reason = reason
        self._inc_metric("polar/dropped_groups")
        self._inc_metric("polar/session_pool/dropped_groups")
        self._inc_metric(category_metric)
        dropped_sessions = max(accumulator.group_size, accumulator.submitted_count)
        self._inc_metric("polar/dropped_sessions", dropped_sessions)
        self._inc_metric("polar/session_pool/dropped_sessions", dropped_sessions)
        logger.warning(
            "Dropping Polar session_pool group %s task=%s because of %s: %s",
            accumulator.group_id,
            accumulator.parent_task_id,
            reason,
            accumulator.rejected_reason,
        )
        self._mark_group_dropped(accumulator.group_id)

    def _current_partial_group(
        self,
        open_groups: dict[int, _SessionGroupAccumulator],
    ) -> _SessionGroupAccumulator | None:
        partials = [
            accumulator
            for accumulator in open_groups.values()
            if accumulator.rejected_reason is None and accumulator.partial
        ]
        if len(partials) > 1:
            raise PolarRolloutSchedulerError(
                f"session_pool invariant violated: {len(partials)} partial open groups"
            )
        if not partials:
            return None
        return min(partials, key=lambda acc: acc.group_id)

    def _can_admit_session_pool_unit(
        self,
        active: dict[asyncio.Task[TaskResult], _PendingSessionUnit],
        run_pending: set[str],
    ) -> bool:
        with self._state_lock:
            admission_paused = self._admission_paused
        if admission_paused:
            return False
        if (
            self._session_pool_admission_count(active, run_pending)
            >= self.config.max_active_sessions
        ):
            return False
        return True

    def _session_pool_admission_count(
        self,
        active: dict[asyncio.Task[TaskResult], _PendingSessionUnit],
        run_pending: set[str],
    ) -> int:
        if self.config.session_pool_release_on_postrun:
            return len(run_pending)
        return len(active)

    def _session_pool_draining(self) -> bool:
        with self._state_lock:
            return self._policy_update_draining

    def _maybe_mark_policy_update_drain_complete(
        self,
        open_groups: dict[int, _SessionGroupAccumulator],
        active: dict[asyncio.Task[TaskResult], _PendingSessionUnit],
        run_pending: set[str] | None = None,
    ) -> None:
        with self._state_lock:
            if not self._policy_update_draining:
                return
            if self.config.session_pool_release_on_postrun:
                has_partial_group = any(
                    accumulator.rejected_reason is None and accumulator.partial
                    for accumulator in open_groups.values()
                )
                if has_partial_group or run_pending:
                    return
            elif open_groups or active:
                return
            started_at = self._policy_update_drain_started_at
            if started_at is not None:
                self._metrics["polar/session_pool/policy_update_drain_seconds"] = (
                    self._metrics.get("polar/session_pool/policy_update_drain_seconds", 0.0)
                    + (time.monotonic() - started_at)
                )
                self._policy_update_drain_started_at = None
            self._policy_update_drain_complete.set()

    def _record_session_pool_counts(
        self,
        active: dict[asyncio.Task[TaskResult], _PendingSessionUnit],
        run_pending: set[str],
        open_groups: dict[int, _SessionGroupAccumulator],
    ) -> None:
        partial_open_groups = sum(
            1
            for accumulator in open_groups.values()
            if accumulator.rejected_reason is None and accumulator.partial
        )
        pending_sessions = sum(
            max(0, accumulator.group_size - accumulator.submitted_count)
            for accumulator in open_groups.values()
            if accumulator.rejected_reason is None
        )
        with self._state_lock:
            self._active_groups = len(open_groups)
            self._active_sessions = self._session_pool_admission_count(active, run_pending)
            self._session_pool_open_groups = len(open_groups)
            self._session_pool_partial_open_groups = partial_open_groups
            self._session_pool_pending_sessions = pending_sessions
            self._metrics["polar/session_pool/run_pending_sessions"] = float(len(run_pending))
            self._metrics["polar/session_pool/final_pending_sessions"] = float(len(active))

    async def _submit_attempt(
        self,
        client: httpx.AsyncClient,
        pending: _PendingGroup,
    ) -> _CompletedGroup:
        payload = _build_submission_payload(
            args=self.args, config=self.config, group=pending.group,
            rollout_id=pending.group_id, task_position=0,
        )
        payload["task_id"] = str(payload["task_id"])
        _attach_scheduler_metadata(
            payload,
            group_id=pending.group_id,
            policy_version=pending.policy_version,
            rollout_step=pending.submitted_rollout_id,
        )
        task_result = await self._submit_payload_result(client, payload)

        rejection_reason = self._task_rejection_reason(task_result, pending.group)
        if rejection_reason is not None:
            raise PolarRolloutSchedulerError(
                f"Task {task_result.task_id} cannot be accepted: {rejection_reason}"
            )

        group_samples = _convert_task_result_to_samples(
            self.config, task_result, pending.group,
            max_tokens=_resolve_max_tokens(self.args),
        )
        if not group_samples:
            raise PolarRolloutSchedulerError(f"Task {task_result.task_id} converted to zero samples")
        if not _has_trainable_tokens(group_samples):
            raise PolarRolloutSchedulerError(
                f"Task {task_result.task_id} produced zero trainable tokens"
            )
        rejection_reason = _low_complete_accept_fraction_rejection_reason(
            self.config, task_result, group_samples
        )
        if rejection_reason is not None:
            raise PolarLowCompleteAcceptFractionError(
                f"Task {task_result.task_id} cannot be accepted: {rejection_reason}"
            )

        return _CompletedGroup(
            group_id=pending.group_id,
            group=pending.group,
            samples=group_samples,
            task_id=task_result.task_id,
            submitted_rollout_id=pending.submitted_rollout_id,
            policy_version=pending.policy_version,
            session_count=len(task_result.results),
        )

    async def _emit_completed(self, completed: _CompletedGroup) -> None:
        self._mark_group_ready(completed)

    def _mark_group_ready(self, completed: _CompletedGroup) -> None:
        with self._state_lock:
            self._ready_groups[completed.group_id] = _ReadyGroup(completed=completed)
            self._ready_group_count = len(self._ready_groups)
        self._inc_metric("polar/completed_groups")

    def _mark_group_dropped(self, group_id: int) -> None:
        with self._state_lock:
            self._ready_groups.setdefault(group_id, _ReadyGroup(dropped=True))
            self._ready_group_count = len(self._ready_groups)

    def _next_group_for_submission(self) -> _DeferredGroup | None:
        try:
            deferred = self.deferred_queue.get_nowait()
            self._inc_metric("polar/deferred_queue_dequeues")
            return deferred
        except queue.Empty:
            pass

        groups = self.data_source.get_samples(1)
        if not groups:
            return None
        group = groups[0]
        if not group:
            raise PolarRolloutSchedulerError("Slime data source returned an empty sample group")
        return _DeferredGroup(group=group)

    def _can_admit_group(
        self,
        active: dict[asyncio.Task[None], _PendingGroup],
        active_session_cost: int,
    ) -> bool:
        with self._state_lock:
            if self._admission_paused:
                return False
        if len(active) >= self.config.max_concurrency:
            return False
        if active_session_cost >= self.config.max_session_concurrency:
            return False
        return len(active) < (self._batch_size * self.config.max_async_level)

    def _task_rejection_reason(self, task_result: TaskResult, group: list[Any]) -> str | None:
        if task_result.status != "completed":
            return f"task status={task_result.status}"
        if not task_result.results:
            return "empty task results"
        if len(task_result.results) != len(group):
            return f"session count {len(task_result.results)} != expected {len(group)}"
        return None

    def _rollout_context(self) -> tuple[int, int]:
        with self._state_lock:
            return self._current_rollout_id, self._policy_version

    def _record_active_counts(
        self,
        active: dict[asyncio.Task[None], _PendingGroup],
        active_session_cost: int,
    ) -> None:
        with self._state_lock:
            self._active_groups = len(active)
            self._active_sessions = active_session_cost

    def _inc_metric(self, key: str, amount: float = 1.0) -> None:
        with self._state_lock:
            self._metrics[key] = self._metrics.get(key, 0.0) + amount

    def _set_fatal(self, exc: BaseException) -> None:
        with self._state_lock:
            if self._fatal_error is None:
                self._fatal_error = exc

    async def _submit_with_callback(
        self, client: httpx.AsyncClient, payload: dict[str, Any]
    ) -> TaskResult:
        """Submit a task, wait on its completion event, and fall back to polling."""
        task_id = payload["task_id"]
        # Register event BEFORE submit so a fast callback cannot arrive first.
        event = asyncio.Event()
        self._task_events[task_id] = event
        payload["callback_url"] = self._callback_url
        base_url = self.config.rollout_server_url
        try:
            resp = await client.post(
                f"{base_url}/rollout/task/submit",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return await self._await_task_result(client, task_id, event)
        finally:
            self._task_events.pop(task_id, None)
            self._task_results.pop(task_id, None)

    async def _submit_payload(
        self, client: httpx.AsyncClient, payload: dict[str, Any]
    ) -> TaskResult:
        if self.config.submit_mode == "operator_samples":
            return await _submit_and_wait_for_task(
                client,
                self.config.rollout_server_url,
                payload,
                submit_path="/rollout/operator_samples/submit",
            )
        return await self._submit_with_callback(client, payload)

    async def _submit_payload_with_callback(
        self, client: httpx.AsyncClient, payload: dict[str, Any]
    ) -> TaskResult:
        """Submit one logical Slime group, chunking Polar sessions if needed."""
        chunks = _chunk_task_payloads(
            payload,
            max_sessions_per_task=self.config.max_sessions_per_task,
        )
        if len(chunks) > 1:
            self._inc_metric("polar/chunked_tasks")
            self._inc_metric("polar/task_chunks", len(chunks))

        async def submit_one(chunk: dict[str, Any]) -> TaskResult:
            return await self._submit_with_callback(client, chunk)

        return await _submit_payload_in_chunks(
            payload,
            max_sessions_per_task=self.config.max_sessions_per_task,
            submit_one=submit_one,
        )

    async def _submit_payload_result(
        self, client: httpx.AsyncClient, payload: dict[str, Any]
    ) -> TaskResult:
        if self.config.submit_mode == "task_request":
            return await self._submit_payload_with_callback(client, payload)
        chunks = _chunk_task_payloads(
            payload,
            max_sessions_per_task=self.config.max_sessions_per_task,
        )
        if len(chunks) > 1:
            self._inc_metric("polar/chunked_tasks")
            self._inc_metric("polar/task_chunks", len(chunks))

        async def submit_one(chunk: dict[str, Any]) -> TaskResult:
            return await self._submit_payload(client, chunk)

        return await _submit_payload_in_chunks(
            payload,
            max_sessions_per_task=self.config.max_sessions_per_task,
            submit_one=submit_one,
        )

    async def _await_task_result(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        event: asyncio.Event,
    ) -> TaskResult:
        """Wait on the completion event with a defensive 60s fallback poll."""
        base_url = self.config.rollout_server_url
        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=_CALLBACK_FALLBACK_POLL_SECONDS)
            except asyncio.TimeoutError:
                status_resp = await client.get(f"{base_url}/rollout/task/{task_id}")
                status_resp.raise_for_status()
                status = TaskStatus.model_validate(status_resp.json())
                if status.status in ("completed", "failed"):
                    return TaskResult(
                        task_id=task_id, status=status.status,
                        results=status.results, result_paths=status.result_paths,
                    )
                continue
            result = self._task_results.get(task_id)
            if result is not None:
                return result
            # Race: event set but result missing — re-poll once.
            status_resp = await client.get(f"{base_url}/rollout/task/{task_id}")
            status_resp.raise_for_status()
            status = TaskStatus.model_validate(status_resp.json())
            return TaskResult(
                task_id=task_id, status=status.status,
                results=status.results, result_paths=status.result_paths,
            )


# ---------------------------------------------------------------------------
# One-shot eval rollout
# ---------------------------------------------------------------------------
async def _run_eval_rollout(
    args: Any,
    rollout_id: int,
    data_source: Any,
) -> Any:
    config = resolve_polar_slime_config(args)
    eval_datasets = list(getattr(args, "eval_datasets", []) or [])
    if eval_datasets:
        data: dict[str, dict[str, Any]] = {}
        metrics: dict[str, Any] = {}
        for dataset_cfg in eval_datasets:
            dataset_name, dataset_data, dataset_metrics = await _run_eval_dataset(
                args=args,
                config=config,
                rollout_id=rollout_id,
                dataset_cfg=dataset_cfg,
            )
            data[dataset_name] = dataset_data
            metrics.update(_prefix_eval_metrics(dataset_name, dataset_metrics))

        RolloutFnEvalOutput = _load_rollout_eval_output_type()
        return RolloutFnEvalOutput(data=data, metrics=metrics)

    logger.warning(
        "Polar eval called without args.eval_datasets; falling back to the training data source. "
        "Pass --eval-prompt-data to evaluate validation prompts."
    )
    sample_groups = _pull_sample_groups(data_source, args.rollout_batch_size)
    dataset_data, metrics = await _submit_eval_groups(
        args=args,
        config=config,
        dataset_name=config.eval_dataset_name,
        rollout_id=rollout_id,
        sample_groups=sample_groups,
    )
    RolloutFnEvalOutput = _load_rollout_eval_output_type()
    return RolloutFnEvalOutput(
        data={config.eval_dataset_name: dataset_data},
        metrics=metrics,
    )


async def _run_eval_dataset(
    *,
    args: Any,
    config: PolarSlimeConfig,
    rollout_id: int,
    dataset_cfg: Any,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    dataset_name = str(getattr(dataset_cfg, "name", "") or config.eval_dataset_name)
    sample_groups = _load_eval_sample_groups(args, dataset_cfg)
    dataset_data, metrics = await _submit_eval_groups(
        args=args,
        config=config,
        dataset_name=dataset_name,
        rollout_id=rollout_id,
        sample_groups=sample_groups,
    )
    return dataset_name, dataset_data, metrics


async def _submit_eval_groups(
    *,
    args: Any,
    config: PolarSlimeConfig,
    dataset_name: str,
    rollout_id: int,
    sample_groups: list[list[Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not sample_groups:
        raise ValueError("Polar eval dataset produced no sample groups")

    timeout = None if config.request_timeout is None else httpx.Timeout(config.request_timeout)
    semaphore = asyncio.Semaphore(config.max_concurrency)

    async def _run_one(position: int, group: list[Any]) -> TaskResult:
        async with semaphore:
            payload = _build_submission_payload(
                args=args, config=config, group=group,
                rollout_id=rollout_id, task_position=position,
            )
            payload["task_id"] = _eval_task_id(
                payload["task_id"],
                dataset_name=dataset_name,
                rollout_id=rollout_id,
                position=position,
            )
            _attach_scheduler_metadata(
                payload,
                group_id=position,
                policy_version=rollout_id,
                rollout_step=rollout_id,
            )

            async def submit_one(chunk: dict[str, Any]) -> TaskResult:
                submit_path = (
                    "/rollout/operator_samples/submit"
                    if config.submit_mode == "operator_samples"
                    else "/rollout/task/submit"
                )
                return await _submit_and_wait_for_task(
                    client,
                    config.rollout_server_url,
                    chunk,
                    submit_path=submit_path,
                )

            return await _submit_payload_in_chunks(
                payload,
                max_sessions_per_task=config.max_sessions_per_task,
                submit_one=submit_one,
            )

    async with httpx.AsyncClient(timeout=timeout) as client:
        task_results = await asyncio.gather(
            *(_run_one(pos, g) for pos, g in enumerate(sample_groups))
        )

    output_groups: list[list[Any]] = []
    max_tokens = _resolve_max_tokens(args)
    for group, task_result in zip(sample_groups, task_results, strict=True):
        output_groups.append(
            _convert_task_result_to_samples(
                config, task_result, group,
                max_tokens=max_tokens,
            )
        )

    metrics = _build_metrics(
        config,
        task_results,
        output_groups,
        reward_filter="completed",
    )
    flat_samples = [sample for group in output_groups for sample in group]
    reward_samples = _completed_session_samples(flat_samples)

    return {
        "rewards": [_extract_sample_reward(s, config.reward_key) for s in reward_samples],
        "all_rewards": [_extract_sample_reward(s, config.reward_key) for s in flat_samples],
        "truncated": [_is_truncated(s) for s in reward_samples],
        "all_truncated": [_is_truncated(s) for s in flat_samples],
        "samples": flat_samples,
    }, metrics


def _eval_task_id(base_task_id: Any, *, dataset_name: str, rollout_id: int, position: int) -> str:
    """Namespace eval task ids away from train task ids.

    Training ids commonly use ``{rollout_id}-{sample.group_index}``; eval uses
    ``position`` as group index, so eval 11 / item 11 would collide with train
    group 11. A suffix keeps task polling and persisted result dirs separate.
    """
    safe_dataset = "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in dataset_name
    )
    return f"{base_task_id}-eval-{safe_dataset}-{rollout_id}-{position}"


def _completed_session_samples(samples: list[Any]) -> list[Any]:
    return [
        sample for sample in samples
        if _sample_session_status(sample) == "COMPLETED"
        and not bool(
            (getattr(sample, "metadata", {}) or {})
            .get("polar", {})
            .get("placeholder")
        )
    ]


def _sample_session_status(sample: Any) -> str | None:
    polar_meta = (getattr(sample, "metadata", {}) or {}).get("polar", {})
    status = polar_meta.get("session_status")
    return getattr(status, "value", status)


def _load_eval_sample_groups(args: Any, dataset_cfg: Any) -> list[list[Any]]:
    Sample = _load_sample_type()
    path = str(getattr(dataset_cfg, "path"))
    input_key = getattr(dataset_cfg, "input_key", None) or getattr(args, "input_key", "prompt")
    label_key = getattr(dataset_cfg, "label_key", None) or getattr(args, "label_key", None)
    metadata_key = getattr(dataset_cfg, "metadata_key", None) or getattr(args, "metadata_key", "metadata")
    tool_key = getattr(dataset_cfg, "tool_key", None) or getattr(args, "tool_key", None)
    group_size = int(
        getattr(dataset_cfg, "n_samples_per_eval_prompt", None)
        or getattr(args, "n_samples_per_eval_prompt", None)
        or 1
    )
    if group_size <= 0:
        raise ValueError("n_samples_per_eval_prompt must be positive")

    groups: list[list[Any]] = []
    sample_index = 0
    for prompt_index, row in enumerate(_read_jsonl_rows(path)):
        if input_key not in row:
            raise KeyError(f"Eval row {prompt_index} in {path} missing input key {input_key!r}")

        metadata = _inject_eval_metadata(dataset_cfg, row.get(metadata_key))
        if tool_key and tool_key in row:
            tools = row[tool_key]
            if isinstance(tools, str):
                tools = json.loads(tools)
            metadata["tools"] = tools

        group: list[Any] = []
        for _ in range(group_size):
            sample = Sample(
                prompt=copy.deepcopy(row[input_key]),
                label=row.get(label_key) if label_key else None,
                metadata=copy.deepcopy(metadata),
                group_index=prompt_index,
                index=sample_index,
            )
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            group.append(sample)
            sample_index += 1
        groups.append(group)

    return groups


def _read_jsonl_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Eval row {line_number} in {path} is not a JSON object")
            rows.append(row)
    return rows


def _inject_eval_metadata(dataset_cfg: Any, sample_metadata: Any) -> dict[str, Any]:
    inject = getattr(dataset_cfg, "inject_metadata", None)
    if callable(inject):
        metadata = inject(sample_metadata)
    elif isinstance(sample_metadata, dict):
        metadata = dict(sample_metadata)
    else:
        metadata = {}
    return metadata


def _prefix_eval_metrics(dataset_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    prefixed: dict[str, Any] = {}
    for key, value in metrics.items():
        if key.startswith("polar/"):
            prefixed[f"polar/eval/{dataset_name}/{key.removeprefix('polar/')}"] = value
        else:
            prefixed[f"polar/eval/{dataset_name}/{key}"] = value
    return prefixed


def _pull_sample_groups(data_source: Any, batch_size: int) -> list[list[Any]]:
    getter = getattr(data_source, "get_samples", None)
    if callable(getter):
        groups = getter(batch_size)
    elif callable(data_source):
        groups = data_source(batch_size)
    else:
        raise ValueError("data_source must expose get_samples(num_samples) or be callable")
    if not isinstance(groups, list):
        raise ValueError("data_source.get_samples must return a list of sample groups")
    for group in groups:
        if not group:
            raise ValueError("Slime data source returned an empty sample group")
    return groups


def _build_metrics(
    config: PolarSlimeConfig,
    task_results: list[TaskResult],
    output_groups: list[list[Any]],
    *,
    reward_filter: str = "all",
) -> dict[str, Any]:
    flat_samples = [sample for group in output_groups for sample in group]
    all_rewards = [_extract_sample_reward(s, config.reward_key) for s in flat_samples]
    completed_rewards = [
        _extract_sample_reward(s, config.reward_key)
        for s in _completed_session_samples(flat_samples)
    ]
    if reward_filter == "all":
        rewards = all_rewards
    elif reward_filter == "completed":
        rewards = completed_rewards
    else:
        raise ValueError("reward_filter must be 'all' or 'completed'")
    metrics: dict[str, Any] = {}
    metrics.update(_polar_extra_metrics(flat_samples, rewards, config.reward_key))
    return metrics


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def generate_rollout_polar_async(args: Any, rollout_id: int, data_source: Any, evaluation: bool = False) -> Any:
    """Slime-compatible async rollout entrypoint.

    Training runs are served by a persistent background worker that pulls
    from ``data_source`` and drains completed groups on each call.
    Evaluation runs are served by a one-shot submit+poll batch over the
    same async HTTP surface.
    """
    if evaluation:
        return asyncio.run(_run_eval_rollout(args, rollout_id, data_source))

    async_worker = get_global_async_worker(args, data_source)
    async_worker.set_rollout_context(rollout_id)
    target = getattr(args, "rollout_batch_size", 1)

    data: list[list[Any]] = []
    start = time.monotonic()
    last_progress = start

    while len(data) < target:
        made_progress = False
        completed_groups = async_worker.drain_completed(
            max_groups=target - len(data),
            rollout_id=rollout_id,
        )
        for completed in completed_groups:
            data.append(completed.samples)
            made_progress = True

        now = time.monotonic()
        if made_progress:
            last_progress = now
        elif now - last_progress > 60:
            logger.warning(
                "No progress for 60s. Queue=%d, accepted=%d/%d",
                async_worker.queue_size(), len(data), target,
            )
            last_progress = now

        if len(data) < target:
            time.sleep(0.05)

    elapsed = time.monotonic() - start
    logger.info("Async rollout collected %d groups in %.1fs (queue=%d)", len(data), elapsed, async_worker.queue_size())

    _maybe_dump_longest_trace_artifact(rollout_id, data)

    RolloutFnTrainOutput = _load_rollout_train_output_type()
    flat = [s for g in data for s in g]
    rewards = [_extract_sample_reward(s, async_worker.config.reward_key) for s in flat]
    metrics: dict[str, Any] = {}
    metrics.update(_polar_extra_metrics(flat, rewards, async_worker.config.reward_key))
    return RolloutFnTrainOutput(samples=data, metrics=metrics)


def _maybe_dump_longest_trace_artifact(
    rollout_id: int, data: list[list[Any]], *, interval: int = _LONGEST_TRACE_ARTIFACT_INTERVAL
) -> None:
    """Dump the longest session in this rollout's batch as a wandb artifact.

    Groups samples by ``session_id``, picks the session with the largest
    aggregated assistant tokens, and writes its full message chain (per
    trace) to a JSON artifact. Silently no-ops if wandb isn't initialized.
    """
    if interval <= 0 or rollout_id % interval != 0:
        return
    try:
        import wandb
    except ImportError:
        return
    if getattr(wandb, "run", None) is None:
        return

    by_session: dict[str, list[Any]] = {}
    for group in data:
        for sample in group:
            sid = getattr(sample, "session_id", None) or "unknown"
            by_session.setdefault(sid, []).append(sample)
    if not by_session:
        return

    def _session_tokens(samples: list[Any]) -> int:
        return sum(int(getattr(s, "response_length", 0) or 0) for s in samples)

    longest_sid, longest_samples = max(by_session.items(), key=lambda kv: _session_tokens(kv[1]))
    total_tokens = _session_tokens(longest_samples)
    if total_tokens <= 0:
        return

    longest_samples = sorted(
        longest_samples,
        key=lambda s: int((s.metadata.get("polar") or {}).get("trace_index", 0) or 0),
    )
    traces = []
    for sample in longest_samples:
        polar_meta = sample.metadata.get("polar") or {}
        trace_debug = polar_meta.get("trace_debug") or {}
        status = getattr(sample, "status", None)
        traces.append({
            "trace_index": polar_meta.get("trace_index"),
            "finish_reason": trace_debug.get("finish_reason"),
            "response_length": int(getattr(sample, "response_length", 0) or 0),
            "status": getattr(status, "value", None) if status is not None else None,
            "prompt_messages": sample.prompt if isinstance(sample.prompt, list) else [],
            "response_messages": trace_debug.get("response_messages") or [],
        })

    first = longest_samples[0]
    first_meta = first.metadata.get("polar") or {}
    reward = getattr(first, "reward", None)
    if isinstance(reward, dict):
        session_reward = float(reward.get("score", 0.0))
    elif isinstance(reward, (int, float)):
        session_reward = float(reward)
    else:
        session_reward = 0.0

    payload = {
        "rollout_id": int(rollout_id),
        "session_id": longest_sid,
        "task_id": first_meta.get("task_id"),
        "node_id": first_meta.get("node_id"),
        "total_assistant_tokens": int(total_tokens),
        "session_reward": session_reward,
        "num_traces": len(traces),
        "traces": traces,
    }

    try:
        with tempfile.TemporaryDirectory() as tmp:
            fpath = Path(tmp) / f"longest_trace_r{rollout_id}.json"
            fpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            artifact = wandb.Artifact(
                name=f"longest_trace_r{rollout_id}", type="rollout-trace"
            )
            artifact.add_file(str(fpath))
            wandb.run.log_artifact(artifact)
    except Exception:
        logger.exception("Failed to log longest-trace wandb artifact")
        return

    logger.info(
        "Logged longest-trace artifact rollout=%d session=%s traces=%d tokens=%d",
        rollout_id, longest_sid, len(traces), total_tokens,
    )


def _group_index_for(group: list[Any]) -> int:
    if group and getattr(group[0], "group_index", None) is not None:
        return int(group[0].group_index)
    return -1


def _extract_sample_reward(sample: Any, reward_key: str) -> float:
    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        if reward_key in reward:
            return float(reward[reward_key])
        if "score" in reward:
            return float(reward["score"])
    if isinstance(reward, (int, float)):
        return float(reward)
    return 0.0


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[lower])
    weight = rank - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)



def _tokens_per_second(tokens: float, duration_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    return float(tokens / duration_s)



def _build_rollout_benchmark_metrics(flat_samples: list[Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    session_seen: set[str] = set()
    session_windows: dict[str, tuple[float, float]] = {}
    second_bucket_counts: Counter[int] = Counter()
    total_input_tokens = 0
    total_generated_tokens = 0
    request_count = 0
    ttfts: list[float] = []
    tpots: list[float] = []
    itls: list[float] = []

    for sample in flat_samples:
        polar_meta = sample.metadata.get("polar", {})
        session_id = polar_meta.get("session_id")
        if not session_id or session_id in session_seen:
            continue
        session_seen.add(session_id)
        if polar_meta.get("placeholder"):
            continue
        if _sample_session_status(sample) != "COMPLETED":
            continue

        timing = polar_meta.get("timing") or {}
        register_to_init_queue_ms = float(timing.get("register_to_init_queue_ms", 0.0) or 0.0)
        init_ms = float(timing.get("init_ms", 0.0) or 0.0)
        run_ms = float(timing.get("run_ms", 0.0) or 0.0)
        postrun_ms = float(timing.get("postrun_ms", 0.0) or 0.0)
        ttft_s = (register_to_init_queue_ms + init_ms) / 1000.0
        run_s = run_ms / 1000.0
        postrun_s = postrun_ms / 1000.0
        if ttft_s <= 0.0 or run_s <= 0.0:
            continue

        token_counts = polar_meta.get("token_counts") or {}
        prompt_tokens = int(token_counts.get("prompt_tokens", 0) or 0)
        response_tokens = int(token_counts.get("response_tokens", getattr(sample, "response_length", 0) or 0) or 0)
        total_input_tokens += prompt_tokens
        total_generated_tokens += response_tokens
        request_count += 1

        if response_tokens > 0:
            ttfts.append(ttft_s * 1000.0)
        if response_tokens > 1:
            tpot_s = run_s / (response_tokens - 1)
            tpots.append(tpot_s * 1000.0)
            itls.extend([tpot_s * 1000.0] * (response_tokens - 1))

        start_s = 0.0
        end_s = ttft_s + run_s + postrun_s
        session_windows[session_id] = (start_s, end_s)

        if response_tokens > 0:
            first_token_time = ttft_s
            second_bucket_counts[int(first_token_time)] += 1
            if response_tokens > 1:
                tpot_s = run_s / (response_tokens - 1)
                current_time = first_token_time
                for _ in range(response_tokens - 1):
                    current_time += tpot_s
                    second_bucket_counts[int(current_time)] += 1

    if request_count == 0 or not session_windows:
        return out

    min_start = min(start for start, _ in session_windows.values())
    max_end = max(end for _, end in session_windows.values())
    benchmark_duration_s = max_end - min_start
    if benchmark_duration_s <= 0.0:
        benchmark_duration_s = 1e-9

    concurrent_per_second: Counter[int] = Counter()
    for start_s, end_s in session_windows.values():
        request_start_second = int(start_s - min_start)
        request_end_second = int(end_s - min_start)
        for second in range(request_start_second, request_end_second + 1):
            concurrent_per_second[second] += 1

    peak_output_tokens_per_s = float(max(second_bucket_counts.values(), default=0))
    peak_concurrent_requests = float(max(concurrent_per_second.values(), default=0))

    def _summary(values: list[float], prefix: str) -> None:
        if not values:
            return
        sorted_values = sorted(values)
        out[f"{prefix}_mean_ms"] = float(sum(sorted_values) / len(sorted_values))
        out[f"{prefix}_median_ms"] = _percentile(sorted_values, 0.5)
        out[f"{prefix}_p99_ms"] = _percentile(sorted_values, 0.99)

    out["rollout_bench/total_input_tokens"] = float(total_input_tokens)
    out["rollout_bench/total_generated_tokens"] = float(total_generated_tokens)
    out["rollout_bench/request_throughput"] = _tokens_per_second(request_count, benchmark_duration_s)
    out["rollout_bench/output_throughput"] = _tokens_per_second(total_generated_tokens, benchmark_duration_s)
    out["rollout_bench/peak_output_token_throughput"] = peak_output_tokens_per_s
    out["rollout_bench/peak_concurrent_requests"] = peak_concurrent_requests
    out["rollout_bench/total_token_throughput"] = _tokens_per_second(
        total_input_tokens + total_generated_tokens, benchmark_duration_s
    )
    _summary(ttfts, "rollout_bench/ttft")
    _summary(tpots, "rollout_bench/tpot")
    _summary(itls, "rollout_bench/itl")
    return out



def _polar_extra_metrics(
    flat_samples: list[Any],
    rewards: list[float],
    reward_key: str,
) -> dict[str, float]:
    """Compact user-facing Polar metrics for W&B."""
    out: dict[str, float] = {}
    seen: set[str] = set()
    register_to_init_queue_ms: list[float] = []
    init_ms: list[float] = []
    run_ms: list[float] = []
    postrun_ms: list[float] = []
    session_is_placeholder: dict[str, bool] = {}
    session_report: dict[str, dict[str, Any]] = {}
    completed_session_rewards: list[float] = []
    policy_staleness: list[float] = []
    for sample in flat_samples:
        polar_meta = sample.metadata.get("polar", {})
        if "policy_staleness" in polar_meta:
            policy_staleness.append(float(polar_meta["policy_staleness"]))
        session_id = polar_meta.get("session_id")
        is_placeholder = bool(polar_meta.get("placeholder"))
        if not session_id:
            continue
        if session_id not in seen:
            seen.add(session_id)
            timing = polar_meta.get("timing") or {}
            if timing:
                register_to_init_queue_ms.append(
                    float(timing.get("register_to_init_queue_ms", 0.0))
                )
                init_ms.append(float(timing.get("init_ms", 0.0)))
                run_ms.append(float(timing.get("run_ms", 0.0)))
                postrun_ms.append(float(timing.get("postrun_ms", 0.0)))
            session_is_placeholder[session_id] = is_placeholder
            evaluation = (polar_meta.get("trajectory_metadata") or {}).get("evaluation") or {}
            report = evaluation.get("report") or {}
            if isinstance(report, dict) and report:
                session_report[session_id] = report
            if _sample_session_status(sample) == "COMPLETED" and not is_placeholder:
                completed_session_rewards.append(
                    _extract_sample_reward(sample, reward_key)
                )

    if init_ms:
        out["polar/session_ms/register_to_init_queue_mean"] = (
            sum(register_to_init_queue_ms) / len(register_to_init_queue_ms)
        )
        out["polar/session_ms/init_mean"] = sum(init_ms) / len(init_ms)
        out["polar/session_ms/run_mean"] = sum(run_ms) / len(run_ms)
        out["polar/session_ms/postrun_mean"] = sum(postrun_ms) / len(postrun_ms)
    if rewards:
        out["polar/reward_mean"] = sum(rewards) / len(rewards)
    if len(rewards) > 1:
        out["polar/reward_std"] = statistics.pstdev(rewards)
    if completed_session_rewards:
        out["polar/reward_mean_completed"] = (
            sum(completed_session_rewards) / len(completed_session_rewards)
        )
    if policy_staleness:
        out["polar/staleness/mean"] = sum(policy_staleness) / len(policy_staleness)

    total_sessions = len(seen)
    empty_sessions = sum(1 for p in session_is_placeholder.values() if p)
    if total_sessions > 0:
        out["polar/rollout_success_rate"] = (
            total_sessions - empty_sessions
        ) / total_sessions
    if session_report:
        graded_sessions = len(session_report)
        resolved = sum(1 for r in session_report.values() if r.get("resolved"))
        out["polar/eval/resolved_rate"] = resolved / graded_sessions

    out.update(_build_rollout_benchmark_metrics(flat_samples))
    return out


def _is_truncated(sample: Any) -> bool:
    status = getattr(sample, "status", None)
    return getattr(status, "value", status) == "truncated"


def _load_rollout_train_output_type() -> Any:
    try:
        from vime.rollout.base_types import RolloutFnTrainOutput
    except ImportError as exc:
        raise ImportError(
            "Slime is required to run Polar rollouts from a Slime trainer."
        ) from exc
    return RolloutFnTrainOutput


def _load_rollout_eval_output_type() -> Any:
    try:
        from vime.rollout.base_types import RolloutFnEvalOutput
    except ImportError as exc:
        raise ImportError(
            "Slime is required to run Polar evaluation rollouts from a Slime trainer."
        ) from exc
    return RolloutFnEvalOutput


def _load_sample_type() -> Any:
    try:
        from vime.utils.types import Sample
    except ImportError as exc:
        raise ImportError(
            "Slime is required to build Polar evaluation samples from eval datasets."
        ) from exc
    return Sample


atexit.register(stop_global_worker)
