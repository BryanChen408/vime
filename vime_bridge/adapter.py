"""Convert Polar rollout results into Slime samples.

Every trace in ``Trajectory.traces`` becomes one Slime ``Sample``.  All
samples produced from the same session share the same ``Sample.index``
(the trajectory's position within the group) so the reward post-processor
can treat them as one trajectory.  Builders own trace curation and per-token
loss masks — the adapter does not infer trainable positions from bridge
details. Traces that lack training tokens are dropped and represented as fully
masked samples so callers can keep the rest of the group trainable.
"""

from __future__ import annotations

from copy import deepcopy
import logging
from typing import Any

from vime_bridge._messages import messages_to_text
from vime_bridge.wire import SessionResult, Trace

logger = logging.getLogger(__name__)


def _trajectory_rollout_id(group_index: int, index: int) -> int:
    """Injective (Cantor pairing) map from a trajectory's (group_index, index)
    to a unique int rollout_id.

    Option A: every trace of one polar trajectory gets this same rollout_id so
    vime's native _convert_samples_to_train_data groups them and its reducer
    treats the trajectory as one rollout. (group_index, index) uniquely identify
    a trajectory within a rollout batch; Cantor pairing is collision-free for any
    non-negative ints, whether ``index`` is per-group or a global counter.
    """
    s = group_index + index
    return s * (s + 1) // 2 + index


class RolloutLogprobError(ValueError):
    """Raised when a trainable Polar trace lacks aligned rollout logprobs."""


def session_result_to_samples(
    result: "SessionResult",
    group_index: int,
    *,
    trajectory_index: int,
    reward_key: str = "score",
    max_tokens: int | None = None,
) -> list[Any]:
    """Convert one Polar session result into Slime samples — one per trace.

    Every usable trace becomes an independent Sample sharing the same
    ``(group_index, index)`` key. Slime's reward post-processor collapses
    them back into a single trajectory for advantage normalization, but
    every trace contributes its own assistant-generated tokens to the
    gradient. 

    Traces with empty tokens or exceeding ``max_tokens`` are dropped
    (logged). If *all* traces are dropped we emit a single zero-gradient
    placeholder so Slime's flattener doesn't crash on an empty list and
    the rest of the group can still train.
    """
    Sample = _load_sample_type()
    traces = result.trajectory.traces
    # Option B (dev_07): if ANY trace in this session was aborted (weight-update
    # cut-off), treat the WHOLE session as non-trainable. This aligns the
    # collection-time 0.6 accept gate (which needs a trainable COMPLETED session)
    # with reward_post_process's "any-abort trajectory failed" rule, so abort-heavy
    # groups get rejected + drain-backfilled instead of reaching training as dead
    # GRPO groups.
    # dev_09: had_abort read removed — abort now rejects the whole session at the
    # Polar builder via trajectory.status="ERROR" (propagates to slime's trainable
    # gate). This trace-level finish_reason=="abort" check stays as a harmless T5
    # fallback for any end-of-turn abort that still reaches training.
    session_has_abort = any(
        getattr(trace, "finish_reason", None) == "abort"
        for trace in traces
    )
    samples: list[Any] = []
    for trace_index, trace in enumerate(traces):
        sample = _build_sample(
            Sample=Sample,
            result=result,
            trace=trace,
            trace_index=trace_index,
            group_index=group_index,
            index=trajectory_index,
            reward_key=reward_key,
            max_tokens=max_tokens,
            session_has_abort=session_has_abort,
        )
        if sample is not None:
            samples.append(sample)

    if samples:
        return samples

    logger.warning(
        "Session %s: no usable trace (traces=%d, max_tokens=%s); emitting dummy placeholder",
        result.session_id, len(traces), max_tokens,
    )
    return [_build_dummy_sample(
        Sample=Sample,
        result=result,
        group_index=group_index,
        index=trajectory_index,
        reward_key=reward_key,
    )]


def _build_sample(
    *,
    Sample: Any,
    result: "SessionResult",
    trace: "Trace",
    trace_index: int,
    group_index: int,
    index: int,
    reward_key: str,
    max_tokens: int | None = None,
    session_has_abort: bool = False,
) -> Any | None:
    prompt_ids = list(trace.prompt_ids)
    response_ids = list(trace.response_ids)

    if not prompt_ids or not response_ids:
        logger.warning(
            "Dropping trace %d from session %s: missing tokens (prompt=%d, response=%d)",
            trace_index, result.session_id, len(prompt_ids), len(response_ids),
        )
        return None

    total_len = len(prompt_ids) + len(response_ids)
    if max_tokens is not None and total_len > max_tokens:
        logger.warning(
            "Dropping trace %d from session %s: total_len=%d > max_tokens=%d",
            trace_index, result.session_id, total_len, max_tokens,
        )
        return None

    prompt_messages = deepcopy(trace.prompt_messages)
    response_messages = deepcopy(trace.response_messages)
    response_text = messages_to_text(response_messages)

    status = _sample_status(Sample, result, trace, session_has_abort=session_has_abort)
    reward_value = _reward_value(trace)

    trainable = status not in (Sample.Status.ABORTED, Sample.Status.FAILED)
    loss_mask = _loss_mask_from_trace(
        trace,
        len(response_ids),
        require_loss_mask=trainable,
        session_id=result.session_id,
        trace_index=trace_index,
    )
    if status in (Sample.Status.ABORTED, Sample.Status.FAILED):
        loss_mask = [0] * len(response_ids)
    trainable_tokens = sum(1 for value in loss_mask if int(value) != 0)
    masked_context_tokens = len(loss_mask) - trainable_tokens

    # Diagnostic: log when a trainable trace has zero trainable tokens.
    if trainable and trainable_tokens == 0 and response_ids:
        logger.warning(
            "0-STEP-DIAG session=%s trace=%d builder=%s response_len=%d "
            "loss_mask_sum=%d loss_mask_first20=%s response_ids_first20=%s "
            "finish_reason=%s response_msgs=%d",
            result.session_id,
            trace_index,
            getattr(result.trajectory.metadata, "builder", "unknown"),
            len(response_ids),
            sum(loss_mask),
            loss_mask[:20],
            response_ids[:20],
            trace.finish_reason,
            len(response_messages),
        )
    response_log_probs = _extract_rollout_log_probs(
        trace,
        response_len=len(response_ids),
        loss_mask=loss_mask,
        require_trainable_logprobs=trainable,
        session_id=result.session_id,
        trace_index=trace_index,
    )

    prompt_value = prompt_messages if prompt_messages else ""

    polar_metadata: dict[str, Any] = {
        "node_id": result.node_id,
        "result_metadata": deepcopy(getattr(result, "metadata", {}) or {}),
        "result_error": result.error,
        "session_id": result.session_id,
        "session_status": result.status,
        "task_id": result.task_id,
        "timing": result.timing.model_dump(mode="python"),
        "trace_index": trace_index,
        "trajectory_key": [group_index, index],
        "trace_metadata": deepcopy(getattr(trace, "metadata", {}) or {}),
        "trajectory_error": result.trajectory.error,
        "trajectory_metadata": deepcopy(result.trajectory.metadata),
        "trajectory_status": result.trajectory.status,
        "token_counts": {
            "prompt_tokens": len(prompt_ids),
            "response_tokens": len(response_ids),
            "trainable_tokens": trainable_tokens,
            "masked_context_tokens": masked_context_tokens,
            "physical_total_tokens": total_len,
        },
        # Preserved for the longest-trace wandb artifact dump; training reads
        # tokens+logprobs, not these.
        "trace_debug": {
            "finish_reason": trace.finish_reason,
            "response_messages": deepcopy(response_messages),
        },
    }
    polar_metadata.update(_scheduler_metadata(result, trace))

    return Sample(
        group_index=group_index,
        index=index,
        # Option A: all traces of a polar trajectory share one rollout_id so
        # vime's native reducer treats the trajectory as one rollout. NOTE this
        # gives per-trajectory TOKEN-weighted mean, which differs from slime's
        # per-trace-equal trajectory_loss when traces differ in length (see
        # docs/design/vime_polar_integration.md §G1).
        rollout_id=_trajectory_rollout_id(group_index, index),
        prompt=prompt_value,
        tokens=prompt_ids + response_ids,
        response=response_text,
        response_length=len(response_ids),
        reward={reward_key: reward_value},
        loss_mask=loss_mask,
        rollout_log_probs=response_log_probs,
        status=status,
        metadata={"polar": polar_metadata},
    )


def _build_dummy_sample(
    *,
    Sample: Any,
    result: "SessionResult",
    group_index: int,
    index: int,
    reward_key: str,
) -> Any:
    """Fully masked placeholder for a session with no usable trace.

    This carries no policy, TIS, or KL contribution. It lets the scheduler
    accept a partially usable group while still surfacing empty sessions in
    Polar metrics.
    """
    polar_metadata: dict[str, Any] = {
        "node_id": result.node_id,
        "result_metadata": deepcopy(getattr(result, "metadata", {}) or {}),
        "result_error": result.error,
        "session_id": result.session_id,
        "session_status": result.status,
        "task_id": result.task_id,
        "timing": result.timing.model_dump(mode="python"),
        "trace_index": -1,
        "trajectory_error": result.trajectory.error,
        "trajectory_metadata": deepcopy(result.trajectory.metadata),
        "trajectory_status": result.trajectory.status,
        "placeholder": True,
    }
    polar_metadata.update(_scheduler_metadata(result, None))
    return Sample(
        group_index=group_index,
        index=index,
        rollout_id=_trajectory_rollout_id(group_index, index),  # per-trajectory (Option A)
        prompt="",
        tokens=[0, 0],
        response="",
        response_length=1,
        reward={reward_key: 0.0},
        loss_mask=[0],
        rollout_log_probs=[0.0],
        status=Sample.Status.ABORTED,
        remove_sample=True,
        metadata={"polar": polar_metadata},
    )


def _reward_value(trace: "Trace") -> float:
    """Read the reward the evaluator already placed on the trace.

    Reward assignment is the evaluator's job (including any broadcasting
    from session-level outcomes). vime_bridge just consumes what's there.
    """
    return float(trace.reward) if trace.reward is not None else 0.0


def _scheduler_metadata(result: "SessionResult", trace: "Trace | None") -> dict[str, Any]:
    keys = {
        "group_id",
        "policy_version",
        "rollout_step",
        "session_pool",
        "parent_task_id",
        "sample_pos",
        "group_size",
    }
    merged: dict[str, Any] = {}
    for source in (
        getattr(result, "metadata", None),
        getattr(result.trajectory, "metadata", None),
        getattr(trace, "metadata", None) if trace is not None else None,
    ):
        if not isinstance(source, dict):
            continue
        for key in keys:
            if key in source:
                merged[key] = source[key]
    return merged


def _sample_status(
    Sample: Any,
    result: "SessionResult",
    trace: "Trace",
    *,
    session_has_abort: bool = False,
) -> Any:
    trajectory_status = result.trajectory.status
    if trajectory_status == "TIMEOUT" or result.status == "TIMEOUT":
        return Sample.Status.ABORTED
    if trajectory_status == "ERROR" or result.status == "ERROR" or result.error or result.trajectory.error:
        return Sample.Status.FAILED
    # Option B (dev_07): any abort in the session -> the whole session is
    # non-trainable (every trace ABORTED). A session with 0 trainable samples is
    # excluded from the 0.6 accept gate (_completed_trainable_session_count), so
    # abort-heavy groups are rejected + drain-backfilled rather than trained as
    # dead GRPO groups. Subsumes the earlier per-trace abort mask.
    if session_has_abort or trace.finish_reason == "abort":
        return Sample.Status.ABORTED
    if trace.finish_reason == "length":
        return Sample.Status.TRUNCATED
    return Sample.Status.COMPLETED


def _extract_rollout_log_probs(
    trace: "Trace",
    *,
    response_len: int,
    loss_mask: list[int],
    require_trainable_logprobs: bool,
    session_id: str,
    trace_index: int,
) -> list[float]:
    logprobs = trace.response_logprobs
    if not logprobs:
        if require_trainable_logprobs and any(loss_mask):
            raise RolloutLogprobError(
                f"Session {session_id} trace {trace_index}: missing rollout_log_probs "
                "for trainable response tokens"
            )
        return [0.0] * response_len

    if len(logprobs) != response_len:
        raise RolloutLogprobError(
            f"Session {session_id} trace {trace_index}: rollout_log_probs length "
            f"{len(logprobs)} != response length {response_len}"
        )

    # response_logprobs is one float per response token (interstitials are 0.0,
    # masked out by loss_mask); the builder guarantees trainable tokens carry
    # their real sampled logprob.
    return [float(value) for value in logprobs]


def _loss_mask_from_trace(
    trace: "Trace",
    response_len: int,
    *,
    require_loss_mask: bool,
    session_id: str,
    trace_index: int,
) -> list[int]:
    """Read and validate the builder-assigned per-response-token loss mask."""
    mask = list(trace.loss_mask)
    if not mask:
        if require_loss_mask:
            raise RolloutLogprobError(
                f"Session {session_id} trace {trace_index}: missing loss_mask"
            )
        return [0] * response_len
    if len(mask) != response_len:
        raise RolloutLogprobError(
            f"Session {session_id} trace {trace_index}: loss_mask length "
            f"{len(mask)} != response length {response_len}"
        )
    return [1 if int(value) else 0 for value in mask]


def _load_sample_type() -> Any:
    try:
        from vime.utils.types import Sample
    except ImportError as exc:
        raise ImportError(
            "Slime is required to convert Polar rollouts into training samples. "
            "Ensure the Slime package is installed in the current environment."
        ) from exc
    return Sample
