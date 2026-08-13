"""P3 event-level (per-attempt) credit assignment — plan §6 (Kevin-32B
reward-to-go, Dr.Kernel TRLOO leave-one-out baseline).

Each pipeline attempt inside a trajectory becomes a discounted reward-to-go
credit, applied as a PER-TOKEN ADDITIVE term on top of the trajectory-level
scalar advantage. The trace is NEVER split into extra samples (no prefix
re-encoding / compute tax); only the advantage varies by attempt segment.
FLOOR (trace-level weighting) is orthogonal and untouched.

Env-gated by POLAR_ATTEMPT_CREDIT (unset -> None everywhere -> zero change).

Contract (produced by the polar builder, stage 2):
  attempt_spans on each trainable trace, in RESPONSE-token coordinates
  [0, response_len): [[start, end, attempt_idx, score|None], ...]
  Spans PARTITION the trace's response stream (segment model, plan §6.2):
  an event's segment extends to the next event's start, and the opening
  segment (trace0 = Skill-dispatch chain, "事件 1 之前的段") carries
  attempt_idx = -1. `attempt_idx` is the SERVER-SIDE executed-attempt ordinal
  (assigned at detection time in session time order; never derived from any
  agent-visible counter). score is the verdict's reward-ladder value, or None
  when the call was detected but its verdict is missing — the position is
  kept for group alignment but earns no credit (never fabricated).

r_t = Delta-best (plan §6.1): only best-improving progress earns process
  credit — max(0, best_t - best_{t-1}) over the ladder scores. Repeated
  identical verdicts yield 0 (compile-only farming is immune by design) and
  the process sum telescopes toward the final best, so it can never dominate
  or double-count the terminal reward.

R_e (reward-to-go, sum aggregation, gamma=0.4):  R_e = sum_{i>=e} gamma^(i-e) * r_i.
  The opening segment (idx -1) resolves to R at the first scored event —
  plan's R_0 = r_1 + gamma*r_2 + ... — so trace0's dispatch decision carries
  the signal of everything it led to.

TRLOO baseline: per attempt-position e (including -1), leave-one-out mean of
R_e over the group's other trajectories that have an R value at position e.
Thin positions (<3 members) fall back to trajectory-level (additive term 0
-> just A_traj).

Scale: the additive term is normalized by the same group reward std (clamped
at STD_FLOOR) that A_traj uses, so the two are comparable; then scaled by
w_process (<=0.3) so the terminal signal stays dominant.
"""

from __future__ import annotations

import os
from typing import Any

_STD_FLOOR = 0.05
_MIN_LOO_MEMBERS = 3
_SEGMENT_ZERO = -1  # opening segment (before the first pipeline event)


def enabled() -> bool:
    # 默认开(polar 生产者同步默认开);POLAR_ATTEMPT_CREDIT=0/false/no/off 关闭。
    return os.environ.get("POLAR_ATTEMPT_CREDIT", "1").lower() not in ("0", "false", "no", "off")


def _gamma() -> float:
    try:
        return float(os.environ.get("POLAR_ATTEMPT_GAMMA", "0.4"))
    except ValueError:
        return 0.4


def _w_process() -> float:
    try:
        w = float(os.environ.get("POLAR_ATTEMPT_W", "0.3"))
    except ValueError:
        w = 0.3
    return max(0.0, min(w, 0.3))  # cap: terminal must dominate (VeRPO anchor)


def attempt_spans(sample: Any) -> list[tuple[int, int, int, float | None]]:
    """Extract [(start, end, attempt_idx, score|None), ...] from the sample
    metadata. Robust to both the flat and the polar.trace_metadata nesting.
    idx may be -1 (opening segment); score may be None (verdict missing)."""
    md = getattr(sample, "metadata", None)
    raw = None
    if isinstance(md, dict):
        raw = md.get("attempt_spans")
        if raw is None:
            polar = md.get("polar") or {}
            # Adapter maps the builder's per-trace metadata to polar.trace_metadata
            # (slime_bridge/adapter.py:168); trajectory_metadata is a session-level
            # fallback. attempt_spans is per-trace, so trace_metadata is authoritative.
            for sub in ("trace_metadata", "trajectory_metadata"):
                cand = ((polar.get(sub) or {}) if isinstance(polar, dict) else {}).get("attempt_spans")
                if cand:
                    raw = cand
                    break
    if not raw:
        return []
    out: list[tuple[int, int, int, float | None]] = []
    for item in raw:
        try:
            start, end, idx = int(item[0]), int(item[1]), int(item[2])
            score = item[3]
            score = float(score) if score is not None else None
        except (TypeError, ValueError, IndexError):
            continue
        if end > start:
            out.append((start, end, idx, score))
    return out


def delta_best(scores_by_idx: dict[int, float]) -> dict[int, float]:
    """r_t = max(0, best_t - best_{t-1}) over ordered attempt ordinals (plan
    §6.1). Only best-improving progress earns process credit; repeated or
    regressing verdicts yield 0. Telescoping: sum(r_t) <= final best."""
    best = 0.0
    out: dict[int, float] = {}
    for idx in sorted(scores_by_idx):
        s = float(scores_by_idx[idx])
        out[idx] = max(0.0, s - best)
        best = max(best, s)
    return out


def reward_to_go(scores_by_idx: dict[int, float], gamma: float) -> dict[int, float]:
    """R_e = sum_{i>=e} gamma^(i-e) * s_i over the ordered attempt indices present."""
    idxs = sorted(scores_by_idx)
    rtg: dict[int, float] = {}
    for e in idxs:
        acc = 0.0
        for i in idxs:
            if i >= e:
                acc += (gamma ** (i - e)) * scores_by_idx[i]
        rtg[e] = acc
    return rtg


def build_batch(
    samples: list[Any],
    key_by_sample: list[tuple[int, int]],
    group_keys: dict[int, list[tuple[int, int]]],
    group_std: dict[int, float],
    response_len: list[int],
) -> list[list[float] | None]:
    """Return, per sample, a per-response-token additive advantage list (len =
    response_len[i]) or None (no spans / disabled -> trajectory-level fallback).

    Scores/R are aggregated per TRAJECTORY across all its traces (a 2-trace
    session's events all live in the work chain while the opening segment
    lives on trace0); the per-token terms are computed PER TRACE from that
    trace's own spans and coordinates — traces never share a term.
    """
    n = len(samples)
    if not enabled():
        return [None] * n

    gamma = _gamma()
    w = _w_process()

    spans_by_sample: list[list[tuple[int, int, int, float | None]]] = [
        attempt_spans(s) for s in samples
    ]
    traj_group: dict[tuple[int, int], int] = {}
    traj_scores: dict[tuple[int, int], dict[int, float]] = {}
    traj_positions: dict[tuple[int, int], set[int]] = {}
    for i, sample in enumerate(samples):
        key = key_by_sample[i]
        if key not in traj_group:
            traj_group[key] = int(sample.group_index) if sample.group_index is not None else -1
        scores = traj_scores.setdefault(key, {})
        positions = traj_positions.setdefault(key, set())
        for (_, _, idx, score) in spans_by_sample[i]:
            positions.add(idx)
            if score is not None and idx >= 0:
                scores[idx] = score

    # R per trajectory from Delta-best progress, gamma-discounted.
    traj_rtg: dict[tuple[int, int], dict[int, float]] = {}
    for key, scores in traj_scores.items():
        traj_rtg[key] = reward_to_go(delta_best(scores), gamma) if scores else {}

    def r_at(key: tuple[int, int], pos: int) -> float | None:
        """R value for a segment position; None when unresolved (no scored
        event at/after it) — such tokens keep A_traj only, never fabricated."""
        rtg = traj_rtg.get(key) or {}
        if pos == _SEGMENT_ZERO:
            # Opening segment: reward-to-go at the first scored event (R_0).
            return rtg[min(rtg)] if rtg else None
        return rtg.get(pos)

    # TRLOO baseline: per group, per position (including the opening segment),
    # leave-one-out mean of R over trajectories that resolve an R there.
    pos_values: dict[int, dict[int, dict[tuple[int, int], float]]] = {}
    for key, positions in traj_positions.items():
        g = traj_group[key]
        for pos in positions:
            r = r_at(key, pos)
            if r is not None:
                pos_values.setdefault(g, {}).setdefault(pos, {})[key] = r

    def baseline(g: int, pos: int, self_key: tuple[int, int]) -> float | None:
        members = pos_values.get(g, {}).get(pos, {})
        if len(members) < _MIN_LOO_MEMBERS:
            return None  # thin position -> trajectory-level fallback (no credit)
        others = [v for k, v in members.items() if k != self_key]
        if not others:
            return None
        return sum(others) / len(others)

    out: list[list[float] | None] = [None] * n
    for i, sample in enumerate(samples):
        spans = spans_by_sample[i]
        rlen = int(response_len[i]) if i < len(response_len) else 0
        if not spans or rlen <= 0:
            continue
        key = key_by_sample[i]
        g = traj_group[key]
        std = max(float(group_std.get(g, 1.0)), _STD_FLOOR)
        term = [0.0] * rlen
        for (start, end, pos, _score) in spans:
            r = r_at(key, pos)
            if r is None:
                continue  # verdict missing for this position -> no credit
            base = baseline(g, pos, key)
            if base is None:
                continue  # thin position -> leave those tokens at A_traj only
            add = w * (r - base) / std
            lo, hi = max(0, start), min(rlen, end)
            for t in range(lo, hi):
                term[t] = add
        out[i] = term
    return out
