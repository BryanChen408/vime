"""P3 attempt-credit tests (plan §6.1/§6.2/§6.3).

Covers the fixed contract:
  - delta_best: only best-improving progress earns credit (farming-immune);
  - per-trajectory score aggregation ACROSS traces (2-trace session) with
    PER-TRACE independent terms (trace0 gets its own term — the old code
    shared the first trace's term across the whole trajectory);
  - opening segment (idx=-1) resolves to R at the first scored event and
    participates in TRLOO position -1;
  - missing verdict -> no credit (never fabricated), position kept;
  - thin position (<3 members) -> trajectory-level fallback;
  - env off -> all None.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vime_bridge import attempt_credit as ac  # noqa: E402


class _Sample:
    def __init__(self, group_index, index, spans, rlen):
        self.group_index = group_index
        self.index = index
        self.metadata = {"attempt_spans": spans} if spans is not None else {}
        self.loss_mask = [1] * rlen


def _batch(monkeypatch, samples, keys, groups, std, rlens=None):
    monkeypatch.setenv("POLAR_ATTEMPT_CREDIT", "1")
    rlens = rlens or [len(s.loss_mask) for s in samples]
    return ac.build_batch(samples, keys, groups, std, rlens)


class TestDeltaBest:
    def test_only_improvements_count(self):
        assert ac.delta_best({0: 0.10, 1: 0.60, 2: 0.60, 3: 0.725, 4: 0.30}) == {
            0: 0.10,
            1: 0.50,
            2: 0.0,
            3: 0.125,
            4: 0.0,
        }

    def test_telescoping_bound(self):
        scores = {0: 0.3, 1: 0.6, 2: 0.85}
        assert sum(ac.delta_best(scores).values()) <= max(scores.values()) + 1e-9


class TestBuildBatch:
    def test_two_trace_trajectory_full_coverage(self, monkeypatch):
        """trace0 (opening segment) and the work chain each get their own term;
        TRLOO fires at the opening position -1 as designed (trace0 在场)."""
        samples = [
            _Sample(0, 0, [[0, 5, -1, None]], 5),                       # T0 trace0
            _Sample(0, 0, [[0, 60, 0, 0.3], [60, 100, 1, 0.6]], 100),  # T0 trace1
            _Sample(0, 1, [[0, 5, -1, None]], 5),                       # T1 trace0
            _Sample(0, 1, [[0, 100, 0, 0.85]], 100),                    # T1 trace1
            _Sample(0, 2, [[0, 5, -1, None]], 5),                       # T2 trace0
            _Sample(0, 2, [[0, 100, 0, 0.0]], 100),                     # T2 trace1
        ]
        keys = [(0, 0), (0, 0), (0, 1), (0, 1), (0, 2), (0, 2)]
        groups = {0: [(0, 0), (0, 1), (0, 2)]}
        terms = _batch(monkeypatch, samples, keys, groups, {0: 0.25})

        # R values: T0 deltas {0:0.3,1:0.3} -> rtg {0:0.42,1:0.3}; T1 rtg {0:0.85}; T2 rtg {0:0.0}
        # position -1 baselines: T0 (0.85+0)/2=0.425, T1 (0.42+0)/2=0.21, T2 (0.42+0.85)/2=0.635
        w, std = 0.3, 0.25
        t0_open = w * (0.42 - 0.425) / std
        t1_open = w * (0.85 - 0.21) / std
        t2_open = w * (0.0 - 0.635) / std

        # trace0 terms: independent per-trace, NOT shared from the other trace
        assert terms[0] == [t0_open] * 5
        assert terms[2] == [t1_open] * 5
        assert terms[4] == [t2_open] * 5
        # T0 trace1: position 0 (3 members) gets credit; position 1 (1 member) thin -> 0
        assert terms[1][0] == t0_open and terms[1][59] == t0_open
        assert terms[1][60] == 0.0 and terms[1][-1] == 0.0
        # T1/T2 trace1: single event position 0, same value as its opening
        assert terms[3] == [t1_open] * 100
        assert terms[5] == [t2_open] * 100
        # directions: the winner's opening is boosted, the loser's punished
        assert t1_open > 0 > t2_open

    def test_missing_verdict_no_credit_but_position_kept(self, monkeypatch):
        samples = [
            _Sample(0, 0, [[0, 100, 0, None]], 100),
            _Sample(0, 1, [[0, 100, 0, 0.5]], 100),
            _Sample(0, 2, [[0, 100, 0, 0.7]], 100),
        ]
        keys = [(0, 0), (0, 1), (0, 2)]
        groups = {0: keys}
        terms = _batch(monkeypatch, samples, keys, groups, {0: 0.2})
        # T0's verdict is missing -> r_at(0) is None -> no credit (never fabricated)
        assert terms[0] == [0.0] * 100
        # the other two still compare at position 0 (3 members incl. T0? No —
        # T0 has no R at position 0, so members = {T1, T2} = 2 < 3 -> thin -> 0)
        assert terms[1] == [0.0] * 100
        assert terms[2] == [0.0] * 100

    def test_opening_falls_back_to_first_scored_event(self, monkeypatch):
        """First event's verdict missing, second scored: opening segment takes
        R at the first SCORED event (plan's R_0 over available events)."""
        samples = [
            _Sample(0, 0, [[0, 5, -1, None]], 5),
            _Sample(0, 0, [[0, 50, 0, None], [50, 100, 1, 0.6]], 100),
            _Sample(0, 1, [[0, 100, 0, 0.4]], 100),
            _Sample(0, 2, [[0, 100, 0, 0.8]], 100),
        ]
        keys = [(0, 0), (0, 0), (0, 1), (0, 2)]
        groups = {0: [(0, 0), (0, 1), (0, 2)]}
        terms = _batch(monkeypatch, samples, keys, groups, {0: 0.2})
        # T0: scores {1:0.6} -> rtg {1:0.6}; opening resolves R_-1 = rtg[1] = 0.6
        # position -1 members: T0(0.6), T1(0.4), T2(0.8) -> baseline T0 = 0.6
        # opening add for T0 = 0.3*(0.6-0.6)/0.2 = 0
        assert terms[0] == [0.0] * 5
        # T0's missing-verdict event position 0 -> no credit; event 1 thin -> 0
        assert terms[1] == [0.0] * 100
        # T1 vs T2 at position 0: members {T1:0.4, T2:0.8} only (T0 has no R at 0)
        # -> 2 < 3 -> thin -> 0
        assert terms[2] == [0.0] * 100
        assert terms[3] == [0.0] * 100

    def test_scores_merge_across_traces_of_one_trajectory(self, monkeypatch):
        """Events living on different traces still form ONE trajectory R."""
        samples = [
            _Sample(0, 0, [[0, 50, 0, 0.2]], 50),   # T0 trace A: event 0
            _Sample(0, 0, [[0, 50, 1, 0.9]], 50),   # T0 trace B: event 1
            _Sample(0, 1, [[0, 100, 0, 0.4]], 100),
            _Sample(0, 2, [[0, 100, 0, 0.6]], 100),
        ]
        keys = [(0, 0), (0, 0), (0, 1), (0, 2)]
        groups = {0: [(0, 0), (0, 1), (0, 2)]}
        terms = _batch(monkeypatch, samples, keys, groups, {0: 0.2})
        # T0: deltas {0:0.2, 1:0.7} -> rtg {0:0.2+0.28=0.48, 1:0.7}
        # position 0: T0 0.48, T1 0.4, T2 0.6 -> base T0 = 0.5 -> add 0.3*(0.48-0.5)/0.2
        expected = 0.3 * (0.48 - 0.5) / 0.2
        assert terms[0] == [expected] * 50
        # position 1: only T0 -> thin -> 0
        assert terms[1] == [0.0] * 50

    def test_disabled_returns_none(self, monkeypatch):
        # 默认开:显式置 0 才关闭
        monkeypatch.setenv("POLAR_ATTEMPT_CREDIT", "0")
        samples = [_Sample(0, 0, [[0, 10, 0, 0.5]], 10)]
        out = ac.build_batch(samples, [(0, 0)], {0: [(0, 0)]}, {0: 0.2}, [10])
        assert out == [None]
