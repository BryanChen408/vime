#!/usr/bin/env python3
"""Compare rollout progress curves across runs.

Reads the JSONL files written by ``_curve_log`` in ``vime_bridge/rollout.py``
(default location ``logs/rollout_curve/<tag>_r<rollout_id>.jsonl``) and prints
t@pct for each run side by side, so two configs can be compared before either
round finishes.

Usage:
    python scripts/compare_rollout_curves.py logs/rollout_curve/*.jsonl
    python scripts/compare_rollout_curves.py a_r2.jsonl b_r2.jsonl --baseline a_r2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Percent-complete checkpoints reported for every run. 10/20 are shown but are
# unreliable in async mode (the queue may already hold groups from the previous
# round), so they are excluded from the verdict.
CHECKPOINTS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
TRUSTED_FROM_PCT = 50


class Curve:
    """One round's progress curve."""

    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = path
        self.target = 0
        self.rollout_id: int | None = None
        # (t, done, cum_tokens, queue, queued_s)
        self.points: list[tuple[float, int, int, int, float]] = []
        self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = ev.get("ev")
                if kind == "round_start":
                    self.target = int(ev.get("target", 0) or 0)
                    self.rollout_id = ev.get("rollout_id")
                elif kind == "drain":
                    self.target = self.target or int(ev.get("target", 0) or 0)
                    self.points.append((
                        float(ev.get("t", 0.0)),
                        int(ev.get("done", 0)),
                        int(ev.get("cum_tokens", 0)),
                        int(ev.get("queue", 0)),
                        float(ev.get("queued_s", 0.0)),
                    ))
        self.points.sort(key=lambda p: p[1])

    @property
    def done(self) -> int:
        return self.points[-1][1] if self.points else 0

    @property
    def pct(self) -> float:
        if not self.target or not self.points:
            return 0.0
        return 100.0 * self.done / self.target

    def t_at(self, pct: int) -> float | None:
        """Time when ``pct`` percent of the target group count was collected."""
        if not self.target:
            return None
        need = max(1, int(round(self.target * pct / 100.0)))
        for t, done, _tokens, _q, _qs in self.points:
            if done >= need:
                return t
        return None

    def tokens_at(self, pct: int) -> int | None:
        if not self.target:
            return None
        need = max(1, int(round(self.target * pct / 100.0)))
        for _t, done, tokens, _q, _qs in self.points:
            if done >= need:
                return tokens
        return None

    def plateau_rate(self, lo_pct: int = 20, hi_pct: int = 60) -> float | None:
        """Tokens/s over the [lo_pct, hi_pct] window, avoiding ramp and tail."""
        t_lo, t_hi = self.t_at(lo_pct), self.t_at(hi_pct)
        k_lo, k_hi = self.tokens_at(lo_pct), self.tokens_at(hi_pct)
        if None in (t_lo, t_hi, k_lo, k_hi) or t_hi <= t_lo:
            return None
        return (k_hi - k_lo) / (t_hi - t_lo)

    def max_queued_s(self) -> float:
        """Worst head-of-line block: group finished but waited to be drained."""
        return max((p[4] for p in self.points), default=0.0)


def _fmt(value: float | None, suffix: str = "s") -> str:
    return "-" if value is None else f"{value:.0f}{suffix}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path, help="curve JSONL files")
    ap.add_argument("--baseline", default=None, help="run name to compare others against (default: first)")
    args = ap.parse_args()

    curves = [Curve(p.stem, p) for p in args.files]
    curves = [c for c in curves if c.points]
    if not curves:
        print("no drain events found in the given files")
        return 1

    width = max(len(c.name) for c in curves) + 2

    header = "run".ljust(width) + "".join(f"t@{p}%".rjust(9) for p in CHECKPOINTS)
    print(header)
    print("-" * len(header))
    for c in curves:
        row = c.name.ljust(width)
        row += "".join(_fmt(c.t_at(p)).rjust(9) for p in CHECKPOINTS)
        print(row)

    print()
    print("run".ljust(width) + "progress".rjust(12) + "plateau tok/s".rjust(16) + "max queued".rjust(13))
    print("-" * (width + 41))
    for c in curves:
        rate = c.plateau_rate()
        print(
            c.name.ljust(width)
            + f"{c.done}/{c.target}".rjust(12)
            + ("-" if rate is None else f"{rate:.0f}").rjust(16)
            + _fmt(c.max_queued_s()).rjust(13)
        )

    base = next((c for c in curves if c.name == args.baseline), curves[0])
    others = [c for c in curves if c is not base]
    if not others:
        return 0

    print()
    print(f"vs baseline {base.name} (negative = faster; only t@{TRUSTED_FROM_PCT}%+ is trustworthy)")
    print("-" * (width + 41))
    for c in others:
        deltas = []
        for p in CHECKPOINTS:
            if p < TRUSTED_FROM_PCT:
                continue
            tb, tc = base.t_at(p), c.t_at(p)
            if tb is None or tc is None or tb <= 0:
                continue
            deltas.append((p, 100.0 * (tc - tb) / tb))
        if not deltas:
            print(f"{c.name.ljust(width)}not comparable yet (need both runs past {TRUSTED_FROM_PCT}%)")
            continue
        detail = "  ".join(f"t@{p}%: {d:+.0f}%" for p, d in deltas)
        mean_delta = sum(d for _p, d in deltas) / len(deltas)
        print(f"{c.name.ljust(width)}{detail}   ->  mean {mean_delta:+.0f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
