"""CPU tests for vime_bridge.pg_floor(slime 2540e19 的 floor 机制移植)。

Standalone: python vime_bridge/tests/test_pg_floor_cpu.py | 或 pytest。
核心不变量:
  1. 权重轨迹内和为 1(可训练 trace 间),零 token trace 恒 0;
  2. floor=0 → 纯 token 比例;K·floor≥1 → 均分;
  3. 分母折算后,trace 对轨迹损失的贡献 = w_i·mean_i(round-trip 恒等);
  4. floor 未设置时 rollout_mask_sums 保持原生「轨迹总 token」语义(不启用即现状)。
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from vime_bridge.pg_floor import (
    floor_adjusted_denominators,
    floor_trace_weights,
    polar_pg_floor,
)


def test_weights_sum_to_one_per_trajectory():
    keys = ["a", "a", "a", "b", "b"]
    toks = [100.0, 50.0, 10.0, 30.0, 70.0]
    w = floor_trace_weights(keys, toks, 0.05)
    assert abs(sum(w[:3]) - 1.0) < 1e-9
    assert abs(sum(w[3:]) - 1.0) < 1e-9


def test_floor_zero_is_pure_token_proportional():
    keys = ["a", "a", "a"]
    toks = [100.0, 50.0, 10.0]
    w = floor_trace_weights(keys, toks, 0.0)
    assert abs(w[0] - 100 / 160) < 1e-9
    assert abs(w[1] - 50 / 160) < 1e-9
    assert abs(w[2] - 10 / 160) < 1e-9


def test_floor_guarantees_minimum_share():
    # K=3, floor=0.05:最短 trace(10/160=6.25% token)权重 0.05+(0.85)(0.0625)>保底;
    # 极限:token 占比远低于 floor 的 trace 被抬到 ~floor
    keys = ["a", "a", "a"]
    toks = [990.0, 9.0, 1.0]
    w = floor_trace_weights(keys, toks, 0.05)
    assert all(wi >= 0.05 - 1e-12 for wi in w)
    # token 最少的 trace 恰好拿 floor + 比例残差
    assert abs(w[2] - (0.05 + 0.85 * 0.001)) < 1e-9


def test_floor_too_large_degenerates_to_equal():
    # K=2, floor=0.6 → K·floor=1.2 ≥ 1 → 均分
    w = floor_trace_weights(["a", "a"], [90.0, 10.0], 0.6)
    assert w == [0.5, 0.5]


def test_zero_token_trace_gets_zero_weight():
    keys = ["a", "a", "a"]
    toks = [100.0, 0.0, 50.0]
    w = floor_trace_weights(keys, toks, 0.05)
    assert w[1] == 0.0
    assert abs(w[0] + w[2] - 1.0) < 1e-9  # 其余归一
    # 全零轨迹:全 0
    assert floor_trace_weights(["b", "b"], [0.0, 0.0], 0.05) == [0.0, 0.0]


def test_denominator_round_trip_identity():
    # D_i = T_i/w_i 之后,trace 贡献 token_sum_i/D_i 必须等于 w_i·mean_i
    keys = ["a", "a", "a"]
    toks = [100.0, 50.0, 10.0]     # T_i
    means = [0.30, 0.20, 0.10]     # 每条 trace 的 token 均值损失 mean_i
    floor = 0.05
    w = floor_trace_weights(keys, toks, floor)
    denoms = floor_adjusted_denominators(keys, toks, floor)
    contrib = [(t * m) / d for t, m, d in zip(toks, means, denoms)]
    expected = [wi * m for wi, m in zip(w, means)]
    for c, e in zip(contrib, expected):
        assert abs(c - e) < 1e-9, (c, e)
    # floor=None 等价的原生语义:分母 = ΣT(token 比例)
    assert abs(sum(contrib) - sum(expected)) < 1e-9


def test_zero_token_trace_denominator_safe():
    denoms = floor_adjusted_denominators(["a", "a"], [100.0, 0.0], 0.05)
    assert denoms[1] == 1.0  # 不产生 0/0;其 masked sum 恒 0,贡献恒 0
    assert denoms[0] > 0


def test_floor_source_precedence_and_validation():
    saved = os.environ.pop("POLAR_TRAJECTORY_PG_FLOOR", None)
    try:
        assert polar_pg_floor(None) is None                      # 未设置
        os.environ["POLAR_TRAJECTORY_PG_FLOOR"] = "0.05"
        assert polar_pg_floor(None) == 0.05                      # env 路径(slime 兼容)
        assert polar_pg_floor(SimpleNamespace(polar_trajectory_pg_floor=0.1)) == 0.1  # flag 优先
        os.environ["POLAR_TRAJECTORY_PG_FLOOR"] = "junk"
        assert polar_pg_floor(None) is None                      # 垃圾值回落 None
        os.environ["POLAR_TRAJECTORY_PG_FLOOR"] = "1.5"
        assert polar_pg_floor(None) is None                      # 越界拒绝
        os.environ["POLAR_TRAJECTORY_PG_FLOOR"] = "0.0"
        assert polar_pg_floor(None) == 0.0                       # 0.0 合法(纯 token 比例)
    finally:
        if saved is None:
            os.environ.pop("POLAR_TRAJECTORY_PG_FLOOR", None)
        else:
            os.environ["POLAR_TRAJECTORY_PG_FLOOR"] = saved


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [OK] {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [XX] {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
