"""POLAR_TRAJECTORY_PG_FLOOR —— 轨迹内逐 trace 的 PG 权重 floor(自 slime 2540e19 移植)。

slime 语义(`_floor_trace_weights`,设计文档 polar_doc/09-trajectory-pg-reducer-floor.md):
一条轨迹(rollout)内每条 trace 的 PG 权重

    w_i = floor + (1 - K·floor) · (T_i / ΣT)      轨迹内 Σw = 1

K = 轨迹内有可训练 token 的 trace 数,T_i = 该 trace 的可训练 token 数。
floor=0 → 纯 token 比例;floor=0.05 → 每条 trace 保底 5% 权重,短 trace 不被长 trace 淹没;
K·floor ≥ 1 → 退化为均分。零 token trace 权重恒 0(其 loss 本来为 0)。

vime 的接法与 slime 不同(刻意):slime 走自定义 reducer 消费 weights;vime 现行 reducer
(get_sum_of_sample_mean)已按 rollout_mask_sums 做「轨迹级 token 加权均值」,语义恰等于
floor=0。所以这里把权重折进分母 —— D_i = T_i / w_i,使每条 trace 对轨迹损失的贡献从
「token_sum_i / ΣT」变成「token_sum_i / D_i = w_i · mean_i」,floor 未设置时 D_i = ΣT,
与现状逐 token 等价。只动发射端一处,metrics/IS 路径的归一空间自动与梯度保持一致。
"""

from __future__ import annotations

import os
from typing import Any


def polar_pg_floor(args: Any | None = None) -> float | None:
    """返回 [0,1) 的 floor,未设置/非法返回 None。

    优先 ``args.polar_trajectory_pg_floor``(启动 flag,vime args 已正式定义),
    其次环境变量 ``POLAR_TRAJECTORY_PG_FLOOR``(slime 兼容路径)。
    """
    value = getattr(args, "polar_trajectory_pg_floor", None) if args is not None else None
    if value is None:
        value = os.environ.get("POLAR_TRAJECTORY_PG_FLOOR")
    if value in (None, ""):
        return None
    try:
        floor = float(value)
    except (TypeError, ValueError):
        return None
    return floor if 0.0 <= floor < 1.0 else None


def floor_trace_weights(trajectory_keys: list[Any], token_counts: list[float], floor: float) -> list[float]:
    """逐 trace 权重,每条轨迹内和为 1;零 token trace 权重 0。

    与 slime ``_floor_trace_weights`` 语义一致。必须在**全 batch(切分前)**计算,
    reducer 才 split-invariant。
    """
    groups: dict[Any, list[int]] = {}
    for idx, key in enumerate(trajectory_keys):
        groups.setdefault(key, []).append(idx)
    weights = [0.0] * len(trajectory_keys)
    for members in groups.values():
        trainable = [i for i in members if float(token_counts[i]) > 0.0]
        if not trainable:
            continue
        tt = [float(token_counts[i]) for i in trainable]
        total = sum(tt) or 1.0
        k = len(trainable)
        if k * floor >= 1.0:  # floor 相对该轨迹的 K 过大 -> 均分
            ws = [1.0 / k] * k
        else:
            ws = [floor + (1.0 - k * floor) * (t / total) for t in tt]
        for i, wi in zip(trainable, ws):
            weights[i] = wi
    return weights


def floor_adjusted_denominators(
    trajectory_keys: list[Any], token_counts: list[float], floor: float
) -> list[float]:
    """把 floor 权重折算成现行 reducer 的等效分母 D_i = T_i / w_i。

    w_i = 0(全掩码 trace)给 1.0:其 masked sum 恒 0,贡献不受分母影响;
    且 clamp_min(denom, 1) 兜底,不会除出 inf。
    """
    weights = floor_trace_weights(trajectory_keys, token_counts, floor)
    denoms: list[float] = []
    for t, w in zip(token_counts, weights, strict=True):
        t = float(t)
        denoms.append(t / w if (w > 0.0 and t > 0.0) else 1.0)
    return denoms
