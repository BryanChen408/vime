"""KernelGym reward computation.

Pure reward strategies migrated from rllm-lilac's
``rllm/environments/kernelgym/kernelgym_env.py`` (``calculate_reward_*``).

Each function takes the raw result dict returned by the kernelGym
``/results/{task_id}`` endpoint plus a :class:`RewardConfig`, and returns a
normalized reward-summary dict. They are intentionally free of any I/O or
framework state so they can be unit-tested on CPU without a running kernelGym
service.

Result dict fields produced by kernelGym (see kernelGym-NPU
``kernelgym/server/api/models.py``)::

    status        str   "completed" | "failed" | "timeout" | "cancelled"
    compiled      bool
    correctness   bool
    speedup       float  reference_runtime / kernel_runtime
    decoy_kernel  bool   reward-hacking marker (kernel only calls reference)
    metadata      dict   coverage counters, diagnostics
    error_message str

Reward-summary dict returned by every strategy::

    reward / score   float   the scalar reward used for training
    speedup          float
    success          bool    compiled and correctness
    correctness      bool
    compiled         bool
    error            str|None
    + coverage counters (num_custom_kernel, num_total_kernels, ...)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Reward-shaping knobs. Mirrors the rllm ``reward_model`` config block."""

    # Which strategy to use: one of REWARD_FUNCS keys.
    reward_func_name: str = "calculate_reward_weighted"
    # Weighted / speedup strategies.
    init_correct_weight: float = 0.5
    init_performance_weight: float = 0.5
    speedup_eps: float = 0.01
    speedup_reward_upper_bound: float = 3.0
    speedup_reward_lower_bound: float = 0.0
    # Penalty applied when the task did not complete / decoy detected
    # (weighted & speedup strategies).
    penalty_score: float = 0.0
    # Stepwise penalties for the "like_kernel" strategy.
    compilation_fail_penalty: float = -0.5
    correctness_fail_penalty: float = -0.3
    perf_degrade_penalty: float = -0.1
    # Optional coverage bonus (only added when correct).
    coverage_enable: bool = False
    coverage_reward_type: str = "time_coverage"  # or "number_coverage"
    coverage_weight: float = 0.25

    @staticmethod
    def from_mapping(cfg: Any | None) -> "RewardConfig":
        """Build from a plain dict (e.g. ``args.kernelgym_reward``).

        Unknown keys are ignored; missing keys fall back to defaults. Also
        accepts the nested rllm shape ``reward_policy.penalties.*`` for the
        per-failure penalties so existing configs port over unchanged.
        """
        if cfg is None:
            return RewardConfig()
        if isinstance(cfg, RewardConfig):
            return cfg
        # OmegaConf DictConfig / dataclass / Mapping all expose dict(...).
        try:
            data = dict(cfg)
        except (TypeError, ValueError):
            return RewardConfig()

        out = RewardConfig()
        for key in (
            "reward_func_name",
            "init_correct_weight",
            "init_performance_weight",
            "speedup_eps",
            "speedup_reward_upper_bound",
            "speedup_reward_lower_bound",
            "penalty_score",
            "compilation_fail_penalty",
            "correctness_fail_penalty",
            "perf_degrade_penalty",
            "coverage_enable",
            "coverage_reward_type",
            "coverage_weight",
        ):
            if key in data and data[key] is not None:
                setattr(out, key, data[key])

        # Nested rllm-style penalties: reward_policy.penalties.{compilation_fail,...}
        policy = data.get("reward_policy") or {}
        penalties = (dict(policy).get("penalties") if policy else None) or {}
        penalties = dict(penalties) if penalties else {}
        if "penalty_score" in penalties and penalties["penalty_score"] is not None:
            out.penalty_score = float(penalties["penalty_score"])
        if "compilation_fail" in penalties and penalties["compilation_fail"] is not None:
            out.compilation_fail_penalty = float(penalties["compilation_fail"])
        if "correctness_fail" in penalties and penalties["correctness_fail"] is not None:
            out.correctness_fail_penalty = float(penalties["correctness_fail"])
        if "perf_degrade" in penalties and penalties["perf_degrade"] is not None:
            out.perf_degrade_penalty = float(penalties["perf_degrade"])

        # Nested rllm-style coverage_reward.{enable,reward_type,weight}
        coverage = data.get("coverage_reward") or {}
        coverage = dict(coverage) if coverage else {}
        if "enable" in coverage and coverage["enable"] is not None:
            out.coverage_enable = bool(coverage["enable"])
        if "reward_type" in coverage and coverage["reward_type"] is not None:
            out.coverage_reward_type = str(coverage["reward_type"])
        if "weight" in coverage and coverage["weight"] is not None:
            out.coverage_weight = float(coverage["weight"])
        return out


def _is_completed(result: dict) -> bool:
    return result.get("status") == "completed"


def _error_message(result: dict) -> str:
    msg = result.get("error_message") or result.get("error") or "Task failed"
    return str(msg)


def _coerce_speedup(result: dict) -> float:
    speedup = result.get("speedup", 0.0)
    if speedup is None:
        return 0.0
    try:
        return float(speedup)
    except (TypeError, ValueError):
        return 0.0


def _coverage_fields(result: dict) -> dict:
    """Pull coverage counters; kernelGym may nest them under ``metadata`` and
    use either singular or plural names. Normalize to a flat dict."""
    metadata = result.get("metadata") or {}

    def _get(*keys: str) -> int:
        for k in keys:
            if k in metadata and metadata[k] is not None:
                return metadata[k]
            if k in result and result[k] is not None:
                return result[k]
        return 0

    return {
        "num_custom_kernel": _get("num_custom_kernels", "num_custom_kernel"),
        "num_total_kernels": _get("num_total_kernels", "num_total_kernel"),
        "custom_kernel_cuda_time_in_profiling_us": _get("custom_kernel_cuda_time_in_profiling_us"),
        "total_kernel_run_time_in_profiling_us": _get("total_kernel_run_time_in_profiling_us"),
    }


def compute_coverage(result: dict, cfg: RewardConfig) -> float:
    fields = _coverage_fields(result)
    num_total = fields["num_total_kernels"] or 0
    total_time = fields["total_kernel_run_time_in_profiling_us"] or 0
    num_coverage = (fields["num_custom_kernel"] / num_total) if num_total > 0 else 0.0
    time_coverage = (
        fields["custom_kernel_cuda_time_in_profiling_us"] / total_time if total_time > 0 else 0.0
    )
    if cfg.coverage_reward_type == "time_coverage":
        return float(time_coverage)
    if cfg.coverage_reward_type == "number_coverage":
        return float(num_coverage)
    raise ValueError(f"Invalid coverage reward type: {cfg.coverage_reward_type}")


def _failure_summary(result: dict, reward_value: float, *, decoy: bool = False) -> dict:
    summary = {
        "reward": reward_value,
        "score": reward_value,
        "speedup": 0.0,
        "success": False,
        "correctness": False,
        "compiled": False,
        "error": ("Reward hacking: Decoy kernel detected" if decoy else _error_message(result)),
    }
    if decoy:
        summary["decoy_kernel"] = True
    # Carry through any extra fields kernelGym returned (non-destructively).
    for key, value in (result or {}).items():
        summary.setdefault(key, value)
    return summary


def _success_summary(result: dict, cfg: RewardConfig, base_reward: float) -> dict:
    correctness = bool(result.get("correctness", False))
    compiled = bool(result.get("compiled", False))
    speedup = _coerce_speedup(result)
    fields = _coverage_fields(result)
    final_reward = base_reward
    if correctness and cfg.coverage_enable:
        final_reward += cfg.coverage_weight * compute_coverage(result, cfg)
    return {
        "reward": final_reward,
        "score": final_reward,
        "speedup": speedup,
        "success": compiled and correctness,
        "correctness": correctness,
        "compiled": compiled,
        "error": result.get("error_message") or result.get("error"),
        "profiling": result.get("profiling"),
        **fields,
    }


def calculate_reward_like_kernel(result: dict, cfg: RewardConfig) -> dict:
    """Stepwise ladder reward (rllm default).

    not compiled -> compilation_fail_penalty; compiled but wrong ->
    correctness_fail_penalty; correct -> 0.2..1.0 by speedup tier.
    """
    if not _is_completed(result):
        return _failure_summary(result, -1.0)
    if result.get("decoy_kernel", False):
        return _failure_summary(result, -1.0, decoy=True)

    correctness = bool(result.get("correctness", False))
    compiled = bool(result.get("compiled", False))
    speedup = _coerce_speedup(result)

    if not compiled:
        reward = cfg.compilation_fail_penalty
    elif not correctness:
        reward = cfg.correctness_fail_penalty
    else:
        if speedup >= 3.0:
            reward = 1.0
        elif speedup >= 2.0:
            reward = 0.8
        elif speedup >= 1.5:
            reward = 0.6
        elif speedup >= 1.2:
            reward = 0.4
        elif speedup >= 1.0:
            reward = 0.2
        else:
            reward = cfg.perf_degrade_penalty
    return {
        "reward": reward,
        "score": reward,
        "speedup": speedup,
        "success": compiled and correctness,
        "correctness": correctness,
        "compiled": compiled,
        "error": result.get("error_message") or result.get("error"),
    }


def calculate_reward_weighted(result: dict, cfg: RewardConfig) -> dict:
    """Weighted reward: correctness weight + a binary "is speedup positive"."""
    if not _is_completed(result):
        return _failure_summary(result, cfg.penalty_score)
    if result.get("decoy_kernel", False):
        return _failure_summary(result, cfg.penalty_score, decoy=True)

    correctness = bool(result.get("correctness", False))
    speedup = _coerce_speedup(result)
    is_speedup_positive = speedup >= (1 + cfg.speedup_eps)
    base = cfg.init_correct_weight * correctness + cfg.init_performance_weight * is_speedup_positive
    return _success_summary(result, cfg, base)


def calculate_reward_speedup(result: dict, cfg: RewardConfig) -> dict:
    """Speedup reward: correctness weight + clamped speedup magnitude."""
    if not _is_completed(result):
        return _failure_summary(result, cfg.penalty_score)
    if result.get("decoy_kernel", False):
        return _failure_summary(result, cfg.penalty_score, decoy=True)

    correctness = bool(result.get("correctness", False))
    speedup = _coerce_speedup(result)
    reward_speedup = min(speedup, cfg.speedup_reward_upper_bound)
    if reward_speedup < cfg.speedup_reward_lower_bound:
        reward_speedup = 0.0
    base = cfg.init_correct_weight * correctness + cfg.init_performance_weight * reward_speedup
    return _success_summary(result, cfg, base)


REWARD_FUNCS = {
    "calculate_reward_like_kernel": calculate_reward_like_kernel,
    "calculate_reward_weighted": calculate_reward_weighted,
    "calculate_reward_speedup": calculate_reward_speedup,
}


def compute_reward_summary(result: dict, cfg: RewardConfig) -> dict:
    """Dispatch to the configured reward strategy; default to weighted."""
    func = REWARD_FUNCS.get(cfg.reward_func_name)
    if func is None:
        logger.warning(
            "Unknown reward_func_name=%s; falling back to calculate_reward_weighted",
            cfg.reward_func_name,
        )
        func = calculate_reward_weighted
    summary = func(result or {}, cfg)
    # Merge raw result first so strategy-computed fields win on conflict.
    return {**(result or {}), **summary}
