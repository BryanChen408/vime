import ray
import math
import logging
import json
import statistics
import re
from collections import Counter
from pathlib import Path
import torch
from vime.utils.metric_utils import has_repetition

from vime.ray.placement_group import create_placement_groups, create_rollout_manager
from vime.utils.arguments import parse_args
from vime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics

logger = logging.getLogger(__name__)


def _reward_value(sample, reward_key):
    reward = sample.get("reward")
    if isinstance(reward, dict):
        if reward_key not in reward:
            raise KeyError(f"reward key {reward_key!r} is missing from sample reward")
        value = float(reward[reward_key])
    elif reward is None:
        raise ValueError("sample reward is missing")
    else:
        value = float(reward)
    if not math.isfinite(value):
        raise ValueError(f"sample reward must be finite, got {value!r}")
    return value


def _sample_metrics(samples, reward_key):
    rewards = [_reward_value(sample, reward_key) for sample in samples]
    response_lengths = [int(sample.get("response_length", 0) or 0) for sample in samples]
    effective_lengths = [
        int(sum(sample["loss_mask"]))
        if sample.get("loss_mask") is not None
        else int(sample.get("response_length", 0) or 0)
        for sample in samples
    ]
    truncated = [sample.get("status") == "truncated" for sample in samples]
    repeated = [has_repetition(sample.get("response", "")) for sample in samples]
    prefix = [sample.get("prefix_cache_info") or {} for sample in samples]
    cached = [int(item.get("cached_tokens", 0) or 0) for item in prefix]
    prompt_tokens = [int(item.get("total_prompt_tokens", 0) or 0) for item in prefix]
    spec = [sample.get("spec_info") or {} for sample in samples]
    polar = [sample.get("metadata", {}).get("polar", {}) for sample in samples]
    session_ids = {item.get("session_id") for item in polar if item.get("session_id")}
    placeholder_sessions = set()
    resolved = []
    staleness = []
    init_ms = []
    run_ms = []
    postrun_ms = []
    queue_ms = []
    reports = []
    evaluation_metrics = []
    report_sessions = set()
    evaluation_metric_sessions = set()
    timing_sessions = set()
    staleness_sessions = set()
    for item in polar:
        session_id = item.get("session_id")
        if session_id and item.get("placeholder"):
            placeholder_sessions.add(session_id)
        staleness_value = item.get("policy_staleness")
        staleness_key = session_id or id(item)
        if staleness_value is not None and staleness_key not in staleness_sessions:
            staleness_sessions.add(staleness_key)
            staleness.append(float(staleness_value))
        timing = item.get("timing") or {}
        timing_key = session_id or id(timing)
        if timing and timing_key not in timing_sessions:
            timing_sessions.add(timing_key)
            queue_ms.append(float(timing.get("register_to_init_queue_ms", 0.0) or 0.0))
            init_ms.append(float(timing.get("init_ms", 0.0) or 0.0))
            run_ms.append(float(timing.get("run_ms", 0.0) or 0.0))
            postrun_ms.append(float(timing.get("postrun_ms", 0.0) or 0.0))
        evaluation = (item.get("trajectory_metadata") or {}).get("evaluation") or {}
        report = evaluation.get("report") or {}
        if isinstance(report, dict) and report:
            # Polar 会在每条 trace 上重复轨迹元数据；每个 session 只统计一次。
            report_key = session_id or id(report)
            if report_key not in report_sessions:
                report_sessions.add(report_key)
                reports.append(report)
                if isinstance(report.get("resolved"), bool):
                    resolved.append(report["resolved"])
        operator_metrics = evaluation.get("metrics") or {}
        metric_key = session_id or id(operator_metrics)
        if isinstance(operator_metrics, dict) and operator_metrics and metric_key not in evaluation_metric_sessions:
            evaluation_metric_sessions.add(metric_key)
            evaluation_metrics.append(operator_metrics)

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    metrics = {
        "num_samples": len(samples),
        "reward_mean": mean(rewards),
        "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "response_length_mean": mean(response_lengths),
        "response_length_median": statistics.median(response_lengths) if response_lengths else 0,
        "response_length_min": min(response_lengths, default=0),
        "response_length_max": max(response_lengths, default=0),
        "effective_response_length_mean": mean(effective_lengths),
        "effective_response_length_median": statistics.median(effective_lengths) if effective_lengths else 0,
        "effective_response_length_min": min(effective_lengths, default=0),
        "effective_response_length_max": max(effective_lengths, default=0),
        "truncated_ratio": mean(truncated),
        "repetition_frac": mean(repeated),
        "prefix_cache_hit_rate": sum(cached) / sum(prompt_tokens) if sum(prompt_tokens) else 0.0,
        "avg_cached_tokens_per_sample": mean(cached),
        "spec_accept_rate": mean([_spec_rate(item) for item in spec]),
        "spec_accept_length": mean([_spec_length(item) for item in spec]),
        "polar_session_count": len(session_ids),
        "polar_rollout_success_rate": (len(session_ids) - len(placeholder_sessions)) / len(session_ids) if session_ids else 0.0,
    }
    if staleness:
        metrics["polar_staleness_mean"] = mean(staleness)
        metrics["polar_staleness_count"] = len(staleness)
    if resolved:
        metrics["polar_eval_resolved_rate"] = mean(resolved)
        metrics["polar_eval_resolved_count"] = len(resolved)
    if timing_sessions:
        metrics.update({
            "polar_session_register_to_init_queue_ms_mean": mean(queue_ms),
            "polar_session_init_ms_mean": mean(init_ms),
            "polar_session_run_ms_mean": mean(run_ms),
            "polar_session_postrun_ms_mean": mean(postrun_ms),
            "polar_session_timing_count": len(timing_sessions),
        })
    metrics.update(_report_metrics(reports))
    metrics.update(_operator_evaluation_metrics(evaluation_metrics, len(session_ids)))
    metrics.update(_pass_at_k(samples))
    return metrics


def _pass_at_k(samples):
    """按 group_index 分组，算 pass@1 和 pass@k。

    分组依据是样本的 group_index（同一个 prompt 的多个候选共享同一 group）；
    候选顺序用样本的 index 字段；判断依据是 Polar 评测器上报的
    correctness_ok / success 字段。

    - pass@1：每个 group 的第 1 个候选正确（单次采样正确率）
    - pass@k：每个 group 的 k 个候选里至少一个正确
    """
    from collections import defaultdict

    groups = defaultdict(dict)  # group_index -> {session_key: (index, metrics)}
    for sample in samples:
        gi = sample.get("group_index")
        if gi is None:
            continue
        polar = sample.get("metadata", {}).get("polar", {})
        session_id = polar.get("session_id")
        metrics = (polar.get("trajectory_metadata") or {}).get("evaluation", {}).get("metrics") or {}
        key = session_id or id(metrics)
        idx = int(sample.get("index", 0) or 0)
        groups[gi][key] = (idx, metrics)

    if not groups:
        return {}

    result = {
        "pass_at_k_num_groups": len(groups),
        "pass_at_k_samples_per_group": max(len(v) for v in groups.values()),
    }
    for field, label in (("ast_check_ok", "ast"), ("correctness_ok", "correctness"), ("success", "success")):
        first_ok = 0
        any_ok = 0
        for v in groups.values():
            ordered = [m for _, m in sorted(v.values(), key=lambda x: x[0])]
            oks = [m.get(field) is True for m in ordered]
            if oks and oks[0]:
                first_ok += 1
            if any(oks):
                any_ok += 1
        result[f"pass_at_1_{label}"] = first_ok / len(groups)
        result[f"pass_at_k_{label}"] = any_ok / len(groups)
    return result


def _flatten_report(value, prefix=""):
    """展开嵌套字典，只保留可聚合的标量。"""
    if not isinstance(value, dict):
        return {prefix: value} if prefix else {}
    out = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            out.update(_flatten_report(item, name))
        elif isinstance(item, (bool, int, float)) and not isinstance(item, str):
            out[name] = item
    return out


def _report_metrics(reports):
    """聚合不同任务可能具有的 Polar report 标量字段。"""
    values = {}
    for report in reports:
        for key, value in _flatten_report(report).items():
            values.setdefault(key, []).append(value)

    result = {}
    if reports:
        result["polar_report_session_count"] = len(reports)
    for key, items in values.items():
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", key).strip("_").lower()
        if not safe:
            continue
        if all(isinstance(v, bool) for v in items):
            result[f"polar_report_{safe}_rate"] = sum(items) / len(items)
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in items):
            result[f"polar_report_{safe}_mean"] = sum(float(v) for v in items) / len(items)
        result[f"polar_report_{safe}_count"] = len(items)

    return result


def _operator_evaluation_metrics(rows, total_sessions):
    """按算子评测器稳定的 ``evaluation.metrics`` 结构聚合。"""
    result = {
        "evaluation_metrics_session_count": len(rows),
        "evaluation_metrics_coverage_rate": len(rows) / total_sessions if total_sessions else 0.0,
    }

    for source_key, output_key in (
        ("success", "evaluation_success_rate"),
        ("ast_check_ok", "ast_pass_rate"),
        ("correctness_ok", "correctness_pass_rate"),
    ):
        values = [row[source_key] for row in rows if isinstance(row.get(source_key), bool)]
        if values:
            result[output_key] = sum(values) / len(values)
            base_key = output_key.removesuffix("_rate")
            result[f"{base_key}_count"] = len(values)
            result[f"{base_key}_coverage_rate"] = len(values) / total_sessions if total_sessions else 0.0

    if "correctness_pass_rate" in result:
        result["accuracy_target_rate"] = result["correctness_pass_rate"]
        result["accuracy_target_count"] = result["correctness_pass_count"]

    errors = Counter(
        str(row["error_type"])
        for row in rows
        if row.get("error_type") not in (None, "")
    )
    if errors:
        result["evaluation_error_type_counts"] = dict(sorted(errors.items()))
        result["evaluation_error_type_rates"] = {
            key: value / len(rows) for key, value in sorted(errors.items())
        }

    perf_rows = [row.get("perf_data") for row in rows if isinstance(row.get("perf_data"), dict)]
    speedups = [
        float(perf["speedup_vs_torch"])
        for perf in perf_rows
        if _valid_number(perf.get("speedup_vs_torch")) and float(perf["speedup_vs_torch"]) > 0
    ]
    if speedups:
        result.update({
            "speedup_vs_torch_geomean": math.exp(sum(math.log(value) for value in speedups) / len(speedups)),
            "speedup_vs_torch_mean": statistics.fmean(speedups),
            "speedup_vs_torch_median": statistics.median(speedups),
            "speedup_vs_torch_min": min(speedups),
            "speedup_vs_torch_max": max(speedups),
            "speedup_vs_torch_std": statistics.pstdev(speedups) if len(speedups) > 1 else 0.0,
            "speedup_vs_torch_count": len(speedups),
            "speedup_vs_torch_coverage_rate": len(speedups) / total_sessions if total_sessions else 0.0,
        })

    latency_paths = {
        "framework_latency_ms": (("framework_latency_ms",), ("framework", "avg_latency_ms")),
        "impl_latency_ms": (("impl_latency_ms",), ("implementation", "avg_latency_ms")),
    }
    for key, paths in latency_paths.items():
        values = [value for perf in perf_rows if (value := _first_number(perf, paths)) is not None]
        if values:
            result[f"{key}_mean"] = statistics.fmean(values)
            result[f"{key}_median"] = statistics.median(values)
            result[f"{key}_min"] = min(values)
            result[f"{key}_max"] = max(values)
            result[f"{key}_count"] = len(values)
            result[f"{key}_coverage_rate"] = len(values) / total_sessions if total_sessions else 0.0

    case_totals = {}
    for key in ("passed_cases", "failed_cases", "total_cases"):
        values = [int(perf[key]) for perf in perf_rows if _valid_number(perf.get(key))]
        if values:
            case_totals[key] = sum(values)
            result[f"benchmark_{key}"] = case_totals[key]
    if case_totals.get("total_cases", 0) > 0 and "passed_cases" in case_totals:
        result["benchmark_case_pass_rate"] = (
            case_totals["passed_cases"] / case_totals["total_cases"]
        )

    # 保留其他标量用于诊断，但不猜测其业务含义。
    values = {}
    for row in rows:
        for key, value in _flatten_report(row).items():
            values.setdefault(key, []).append(value)
    for key, items in values.items():
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", key).strip("_").lower()
        if all(isinstance(value, bool) for value in items):
            result[f"evaluation_metric_{safe}_rate"] = sum(items) / len(items)
        elif all(_valid_number(value) for value in items):
            result[f"evaluation_metric_{safe}_mean"] = statistics.fmean(float(value) for value in items)
        else:
            continue
        result[f"evaluation_metric_{safe}_count"] = len(items)
    return result


def _valid_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _spec_rate(item):
    draft = int(item.get("spec_draft_token_num", 0) or 0)
    return float(item.get("spec_accept_token_num", 0) or 0) / draft if draft > 0 else 0.0


def _spec_length(item):
    verify_count = int(item.get("spec_verify_ct", 0) or 0)
    return float(item.get("completion_token_num", 0) or 0) / verify_count if verify_count > 0 else 0.0


def _first_number(data, paths):
    for path in paths:
        value = data
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if _valid_number(value):
            return float(value)
    return None


def _aggregate_eval_files(paths, reward_key):
    samples = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        samples.extend(payload.get("samples", []))
    if not samples:
        raise ValueError("evaluation produced no samples")
    return _sample_metrics(samples, reward_key)


def evaluate(args):
    configure_logger()
    if not args.save_debug_rollout_data:
        raise ValueError("--save-debug-rollout-data is required for evaluation metrics")
    init_tracking(args)

    rollout_manager = None
    try:
        placement_groups = create_placement_groups(args)
        rollout_manager, _ = create_rollout_manager(args, placement_groups["rollout"])
        router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
        update_tracking_open_metrics(args, router_addr)

        start_rollout_id = args.start_rollout_id or 0
        rollout_ids = []
        result_paths = []
        for rollout_id in range(start_rollout_id, args.num_rollout):
            ray.get(rollout_manager.eval.remote(rollout_id))
            rollout_ids.append(rollout_id)
            logger.info("evaluation rollout finished: rollout_id=%s", rollout_id)
            result_paths.append(Path(args.save_debug_rollout_data.format(rollout_id=f"eval_{rollout_id}")))
        overall = _aggregate_eval_files(result_paths, args.reward_key)
        summary_path = Path(args.save_debug_rollout_data.format(rollout_id="summary")).with_name(
            "evaluation_summary.json"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "overall": overall,
                    "rollout_ids": rollout_ids,
                    "result_files": [str(path) for path in result_paths],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        logger.info("evaluation summary saved to %s", summary_path)
    finally:
        try:
            if rollout_manager is not None:
                ray.get(rollout_manager.dispose.remote())
        finally:
            finish_tracking(args)
            ray.shutdown()


if __name__ == "__main__":
    evaluate(parse_args())
