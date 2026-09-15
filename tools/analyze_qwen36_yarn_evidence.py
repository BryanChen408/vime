#!/usr/bin/env python3
"""Summarize or compare response-aligned Qwen3.6 YaRN validation tensors."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def _evidence_files(path: Path) -> list[Path]:
    files = sorted(path.glob("tis_evidence_step*_mb*_rank*.pt"))
    if not files:
        raise ValueError(f"no schema-v1 TIS evidence files found under {path}")
    return files


def load_evidence(path: Path) -> list[dict[str, Any]]:
    records = [torch.load(file, map_location="cpu", weights_only=False) for file in _evidence_files(path)]
    for file, record in zip(_evidence_files(path), records, strict=True):
        if record.get("schema_version") != 1:
            raise ValueError(f"unsupported evidence schema in {file}: {record.get('schema_version')!r}")
    return records


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return math.nan
    return float(torch.quantile(values.float(), q).item())


def _stats(values: torch.Tensor) -> dict[str, float | int]:
    values = values.float().flatten()
    if values.numel() == 0:
        return {"count": 0, "mean": math.nan, "p50": math.nan, "p95": math.nan, "p99": math.nan, "max": math.nan}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "max": float(values.max().item()),
    }


def _signed_stats(values: torch.Tensor) -> dict[str, float | int]:
    values = values.float().flatten()
    if values.numel() == 0:
        return {
            "count": 0,
            "mean": math.nan,
            "min": math.nan,
            "p05": math.nan,
            "p50": math.nan,
            "p95": math.nan,
            "max": math.nan,
        }
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "min": float(values.min().item()),
        "p05": _quantile(values, 0.05),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "max": float(values.max().item()),
    }


def _delta_diagnostics(delta: torch.Tensor) -> dict[str, Any]:
    delta = delta.float().flatten()
    importance_ratio = torch.exp(delta)
    return {
        "signed_delta": _signed_stats(delta),
        "absolute_delta": _stats(delta.abs()),
        "importance_ratio": {
            **_stats(importance_ratio),
            "min": float(importance_ratio.min().item()),
        },
        "fractions": {
            "abs_delta_gt_0_1": float((delta.abs() > 0.1).float().mean().item()),
            "abs_delta_gt_1": float((delta.abs() > 1.0).float().mean().item()),
            "importance_ratio_gt_2": float((importance_ratio > 2.0).float().mean().item()),
            "importance_ratio_outside_0_5_2": float(
                ((importance_ratio < 0.5) | (importance_ratio > 2.0)).float().mean().item()
            ),
        },
    }


def _sample_key(record: dict[str, Any], index: int) -> tuple[Any, ...]:
    return (
        int(record["step"]),
        int(record["rollout_ids"][index]),
        int(record["sample_indices"][index]),
        record["sample_token_sha256"][index],
    )


def _valid_vectors(record: dict[str, Any], index: int) -> dict[str, torch.Tensor]:
    mask = record["local_loss_masks"][index].bool().flatten()
    vectors = {
        "positions": record["local_logit_positions"][index].flatten(),
        "target_token_ids": record["local_target_token_ids"][index].flatten(),
        "train_log_probs": record["train_log_probs"][index].float().flatten(),
        "current_log_probs": record["current_log_probs"][index].float().flatten(),
        "rollout_log_probs": record["rollout_log_probs"][index].float().flatten(),
    }
    expected = int(mask.numel())
    for name, value in vectors.items():
        if value.numel() != expected:
            raise ValueError(
                f"{name} length {value.numel()} does not match local mask length {expected} "
                f"for {_sample_key(record, index)}"
            )
        vectors[name] = value[mask]
    return vectors


def _position_coverage(records: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    expected: dict[tuple[Any, ...], Counter[int]] = {}
    observed: dict[tuple[Any, ...], Counter[int]] = {}
    for record in records:
        for index, (total_length, response_length) in enumerate(
            zip(record["total_lengths"], record["response_lengths"], strict=True)
        ):
            key = _sample_key(record, index)
            prompt_length = int(total_length) - int(response_length)
            full_mask = record["loss_masks"][index].bool().flatten()
            full_positions = torch.arange(prompt_length - 1, int(total_length) - 1)[full_mask]
            expected.setdefault(key, Counter(int(value) for value in full_positions.tolist()))
            local = _valid_vectors(record, index)["positions"]
            observed.setdefault(key, Counter()).update(int(value) for value in local.tolist())

    missing_samples = sorted(str(key) for key in expected.keys() - observed.keys())
    extra_samples = sorted(str(key) for key in observed.keys() - expected.keys())
    mismatched = sorted(str(key) for key in expected.keys() & observed.keys() if expected[key] != observed[key])
    ok = not missing_samples and not extra_samples and not mismatched
    return ok, {
        "sample_count": len(expected),
        "missing_samples": missing_samples,
        "extra_samples": extra_samples,
        "mismatched_samples": mismatched,
    }


def summarize(path: Path) -> dict[str, Any]:
    records = load_evidence(path)
    train_values = []
    current_values = []
    rollout_values = []
    positions = []
    total_lengths = []
    fingerprints = set()
    parallel_configs = set()

    for record in records:
        fingerprints.add(json.dumps(record["yarn_fingerprint"], sort_keys=True, default=str))
        parallel_configs.add(json.dumps(record["parallel_config"], sort_keys=True, default=str))
        total_lengths.extend(int(value) for value in record["total_lengths"])
        for index in range(len(record["total_lengths"])):
            vectors = _valid_vectors(record, index)
            train_values.append(vectors["train_log_probs"])
            current_values.append(vectors["current_log_probs"])
            rollout_values.append(vectors["rollout_log_probs"])
            positions.append(vectors["positions"])

    train = torch.cat(train_values)
    current = torch.cat(current_values)
    rollout = torch.cat(rollout_values)
    all_positions = torch.cat(positions) if positions else torch.empty(0, dtype=torch.int64)
    train_rollout_delta = train - rollout
    current_train_delta = current - train
    importance_ratio = torch.exp(train_rollout_delta)
    position_ok, position_detail = _position_coverage(records)
    all_finite = bool(
        torch.isfinite(train).all()
        and torch.isfinite(current).all()
        and torch.isfinite(rollout).all()
        and torch.isfinite(importance_ratio).all()
    )

    result = {
        "evidence_dir": str(path),
        "file_count": len(records),
        "steps": sorted({int(record["step"]) for record in records}),
        "context_parallel_sizes": sorted({int(record["cp_size"]) for record in records}),
        "fingerprints": [json.loads(value) for value in sorted(fingerprints)],
        "parallel_configs": [json.loads(value) for value in sorted(parallel_configs)],
        "max_total_length": max(total_lengths),
        "max_logit_position": int(all_positions.max().item()) if all_positions.numel() else None,
        "all_finite": all_finite,
        "packed_position_coverage_ok": position_ok,
        "packed_position_coverage": position_detail,
        "train_rollout_abs_diff": _stats(train_rollout_delta.abs()),
        "current_train_abs_diff": _stats(current_train_delta.abs()),
        "importance_ratio": {
            **_stats(importance_ratio),
            "min": float(importance_ratio.min().item()),
        },
    }
    mean_abs_diff = result["train_rollout_abs_diff"]["mean"]
    tis_mean = result["importance_ratio"]["mean"]
    result["gates"] = {
        "finite": all_finite,
        "position_coverage": position_ok,
        "train_rollout_mean_abs_diff_le_0_05": bool(mean_abs_diff <= 0.05),
        "train_rollout_mean_abs_diff_le_0_1": bool(mean_abs_diff <= 0.1),
        "importance_ratio_mean_in_0_99_1_01": bool(0.99 <= tis_mean <= 1.01),
    }
    return result


def boundary_analysis(
    path: Path,
    *,
    boundary: int | None = None,
    bin_size: int = 32_768,
) -> dict[str, Any]:
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")
    records = load_evidence(path)
    if boundary is None:
        boundaries = {
            int(record["yarn_fingerprint"]["original_max_position_embeddings"])
            for record in records
            if record["yarn_fingerprint"].get("original_max_position_embeddings") is not None
        }
        if len(boundaries) != 1:
            raise ValueError(
                "could not infer one original_max_position_embeddings value; "
                f"found {sorted(boundaries)}"
            )
        boundary = boundaries.pop()

    positions = []
    deltas = []
    for record in records:
        for index in range(len(record["total_lengths"])):
            vectors = _valid_vectors(record, index)
            positions.append(vectors["positions"].long())
            deltas.append(vectors["train_log_probs"] - vectors["rollout_log_probs"])
    all_positions = torch.cat(positions)
    all_deltas = torch.cat(deltas).float()
    max_position = int(all_positions.max().item())
    if max_position < boundary:
        raise ValueError(f"no extrapolated positions at or above boundary {boundary}")

    extrapolated_width = max_position - boundary + 1
    before_start = boundary - extrapolated_width

    def region(start: int, end: int) -> dict[str, Any]:
        mask = (all_positions >= start) & (all_positions < end)
        if not mask.any():
            raise ValueError(f"no evidence tokens in position range [{start}, {end})")
        return {
            "position_start_inclusive": start,
            "position_end_exclusive": end,
            **_delta_diagnostics(all_deltas[mask]),
        }

    before = region(before_start, boundary)
    after = region(boundary, max_position + 1)
    before_abs = before["absolute_delta"]
    after_abs = after["absolute_delta"]
    before_clip = before["fractions"]["importance_ratio_gt_2"]
    after_clip = after["fractions"]["importance_ratio_gt_2"]
    comparison = {
        "equal_width_tokens_per_sample": extrapolated_width,
        "after_to_before_abs_mean_ratio": after_abs["mean"] / before_abs["mean"],
        "after_to_before_abs_p95_ratio": after_abs["p95"] / before_abs["p95"],
        "after_minus_before_importance_ratio_gt_2_fraction": after_clip - before_clip,
        "diagnostic_thresholds": {
            "max_abs_mean_ratio": 1.10,
            "max_abs_p95_ratio": 1.10,
            "max_importance_ratio_gt_2_fraction_increase": 0.01,
        },
    }
    comparison["no_boundary_cliff"] = bool(
        comparison["after_to_before_abs_mean_ratio"] <= 1.10
        and comparison["after_to_before_abs_p95_ratio"] <= 1.10
        and comparison["after_minus_before_importance_ratio_gt_2_fraction"] <= 0.01
    )

    bins = []
    for start in range(0, max_position + 1, bin_size):
        end = min(start + bin_size, max_position + 1)
        mask = (all_positions >= start) & (all_positions < end)
        if mask.any():
            bins.append(
                {
                    "position_start_inclusive": start,
                    "position_end_exclusive": end,
                    **_delta_diagnostics(all_deltas[mask]),
                }
            )

    return {
        "evidence_dir": str(path),
        "boundary_position": boundary,
        "max_logit_position": max_position,
        "token_count": int(all_positions.numel()),
        "regions": {
            "all": region(int(all_positions.min().item()), max_position + 1),
            "within_original": region(int(all_positions.min().item()), boundary),
            "equal_width_before_boundary": before,
            "extrapolated": after,
        },
        "boundary_comparison": comparison,
        "position_bins": bins,
    }


def _token_map(records: list[dict[str, Any]], field: str) -> dict[tuple[Any, ...], float]:
    result: dict[tuple[Any, ...], float] = {}
    for record in records:
        for index in range(len(record["total_lengths"])):
            sample_key = _sample_key(record, index)
            vectors = _valid_vectors(record, index)
            for position, token_id, value in zip(
                vectors["positions"].tolist(),
                vectors["target_token_ids"].tolist(),
                vectors[field].tolist(),
                strict=True,
            ):
                key = (*sample_key, int(position), int(token_id))
                if key in result:
                    raise ValueError(f"duplicate valid token in evidence: {key}")
                result[key] = float(value)
    return result


def compare(reference: Path, candidate: Path) -> dict[str, Any]:
    reference_records = load_evidence(reference)
    candidate_records = load_evidence(candidate)
    reference_values = _token_map(reference_records, "train_log_probs")
    candidate_values = _token_map(candidate_records, "train_log_probs")
    reference_keys = set(reference_values)
    candidate_keys = set(candidate_values)
    shared_keys = sorted(reference_keys & candidate_keys, key=str)
    deltas = torch.tensor(
        [abs(reference_values[key] - candidate_values[key]) for key in shared_keys],
        dtype=torch.float32,
    )
    return {
        "reference_dir": str(reference),
        "candidate_dir": str(candidate),
        "reference_token_count": len(reference_keys),
        "candidate_token_count": len(candidate_keys),
        "shared_token_count": len(shared_keys),
        "token_sets_equal": reference_keys == candidate_keys,
        "missing_from_candidate": len(reference_keys - candidate_keys),
        "extra_in_candidate": len(candidate_keys - reference_keys),
        "train_logprob_abs_diff": _stats(deltas),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("summary", help="Summarize one evidence directory")
    summary_parser.add_argument("evidence_dir", type=Path)
    summary_parser.add_argument("--output", type=Path)

    compare_parser = subparsers.add_parser("compare", help="Compare training logprobs for identical tokens")
    compare_parser.add_argument("reference_dir", type=Path)
    compare_parser.add_argument("candidate_dir", type=Path)
    compare_parser.add_argument("--output", type=Path)

    boundary_parser = subparsers.add_parser(
        "boundary", help="Compare train/rollout deltas before and after the YaRN boundary"
    )
    boundary_parser.add_argument("evidence_dir", type=Path)
    boundary_parser.add_argument("--boundary", type=int)
    boundary_parser.add_argument("--bin-size", type=int, default=32_768)
    boundary_parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "summary":
        result = summarize(args.evidence_dir)
    elif args.command == "compare":
        result = compare(args.reference_dir, args.candidate_dir)
    else:
        result = boundary_analysis(
            args.evidence_dir,
            boundary=args.boundary,
            bin_size=args.bin_size,
        )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
