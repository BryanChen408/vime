#!/usr/bin/env python3
"""Train-Inference Consistency Analysis

Loads per-token logprobs saved via ``VIME_SAVE_TIS_LOGPROBS`` and computes:
  - quantiles (1%, 5%, 25%, 50%, 75%, 95%, 99%)
  - scatter plot (train vs rollout logprobs)
  - Pearson & Spearman correlation coefficients
  - cosine similarity between train and rollout logprob vectors

Usage:
    python scripts/analyze_consistency.py <tis_dir> [output_dir]
"""

import os, sys, json, glob
from pathlib import Path

import numpy as np
import torch
from scipy import stats


def load_tis_logprobs(tis_dir: str) -> dict[int, dict]:
    """Load all TIS logprob .pt files, grouped by training step."""
    steps: dict[int, list[dict]] = {}
    pattern = os.path.join(tis_dir, "tis_logprobs_step*_rank*.pt")
    for fpath in sorted(glob.glob(pattern)):
        basename = os.path.basename(fpath)
        # format: tis_logprobs_step<N>_rank<R>.pt
        step_str = basename.split("_step")[1].split("_rank")[0]
        step = int(step_str)
        data = torch.load(fpath, map_location="cpu", weights_only=True)
        steps.setdefault(step, []).append(data)

    # Merge all ranks for each step
    merged = {}
    for step, shards in steps.items():
        old_logprobs = torch.cat([s["old_log_probs"] for s in shards])
        rollout_logprobs = torch.cat([s["rollout_log_probs"] for s in shards])
        merged[step] = {
            "old_log_probs": old_logprobs,
            "rollout_log_probs": rollout_logprobs,
        }
    return merged


def compute_metrics(train_lp: torch.Tensor, rollout_lp: torch.Tensor) -> dict:
    """Compute all consistency metrics between train and rollout logprobs."""
    t = train_lp.float().flatten().numpy()
    r = rollout_lp.float().flatten().numpy()

    # Remove inf/nan
    mask = np.isfinite(t) & np.isfinite(r)
    t, r = t[mask], r[mask]

    if len(t) < 2:
        return {"error": "not enough finite samples"}

    diff = np.abs(t - r)

    metrics = {
        "n_samples": len(t),
        "abs_diff_mean": float(np.mean(diff)),
        "abs_diff_std": float(np.std(diff)),
        "abs_diff_median": float(np.median(diff)),
        "abs_diff_quantiles": {
            "p1": float(np.quantile(diff, 0.01)),
            "p5": float(np.quantile(diff, 0.05)),
            "p25": float(np.quantile(diff, 0.25)),
            "p50": float(np.quantile(diff, 0.50)),
            "p75": float(np.quantile(diff, 0.75)),
            "p95": float(np.quantile(diff, 0.95)),
            "p99": float(np.quantile(diff, 0.99)),
        },
        "train_lp_mean": float(np.mean(t)),
        "train_lp_std": float(np.std(t)),
        "rollout_lp_mean": float(np.mean(r)),
        "rollout_lp_std": float(np.std(r)),
    }

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(t, r)
    metrics["pearson_r"] = float(pearson_r)
    metrics["pearson_pvalue"] = float(pearson_p)

    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(t, r)
    metrics["spearman_r"] = float(spearman_r)
    metrics["spearman_pvalue"] = float(spearman_p)

    # Cosine similarity
    cos_sim = float(np.dot(t, r) / (np.linalg.norm(t) * np.linalg.norm(r)))
    metrics["cosine_similarity"] = cos_sim

    return metrics


def make_scatter_data(train_lp, rollout_lp, max_points=5000):
    """Subsample for scatter plot if too many points."""
    t = train_lp.float().flatten().numpy()
    r = rollout_lp.float().flatten().numpy()
    mask = np.isfinite(t) & np.isfinite(r)
    t, r = t[mask], r[mask]
    if len(t) > max_points:
        idx = np.random.default_rng(42).choice(len(t), max_points, replace=False)
        t, r = t[idx], r[idx]
    return t.tolist(), r.tolist()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    tis_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else tis_dir

    if not os.path.isdir(tis_dir):
        print(f"ERROR: TIS directory not found: {tis_dir}")
        sys.exit(1)

    print(f"Loading TIS logprobs from: {tis_dir}")
    data = load_tis_logprobs(tis_dir)

    if not data:
        print("ERROR: No TIS logprob files found!")
        sys.exit(1)

    print(f"Found {len(data)} training steps with TIS data")

    all_metrics = {}
    all_scatter = {}

    for step in sorted(data.keys()):
        d = data[step]
        t_lp = d["old_log_probs"]
        r_lp = d["rollout_log_probs"]
        print(f"\n--- Step {step} ---")
        print(f"  Train logprobs shape: {t_lp.shape}, Rollout logprobs shape: {r_lp.shape}")

        metrics = compute_metrics(t_lp, r_lp)
        all_metrics[f"step_{step}"] = metrics

        for k, v in metrics.items():
            if not isinstance(v, dict):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}:")
                for qk, qv in v.items():
                    print(f"    {qk}: {qv}")

        # Scatter data
        scatter_x, scatter_y = make_scatter_data(t_lp, r_lp)
        all_scatter[f"step_{step}"] = {"train_logprob": scatter_x, "rollout_logprob": scatter_y}

    # Save metrics as JSON
    metrics_path = os.path.join(output_dir, "consistency_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    # Save scatter data
    scatter_path = os.path.join(output_dir, "consistency_scatter.json")
    with open(scatter_path, "w") as f:
        json.dump(all_scatter, f)
    print(f"Scatter data saved to: {scatter_path}")

    # Print summary across steps
    print("\n=== SUMMARY ===")
    abs_diffs = [m["abs_diff_mean"] for m in all_metrics.values() if "abs_diff_mean" in m]
    pearsons = [m["pearson_r"] for m in all_metrics.values() if "pearson_r" in m]
    cosines = [m["cosine_similarity"] for m in all_metrics.values() if "cosine_similarity" in m]

    if abs_diffs:
        print(f"abs_diff_mean: {np.mean(abs_diffs):.6f} ± {np.std(abs_diffs):.6f}")
    if pearsons:
        print(f"pearson_r: {np.mean(pearsons):.6f} ± {np.std(pearsons):.6f}")
    if cosines:
        print(f"cosine_similarity: {np.mean(cosines):.6f} ± {np.std(cosines):.6f}")

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
