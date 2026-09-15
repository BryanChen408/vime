#!/usr/bin/env python3
"""Build deterministic long-context training samples from a real rollout dump."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-tokens", type=int, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument(
        "--pattern",
        choices=("source-response", "repeat-token"),
        default="source-response",
        help="How to extend the response to the requested length.",
    )
    parser.add_argument("--token-id", type=int, default=87)
    parser.add_argument("--rollout-logprob", type=float, default=-0.1)
    return parser.parse_args()


def _tile(values: list[Any], length: int) -> list[Any]:
    if not values:
        raise ValueError("cannot tile an empty source sequence")
    repeats, remainder = divmod(length, len(values))
    return values * repeats + values[:remainder]


def _build_sample(
    template: dict[str, Any],
    *,
    sample_index: int,
    total_tokens: int,
    prompt_tokens: int,
    pattern: str,
    token_id: int,
    rollout_logprob: float,
) -> dict[str, Any]:
    sample = copy.deepcopy(template)
    prefix = list(sample["tokens"][:prompt_tokens])
    if len(prefix) != prompt_tokens:
        raise ValueError(f"source sample has only {len(prefix)} tokens, need {prompt_tokens}")
    response_tokens = total_tokens - prompt_tokens
    if pattern == "source-response":
        source_response_length = int(sample["response_length"])
        source_response_tokens = list(sample["tokens"][-source_response_length:])
        source_rollout_logprobs = list(sample["rollout_log_probs"])
        if len(source_response_tokens) != len(source_rollout_logprobs):
            raise ValueError(
                "source response token/logprob lengths differ: "
                f"{len(source_response_tokens)} != {len(source_rollout_logprobs)}"
            )
        extended_response_tokens = _tile(source_response_tokens, response_tokens)
        extended_rollout_logprobs = _tile(source_rollout_logprobs, response_tokens)
    else:
        extended_response_tokens = [token_id] * response_tokens
        extended_rollout_logprobs = [rollout_logprob] * response_tokens

    sample.update(
        group_index=0,
        index=sample_index,
        rollout_id=sample_index,
        tokens=prefix + extended_response_tokens,
        response=f"synthetic long-context validation sample {sample_index}",
        response_length=response_tokens,
        reward={"score": float(sample_index)},
        loss_mask=[1] * response_tokens,
        rollout_log_probs=extended_rollout_logprobs,
        rollout_routed_experts=None,
        remove_sample=False,
        status="completed",
        weight_versions=[],
        train_metadata=None,
    )
    metadata = dict(sample.get("metadata") or {})
    metadata["qwen36_yarn_validation"] = {
        "synthetic": True,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "pattern": pattern,
        "source_response_tokens": int(template["response_length"]),
        "unique_response_token_ids": len(set(extended_response_tokens)),
    }
    if pattern == "repeat-token":
        metadata["qwen36_yarn_validation"]["repeated_token_id"] = token_id
        metadata["qwen36_yarn_validation"]["rollout_logprob"] = rollout_logprob
    sample["metadata"] = metadata
    return sample


def main() -> None:
    args = _parse_args()
    if args.total_tokens <= args.prompt_tokens:
        raise SystemExit("--total-tokens must be greater than --prompt-tokens")
    if args.samples < 2:
        raise SystemExit("--samples must be at least 2 so GRPO has a non-zero reward contrast")

    source = torch.load(args.source, map_location="cpu", weights_only=False)
    templates = source.get("samples") if isinstance(source, dict) else None
    if not templates:
        raise ValueError(f"no samples found in {args.source}")

    samples = [
        _build_sample(
            templates[index % len(templates)],
            sample_index=index,
            total_tokens=args.total_tokens,
            prompt_tokens=args.prompt_tokens,
            pattern=args.pattern,
            token_id=args.token_id,
            rollout_logprob=args.rollout_logprob,
        )
        for index in range(args.samples)
    ]
    payload = {"rollout_id": 0, "samples": samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    result = {
        "output": str(args.output),
        "sha256": digest,
        "file_bytes": args.output.stat().st_size,
        "sample_count": len(samples),
        "total_tokens_per_sample": args.total_tokens,
        "prompt_tokens_per_sample": args.prompt_tokens,
        "response_tokens_per_sample": args.total_tokens - args.prompt_tokens,
        "pattern": args.pattern,
        "unique_response_token_ids": [
            len(set(sample["tokens"][args.prompt_tokens :])) for sample in samples
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
