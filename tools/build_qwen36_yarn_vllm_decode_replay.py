#!/usr/bin/env python3
"""Build a long-context replay from tokens genuinely decoded by vLLM."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _request_json(url: str, body: dict[str, Any], timeout: float) -> tuple[int, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _DIRECT_OPENER.open(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        payload = error.read().decode("utf-8", errors="replace")
        try:
            parsed: Any = json.loads(payload)
        except json.JSONDecodeError:
            parsed = payload
        return error.code, parsed


def _wait_until_ready(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(f"{base_url}/v1/models", method="GET")
            with _DIRECT_OPENER.open(request, timeout=10) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(5)
    raise TimeoutError(f"vLLM endpoint did not become ready in {timeout}s: {last_error}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stats(values: list[float]) -> dict[str, float | int]:
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "count": len(values),
        "mean": float(tensor.mean().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _build_sample(
    template: dict[str, Any],
    *,
    sample_index: int,
    prompt_tokens: list[int],
    response_tokens: list[int],
    rollout_logprobs: list[float],
    model: str,
) -> dict[str, Any]:
    sample = copy.deepcopy(template)
    response_length = len(response_tokens)
    sample.update(
        group_index=0,
        index=sample_index,
        rollout_id=sample_index,
        tokens=prompt_tokens + response_tokens,
        response=f"vLLM decode long-context validation sample {sample_index}",
        response_length=response_length,
        reward={"score": float(sample_index)},
        loss_mask=[1] * response_length,
        rollout_log_probs=rollout_logprobs,
        rollout_routed_experts=None,
        remove_sample=False,
        status="completed",
        weight_versions=[],
        train_metadata=None,
    )
    metadata = dict(sample.get("metadata") or {})
    metadata["qwen36_yarn_validation"] = {
        "synthetic_prompt": True,
        "vllm_decode_response": True,
        "vllm_model": model,
        "total_tokens": len(sample["tokens"]),
        "prompt_tokens": len(prompt_tokens),
        "response_tokens": response_length,
        "unique_response_token_ids": len(set(response_tokens)),
    }
    sample["metadata"] = metadata
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:15000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=262_144)
    parser.add_argument("--total-tokens", type=int, default=300_000)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--wait-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=14400)
    args = parser.parse_args()

    if args.total_tokens <= args.prompt_tokens:
        raise SystemExit("--total-tokens must be greater than --prompt-tokens")
    if args.samples < 2:
        raise SystemExit("--samples must be at least 2 so GRPO has a reward contrast")

    source = torch.load(args.input, map_location="cpu", weights_only=False)
    templates = source.get("samples") if isinstance(source, dict) else None
    if not templates:
        raise ValueError(f"no samples found in {args.input}")

    base_url = args.base_url.rstrip("/")
    _wait_until_ready(base_url, args.wait_timeout)
    response_length = args.total_tokens - args.prompt_tokens
    started_at = datetime.now(timezone.utc).isoformat()
    decoded_samples = []
    sample_results = []

    for sample_index in range(args.samples):
        template = templates[sample_index % len(templates)]
        prompt_tokens = [int(token) for token in template["tokens"][: args.prompt_tokens]]
        if len(prompt_tokens) != args.prompt_tokens:
            raise ValueError(
                f"sample {sample_index} has {len(prompt_tokens)} prompt tokens, "
                f"expected {args.prompt_tokens}"
            )
        request_body = {
            "model": args.model,
            "prompt": prompt_tokens,
            "max_tokens": response_length,
            "min_tokens": response_length,
            "ignore_eos": True,
            "temperature": args.temperature,
            "seed": args.seed + sample_index,
            "logprobs": 0,
            "return_token_ids": True,
            "add_special_tokens": False,
        }
        started = time.monotonic()
        status, response = _request_json(
            f"{base_url}/v1/completions",
            request_body,
            timeout=args.request_timeout,
        )
        duration = time.monotonic() - started
        if status != 200 or not isinstance(response, dict):
            raise RuntimeError(
                f"sample {sample_index} decode failed: HTTP {status}: {response!r}"
            )
        choices = response.get("choices") or []
        if len(choices) != 1:
            raise ValueError(f"sample {sample_index} returned {len(choices)} choices")
        choice = choices[0]
        response_tokens = [int(token) for token in choice.get("token_ids") or []]
        logprobs = (choice.get("logprobs") or {}).get("token_logprobs") or []
        rollout_logprobs = [float(value) for value in logprobs]
        if len(response_tokens) != response_length:
            raise ValueError(
                f"sample {sample_index} generated {len(response_tokens)} tokens, "
                f"expected {response_length}; finish_reason={choice.get('finish_reason')!r}"
            )
        if len(rollout_logprobs) != response_length:
            raise ValueError(
                f"sample {sample_index} returned {len(rollout_logprobs)} logprobs, "
                f"expected {response_length}"
            )
        if not bool(torch.isfinite(torch.tensor(rollout_logprobs)).all().item()):
            raise ValueError(f"sample {sample_index} contains non-finite logprobs")

        decoded_samples.append(
            _build_sample(
                template,
                sample_index=sample_index,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                rollout_logprobs=rollout_logprobs,
                model=args.model,
            )
        )
        sample_result = {
            "sample_index": sample_index,
            "duration_seconds": duration,
            "prompt_tokens": len(prompt_tokens),
            "response_tokens": len(response_tokens),
            "total_tokens": len(prompt_tokens) + len(response_tokens),
            "unique_response_token_ids": len(set(response_tokens)),
            "finish_reason": choice.get("finish_reason"),
            "usage": response.get("usage"),
            "response_logprobs": _stats(rollout_logprobs),
        }
        sample_results.append(sample_result)
        print(json.dumps(sample_result, sort_keys=True), flush=True)

    payload = dict(source)
    payload["samples"] = decoded_samples
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)

    result = {
        "schema_version": 1,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "model": args.model,
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "prompt_tokens": args.prompt_tokens,
        "response_tokens": response_length,
        "total_tokens": args.total_tokens,
        "sample_count": len(decoded_samples),
        "temperature": args.temperature,
        "samples": sample_results,
        "passed": True,
    }
    _write_json(args.summary, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
