#!/usr/bin/env python3
"""Score an exact rollout replay with vLLM prompt logprobs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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


def _extract_chosen_logprobs(
    prompt_tokens: list[int], prompt_logprobs: list[Any]
) -> list[float | None]:
    if len(prompt_logprobs) != len(prompt_tokens):
        raise ValueError(
            "vLLM prompt token/logprob lengths differ: "
            f"{len(prompt_tokens)} != {len(prompt_logprobs)}"
        )
    chosen: list[float | None] = []
    for index, (token_id, position) in enumerate(zip(prompt_tokens, prompt_logprobs, strict=True)):
        if index == 0 and position is None:
            chosen.append(None)
            continue
        if not isinstance(position, dict):
            raise ValueError(f"invalid prompt_logprobs entry at position {index}: {position!r}")
        token_data = position.get(str(token_id), position.get(token_id))
        if not isinstance(token_data, dict) or "logprob" not in token_data:
            raise ValueError(
                f"target token {token_id} missing from prompt_logprobs at position {index}"
            )
        value = float(token_data["logprob"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite logprob at position {index}: {value}")
        chosen.append(value)
    return chosen


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:15000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-total-tokens", type=int, default=300_000)
    parser.add_argument("--wait-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=14400)
    args = parser.parse_args()

    source = torch.load(args.input, map_location="cpu", weights_only=False)
    source_samples = source.get("samples") if isinstance(source, dict) else None
    if not source_samples:
        raise ValueError(f"no samples found in {args.input}")

    _wait_until_ready(args.base_url.rstrip("/"), args.wait_timeout)
    started_at = datetime.now(timezone.utc).isoformat()
    scored_samples = []
    sample_results = []
    for sample_index, source_sample in enumerate(source_samples):
        sample = copy.deepcopy(source_sample)
        prompt_tokens = [int(token) for token in sample["tokens"]]
        if len(prompt_tokens) != args.expected_total_tokens:
            raise ValueError(
                f"sample {sample_index} has {len(prompt_tokens)} tokens, "
                f"expected {args.expected_total_tokens}"
            )
        request_body = {
            "model": args.model,
            "prompt": prompt_tokens,
            "max_tokens": 0,
            "temperature": 0,
            "echo": True,
            "prompt_logprobs": 0,
            "return_token_ids": True,
        }
        started = time.monotonic()
        status, response = _request_json(
            f"{args.base_url.rstrip('/')}/v1/completions",
            request_body,
            timeout=args.request_timeout,
        )
        duration = time.monotonic() - started
        if status != 200 or not isinstance(response, dict):
            raise RuntimeError(f"sample {sample_index} scoring failed: HTTP {status}: {response!r}")
        choices = response.get("choices") or []
        if len(choices) != 1:
            raise ValueError(f"sample {sample_index} returned {len(choices)} choices")
        choice = choices[0]
        returned_tokens = choice.get("prompt_token_ids")
        if returned_tokens != prompt_tokens:
            raise ValueError(f"sample {sample_index} prompt token IDs were not preserved")
        chosen = _extract_chosen_logprobs(prompt_tokens, choice.get("prompt_logprobs") or [])
        prompt_length = len(prompt_tokens) - int(sample["response_length"])
        response_logprobs = chosen[prompt_length:]
        if any(value is None for value in response_logprobs):
            raise ValueError(f"sample {sample_index} response contains an unscored token")
        scored = [float(value) for value in response_logprobs]
        if len(scored) != int(sample["response_length"]):
            raise ValueError(f"sample {sample_index} response logprob length mismatch")
        sample["rollout_log_probs"] = scored
        metadata = dict(sample.get("metadata") or {})
        validation = dict(metadata.get("qwen36_yarn_validation") or {})
        validation.update(
            vllm_prompt_logprobs=True,
            vllm_model=args.model,
            vllm_total_tokens=len(prompt_tokens),
        )
        metadata["qwen36_yarn_validation"] = validation
        sample["metadata"] = metadata
        scored_samples.append(sample)
        sample_results.append(
            {
                "sample_index": sample_index,
                "duration_seconds": duration,
                "prompt_tokens": len(prompt_tokens),
                "response_tokens": len(scored),
                "unique_prompt_token_ids": len(set(prompt_tokens)),
                "usage": response.get("usage"),
                "response_logprobs": _stats(scored),
            }
        )
        print(json.dumps(sample_results[-1], sort_keys=True), flush=True)

    payload = dict(source)
    payload["samples"] = scored_samples
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)

    result = {
        "schema_version": 1,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "sample_count": len(scored_samples),
        "samples": sample_results,
        "passed": True,
    }
    _write_json(args.summary, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
