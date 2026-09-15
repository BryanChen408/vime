#!/usr/bin/env python3
"""Send an exact-length token prompt to a vLLM completions endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _request_json(url: str, *, body: dict[str, Any] | None, timeout: float) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with _DIRECT_OPENER.open(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
            return response.status, json.loads(payload)
    except urllib.error.HTTPError as error:
        payload = error.read().decode("utf-8", errors="replace")
        try:
            parsed: Any = json.loads(payload)
        except json.JSONDecodeError:
            parsed = payload
        return error.code, parsed


def _wait_until_ready(base_url: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        try:
            status, models = _request_json(f"{base_url}/v1/models", body=None, timeout=10)
            if status == 200:
                return models
            last_error = f"HTTP {status}: {models!r}"
        except Exception as error:  # endpoint is expected to be absent during engine startup
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(5)
    raise TimeoutError(f"vLLM endpoint did not become ready in {timeout}s: {last_error}")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _tile(values: list[int], length: int) -> list[int]:
    if not values:
        raise ValueError("cannot tile an empty source token sequence")
    repeats, remainder = divmod(length, len(values))
    return values * repeats + values[:remainder]


def _load_prompt_tokens(path: Path, sample_index: int, length: int) -> list[int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not samples:
        raise ValueError(f"no samples found in {path}")
    source_tokens = list(samples[sample_index % len(samples)]["tokens"])
    return _tile(source_tokens, length)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:15000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=262145)
    parser.add_argument("--token-id", type=int, default=87)
    parser.add_argument("--source-rollout", type=Path)
    parser.add_argument("--source-sample-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--expected-max-model-len", type=int, default=270336)
    parser.add_argument("--wait-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=14400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.prompt_tokens <= 262144:
        parser.error("--prompt-tokens must cross the original 262144-token boundary")
    if args.prompt_tokens + args.max_tokens > args.expected_max_model_len:
        parser.error("prompt plus generation exceeds --expected-max-model-len")

    started_at = datetime.now(timezone.utc).isoformat()
    models = _wait_until_ready(args.base_url.rstrip("/"), args.wait_timeout)
    if args.source_rollout is not None:
        prompt = _load_prompt_tokens(
            args.source_rollout, args.source_sample_index, args.prompt_tokens
        )
        prompt_pattern = "tiled-source-rollout"
    else:
        prompt = [args.token_id] * args.prompt_tokens
        prompt_pattern = "repeat-token"
    prompt_bytes = b"".join(struct.pack("<I", token) for token in prompt)
    prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
    request_body = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "logprobs": 1,
    }

    start = time.monotonic()
    status, response = _request_json(
        f"{args.base_url.rstrip('/')}/v1/completions",
        body=request_body,
        timeout=args.request_timeout,
    )
    duration = time.monotonic() - start
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    choices = response.get("choices", []) if isinstance(response, dict) else []
    observed_prompt_tokens = usage.get("prompt_tokens")
    observed_completion_tokens = usage.get("completion_tokens")
    checks = {
        "http_200": status == 200,
        "prompt_crossed_original_boundary": args.prompt_tokens > 262144,
        "prompt_token_count_preserved": observed_prompt_tokens == args.prompt_tokens,
        "completion_token_count_matches": observed_completion_tokens == args.max_tokens,
        "choice_returned": len(choices) == 1,
    }
    result = {
        "schema_version": 1,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration,
        "base_url": args.base_url,
        "models_response": models,
        "request": {
            "model": args.model,
            "prompt_tokens": args.prompt_tokens,
            "prompt_token_id": args.token_id,
            "prompt_pattern": prompt_pattern,
            "source_rollout": (
                str(args.source_rollout) if args.source_rollout is not None else None
            ),
            "source_sample_index": args.source_sample_index,
            "unique_prompt_token_ids": len(set(prompt)),
            "prompt_token_ids_sha256_le_u32": prompt_sha256,
            "max_tokens": args.max_tokens,
            "expected_max_model_len": args.expected_max_model_len,
        },
        "http_status": status,
        "response": response,
        "checks": checks,
        "passed": all(checks.values()),
    }
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
