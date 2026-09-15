#!/usr/bin/env python3
"""Resolve a Qwen3.6 YaRN override through the installed vLLM config path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--hf-overrides", type=json.loads, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    return parser.parse_args()


def _rope_parameters(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("config is missing text_config")
    rope_parameters = text_config.get("rope_parameters")
    if not isinstance(rope_parameters, dict):
        raise ValueError("text_config is missing rope_parameters")
    return rope_parameters


def main() -> None:
    args = _parse_args()
    config_path = args.model / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"checkpoint config does not exist: {config_path}")
    if args.max_model_len <= 0:
        raise SystemExit("--max-model-len must be positive")
    if os.environ.get("VLLM_ALLOW_LONG_MAX_MODEL_LEN") != "1":
        raise SystemExit("VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 is required for this preflight")

    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    requested_rope = _rope_parameters(args.hf_overrides)
    if requested_rope.get("rope_type") != "yarn":
        raise SystemExit("HF override must select rope_type=yarn")

    from vllm.engine.arg_utils import EngineArgs

    engine_args = EngineArgs(
        model=str(args.model),
        hf_overrides=args.hf_overrides,
        max_model_len=args.max_model_len,
        skip_tokenizer_init=True,
        language_model_only=True,
    )
    model_config = engine_args.create_model_config()
    resolved_rope = model_config.hf_config.text_config.rope_parameters
    if resolved_rope != requested_rope:
        raise SystemExit(
            "vLLM resolved a different YaRN fingerprint:\n"
            f"requested={json.dumps(requested_rope, sort_keys=True)}\n"
            f"resolved={json.dumps(resolved_rope, sort_keys=True)}"
        )
    if model_config.max_model_len != args.max_model_len:
        raise SystemExit(
            f"vLLM resolved max_model_len={model_config.max_model_len}, "
            f"expected {args.max_model_len}"
        )

    print(
        json.dumps(
            {
                "model": str(args.model),
                "base_rope_parameters": _rope_parameters(base_config),
                "resolved_architectures": model_config.architectures,
                "resolved_max_model_len": model_config.max_model_len,
                "resolved_rope_parameters": resolved_rope,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
