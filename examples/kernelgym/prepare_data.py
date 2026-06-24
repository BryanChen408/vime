"""Convert rllm-lilac KernelGym data into vime's prompt-data format.

rllm record (one JSON object per line)::

    {"task": {"problem_id": ..., "reference_code": ..., "entry_point": ...,
              "prompt": ...}, "backend": "triton"|"cuda"}

vime record::

    {"prompt": "```python\\n<reference_code>\\n```",
     "label": "<problem_id>",
     "metadata": {"reference_code": ..., "entry_point": ..., "backend": ...,
                  "problem_id": ...}}

The ``prompt`` string carries the reference code so vime's length filter sees a
realistic prompt; the env reconstructs the full system+user conversation from
``metadata`` at rollout time (so the system prompt lives in code, not data).

Usage::

    python -m examples.kernelgym.prepare_data \\
        --input /path/to/rllm-lilac/data/drkernel_rl_data.jsonl \\
        --output examples/kernelgym/data/kernelgym_train.jsonl

    python -m examples.kernelgym.prepare_data \\
        --input /path/to/rllm-lilac/data/kernelbench_val.jsonl \\
        --output examples/kernelgym/data/kernelgym_val.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Iterable

INITIAL_USER_TEMPLATE = "```python\n{reference_code}\n```"


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def convert_record(rec: dict[str, Any], default_backend: str = "triton") -> dict[str, Any] | None:
    """Map one rllm record to one vime record. Returns None if unusable."""
    # Records are usually {"task": {...}, "backend": ...}; tolerate a flat shape.
    task = rec.get("task") if isinstance(rec.get("task"), dict) else rec
    reference_code = (task.get("reference_code") or "").strip()
    if not reference_code:
        return None
    problem_id = str(task.get("problem_id") or task.get("task_id") or "task")
    entry_point = task.get("entry_point") or "Model"
    backend = rec.get("backend") or task.get("backend") or default_backend

    return {
        "prompt": INITIAL_USER_TEMPLATE.format(reference_code=reference_code),
        "label": problem_id,
        "metadata": {
            "reference_code": reference_code,
            "entry_point": entry_point,
            "backend": backend,
            "problem_id": problem_id,
        },
    }


def _write_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_parquet(records: list[dict], path: str) -> None:
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("parquet output requires pandas/pyarrow; use --format jsonl") from exc
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(records).to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, nargs="+", help="One or more rllm KernelGym jsonl files.")
    parser.add_argument("--output", required=True, help="Output path (.jsonl or .parquet).")
    parser.add_argument("--format", choices=["jsonl", "parquet"], default=None, help="Defaults from --output suffix.")
    parser.add_argument("--default-backend", default="triton", choices=["triton", "cuda"])
    parser.add_argument("--limit", type=int, default=None, help="Keep at most N records (after shuffle).")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records: list[dict] = []
    skipped = 0
    for path in args.input:
        for rec in _iter_jsonl(path):
            converted = convert_record(rec, default_backend=args.default_backend)
            if converted is None:
                skipped += 1
                continue
            records.append(converted)

    if args.shuffle:
        random.Random(args.seed).shuffle(records)
    if args.limit is not None:
        records = records[: args.limit]

    fmt = args.format or ("parquet" if args.output.endswith(".parquet") else "jsonl")
    if fmt == "parquet":
        _write_parquet(records, args.output)
    else:
        _write_jsonl(records, args.output)

    print(f"Wrote {len(records)} records to {args.output} (format={fmt}, skipped={skipped}).")


if __name__ == "__main__":
    main()
