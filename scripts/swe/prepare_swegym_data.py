#!/usr/bin/env python3
"""Prepare the 64-task SWE-Gym train JSONL for vime --prompt-data (task_request mode).

The colleague's fully-materialized ``swegym_train_64.jsonl`` (with every instance
dict swebench_harness needs) already exists in the polar repo. The ONLY thing it
lacks for the docker backend is a ``metadata.docker_image`` field, so the default
mode here is a pure OFFLINE transform: read that jsonl, inject docker_image, write
out. No network / HuggingFace needed.

Each output row:
    prompt   = [{"role": "user", "content": <problem_statement>}]
    label    = ""
    metadata = {instance_id, instance (full dict), split, docker_image}

``docker_image`` is load-bearing: the docker task template resolves ``runtime.image``
from ``{sample.metadata.docker_image}``, so every row must carry the tag that is
``docker load``ed on the polar host. Naming mirrors sample_tasks.registry_image_for_instance_id.

Row consumption:
    vime run script:  --input-key prompt --label-key label --metadata-key metadata
    task template:    {sample.metadata.docker_image}  {sample.metadata.instance}

Usage:
    # default: offline, augment the colleague's committed jsonl
    python3 scripts/swe/prepare_swegym_data.py \
        --input-jsonl /workspace/swebench/ProRL-Agent-Server/examples/swegym_slime_grpo/swegym_train_64.jsonl \
        --output      /home/docker/datasets/swegym/swegym_train_64.jsonl

    # fallback: regenerate from the HuggingFace dataset (needs network)
    python3 scripts/swe/prepare_swegym_data.py --from-dataset --output ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

SPLIT = "train"
DEFAULT_OUTPUT = Path("/home/docker/datasets/swegym/swegym_train_64.jsonl")
# The colleague's committed, fully-materialized 64-task jsonl (no docker_image yet).
DEFAULT_INPUT_JSONL = Path(
    "/workspace/swebench/ProRL-Agent-Server/examples/swegym_slime_grpo/swegym_train_64.jsonl"
)
DEFAULT_POLAR_SWEGYM_DIR = (
    "/home/docker/cannbot_debug/ProRL-Agent-Server/examples/swegym_slime_grpo"
)

# The 64 SWE-Gym instances whose docker images we load locally. Verbatim from the
# upstream prepare_data_64.py / build_sifs_from_docker.py selection (tests assert
# this equals the colleague's list). Keep in sync with the loaded images.
INSTANCE_IDS_64 = (
    # conan-io (3)
    "conan-io__conan-13403", "conan-io__conan-15422", "conan-io__conan-15699",
    # dask (6)
    "dask__dask-10972", "dask__dask-7191", "dask__dask-8686",
    "dask__dask-9213", "dask__dask-9378", "dask__dask-9627",
    # getmoto/moto (15)
    "getmoto__moto-4950", "getmoto__moto-4986", "getmoto__moto-5020",
    "getmoto__moto-5134", "getmoto__moto-5582", "getmoto__moto-5587",
    "getmoto__moto-5865", "getmoto__moto-5959", "getmoto__moto-6114",
    "getmoto__moto-6178", "getmoto__moto-6299", "getmoto__moto-6469",
    "getmoto__moto-7111", "getmoto__moto-7167", "getmoto__moto-7212",
    # iterative/dvc (5)
    "iterative__dvc-1651", "iterative__dvc-2231", "iterative__dvc-4778",
    "iterative__dvc-4785", "iterative__dvc-5839",
    # pandas-dev (13)
    "pandas-dev__pandas-49118", "pandas-dev__pandas-49766", "pandas-dev__pandas-50713",
    "pandas-dev__pandas-51605", "pandas-dev__pandas-51936", "pandas-dev__pandas-52076",
    "pandas-dev__pandas-52077", "pandas-dev__pandas-52516", "pandas-dev__pandas-53856",
    "pandas-dev__pandas-57058", "pandas-dev__pandas-57089", "pandas-dev__pandas-57173",
    "pandas-dev__pandas-57957",
    # Project-MONAI (12)
    "Project-MONAI__MONAI-1571", "Project-MONAI__MONAI-2238", "Project-MONAI__MONAI-2696",
    "Project-MONAI__MONAI-3403", "Project-MONAI__MONAI-4109", "Project-MONAI__MONAI-4583",
    "Project-MONAI__MONAI-5183", "Project-MONAI__MONAI-5423", "Project-MONAI__MONAI-5543",
    "Project-MONAI__MONAI-5640", "Project-MONAI__MONAI-6560", "Project-MONAI__MONAI-6756",
    # pydantic (4)
    "pydantic__pydantic-8004", "pydantic__pydantic-8072", "pydantic__pydantic-8316",
    "pydantic__pydantic-8511",
    # python/mypy (6)
    "python__mypy-11135", "python__mypy-11567", "python__mypy-12943",
    "python__mypy-15184", "python__mypy-16869", "python__mypy-5617",
)
assert len(INSTANCE_IDS_64) == len(set(INSTANCE_IDS_64)), "duplicate instance id in INSTANCE_IDS_64"

# Mirror of sample_tasks.registry_image_for_instance_id — vendored so this script
# and its test stay self-contained. Must stay in sync with the polar repo.
_LEGACY_SWEBENCH_IMAGE_REPOS = {
    "marshmallow-code/marshmallow", "pydicom/pydicom", "pylint-dev/astroid",
    "pvlib/pvlib-python", "pyvista/pyvista", "sqlfluff/sqlfluff",
}


def registry_image_for_instance_id(instance_id: str) -> str:
    owner, repo_with_issue = instance_id.split("__", 1)
    repo, issue_id = repo_with_issue.rsplit("-", 1)
    if f"{owner}/{repo}" in _LEGACY_SWEBENCH_IMAGE_REPOS:
        return f"swebench/sweb.eval.x86_64.{owner}_1776_{repo}-{issue_id}:latest"
    suffix = instance_id.replace("__", "_s_").lower()
    return f"xingyaoww/sweb.eval.x86_64.{suffix}:latest"


def augment_row(row: dict[str, Any]) -> dict[str, Any]:
    """Offline: add metadata.docker_image to an already-prepared row (in place)."""
    md = row["metadata"]
    md["docker_image"] = registry_image_for_instance_id(str(md["instance_id"]))
    return row


def build_row(instance: dict[str, Any], split: str = SPLIT) -> dict[str, Any]:
    """Build a fresh row from a raw dataset instance dict (--from-dataset path)."""
    instance_id = str(instance["instance_id"])
    return {
        "prompt": [{"role": "user", "content": str(instance["problem_statement"]).strip()}],
        "label": "",
        "metadata": {
            "instance_id": instance_id,
            "instance": instance,
            "split": split,
            "docker_image": registry_image_for_instance_id(instance_id),
        },
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _select_64(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = set(INSTANCE_IDS_64)
    selected = [r for r in rows if str(r["metadata"]["instance_id"]) in wanted]
    if len(selected) != len(wanted):
        found = {str(r["metadata"]["instance_id"]) for r in selected}
        raise SystemExit(
            f"Found {len(selected)}/{len(wanted)} of the 64 instances in input; missing: "
            + ", ".join(sorted(wanted - found))
        )
    return selected


def _rows_from_dataset() -> list[dict[str, Any]]:
    polar_dir = os.environ.get("POLAR_SWEGYM_DIR", DEFAULT_POLAR_SWEGYM_DIR)
    sys.path.insert(0, polar_dir)
    from sample_tasks import fetch_dataset_instances  # lazy: needs the HF dataset

    wanted = set(INSTANCE_IDS_64)
    instances = [i for i in fetch_dataset_instances(SPLIT, refresh=False)
                 if str(i["instance_id"]) in wanted]
    return [build_row(i) for i in instances]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL,
                        help="Existing prepared jsonl to augment with docker_image (offline).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--from-dataset", action="store_true",
                        help="Regenerate from the HuggingFace SkyRL-v0-293 dataset (needs network).")
    args = parser.parse_args()

    if args.from_dataset:
        rows = _rows_from_dataset()
    elif args.input_jsonl.is_file():
        rows = [augment_row(r) for r in _select_64(_load_jsonl(args.input_jsonl))]
    else:
        raise SystemExit(
            f"Input jsonl not found: {args.input_jsonl}\n"
            "Point --input-jsonl at the colleague's swegym_train_64.jsonl, "
            "or pass --from-dataset to fetch from HuggingFace."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n")
    print(f"Wrote {len(rows)} tasks to {args.output}")


if __name__ == "__main__":
    main()
