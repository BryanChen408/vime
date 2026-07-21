"""CPU-only test for scripts/swe/prepare_swegym_data.py (no dataset / no network).

Covers the docker_image naming (the load-bearing bit) and the row shape the vime
run script + task template consume. The dataset fetch (main()) is not exercised.

Run: TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=<vime repo root> \
     python -m pytest vime_bridge/tests/test_prepare_swegym_data_cpu.py -q
"""
from __future__ import annotations

import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "swe"))
import prepare_swegym_data as psd  # noqa: E402


def test_64_ids_unique_and_complete():
    assert len(psd.INSTANCE_IDS_64) == 64
    assert len(set(psd.INSTANCE_IDS_64)) == 64


def test_docker_image_naming_default():
    assert (
        psd.registry_image_for_instance_id("getmoto__moto-4950")
        == "xingyaoww/sweb.eval.x86_64.getmoto_s_moto-4950:latest"
    )
    # owner/repo casing is lowercased; __ -> _s_
    assert (
        psd.registry_image_for_instance_id("Project-MONAI__MONAI-1571")
        == "xingyaoww/sweb.eval.x86_64.project-monai_s_monai-1571:latest"
    )


def test_docker_image_naming_legacy():
    # legacy repos use the swebench/... _1776_ convention
    assert (
        psd.registry_image_for_instance_id("pvlib__pvlib-python-1234")
        == "swebench/sweb.eval.x86_64.pvlib_1776_pvlib-python-1234:latest"
    )


def test_every_selected_id_maps_to_an_image():
    # no id in our set should raise (all parse into owner__repo-issue)
    for iid in psd.INSTANCE_IDS_64:
        img = psd.registry_image_for_instance_id(iid)
        assert img.endswith(":latest") and "sweb.eval.x86_64." in img


def test_build_row_shape_and_image():
    instance = {
        "instance_id": "getmoto__moto-4950",
        "problem_statement": "  Fix the S3 bug.  ",
        "base_commit": "abc",
        "FAIL_TO_PASS": ["tests/test_s3.py::test_x"],
    }
    row = psd.build_row(instance)
    assert row["prompt"] == [{"role": "user", "content": "Fix the S3 bug."}]  # stripped
    assert row["label"] == ""
    md = row["metadata"]
    assert md["instance_id"] == "getmoto__moto-4950"
    assert md["split"] == "train"
    # full instance dict carried verbatim for swebench_harness grading
    assert md["instance"] is instance
    assert md["docker_image"] == "xingyaoww/sweb.eval.x86_64.getmoto_s_moto-4950:latest"


def test_row_is_json_serializable():
    import json

    instance = {"instance_id": "dask__dask-7191", "problem_statement": "x"}
    json.dumps(psd.build_row(instance), ensure_ascii=True)  # must not raise


def test_augment_row_adds_docker_image_offline():
    # the default offline path: take a colleague-style row (no docker_image) and inject it
    row = {
        "prompt": [{"role": "user", "content": "x"}],
        "label": "",
        "metadata": {
            "instance_id": "getmoto__moto-4950",
            "instance": {"instance_id": "getmoto__moto-4950", "problem_statement": "x"},
            "split": "train",
        },
    }
    out = psd.augment_row(row)
    assert out["metadata"]["docker_image"] == "xingyaoww/sweb.eval.x86_64.getmoto_s_moto-4950:latest"
    assert out["metadata"]["instance_id"] == "getmoto__moto-4950"  # untouched
    assert isinstance(out["metadata"]["instance"], dict)           # untouched


def test_select_64_filters_superset_and_validates():
    # a 293-style superset must filter down to exactly the 64 we have images for
    rows = [{"metadata": {"instance_id": iid}} for iid in psd.INSTANCE_IDS_64]
    rows.append({"metadata": {"instance_id": "someoneelse__repo-1"}})  # extra -> dropped
    sel = psd._select_64(rows)
    assert len(sel) == 64
    assert {r["metadata"]["instance_id"] for r in sel} == set(psd.INSTANCE_IDS_64)
