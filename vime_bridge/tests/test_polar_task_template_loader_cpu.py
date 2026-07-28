"""CPU-only unit test for --polar-task-template loading (task_request mode).

Covers `_load_polar_task_template` (None / mapping / YAML-file path / invalid)
and that `resolve_polar_slime_config` selects the submit mode from the template.
No NPU, no network, no training.

Run: TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=<vime repo root> \
     python -m pytest vime_bridge/tests/test_polar_task_template_loader_cpu.py -q
"""
from __future__ import annotations

import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

from types import SimpleNamespace

import pytest

from vime_bridge.config import _load_polar_task_template, resolve_polar_slime_config


def test_load_none_or_empty_returns_empty_mapping():
    assert _load_polar_task_template(None) == {}
    assert _load_polar_task_template("") == {}


def test_load_mapping_is_deepcopied():
    src = {"agent": {"harness": "codex"}}
    out = _load_polar_task_template(src)
    assert out == src
    out["agent"]["harness"] = "mutated"
    assert src["agent"]["harness"] == "codex"  # input must be untouched


def test_load_yaml_file_path(tmp_path):
    p = tmp_path / "swe_task_template.yaml"
    p.write_text(
        "agent:\n"
        "  harness: codex\n"
        "runtime:\n"
        "  backend: docker\n"
        '  image: "{sample.metadata.docker_image}"\n'
    )
    out = _load_polar_task_template(str(p))
    assert out["agent"]["harness"] == "codex"
    assert out["runtime"]["backend"] == "docker"
    # per-sample placeholder must survive verbatim (rendered later by the bridge)
    assert out["runtime"]["image"] == "{sample.metadata.docker_image}"


def test_load_non_mapping_file_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError):
        _load_polar_task_template(str(p))


def test_load_invalid_type_rejected():
    with pytest.raises(ValueError):
        _load_polar_task_template(123)


def _min_args(**overrides):
    """Minimal args stub; resolve_polar_slime_config defaults everything else."""
    base = dict(
        polar_url="http://polar:8080",
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        update_weights_interval=1,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_submit_mode_autoselects_task_request_with_template():
    args = _min_args(polar_task_template={"agent": {"harness": "codex"}})
    cfg = resolve_polar_slime_config(args)
    assert cfg.submit_mode == "task_request"
    assert cfg.task_template["agent"]["harness"] == "codex"


def test_submit_mode_defaults_operator_samples_without_template():
    cfg = resolve_polar_slime_config(_min_args())
    assert cfg.submit_mode == "operator_samples"


def test_explicit_submit_mode_overrides_autoselect():
    args = _min_args(polar_submit_mode="operator_samples")
    assert resolve_polar_slime_config(args).submit_mode == "operator_samples"


def test_task_request_requires_agent_spec():
    args = _min_args(polar_task_template={"runtime": {"backend": "docker"}})
    with pytest.raises(ValueError):
        resolve_polar_slime_config(args)
