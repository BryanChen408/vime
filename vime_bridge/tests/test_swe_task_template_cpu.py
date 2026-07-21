"""CPU-only test that the SWE-Gym task template is consumable + renders correctly.

Loads scripts/swe/swe_task_template.yaml.in (after the ${AGENT_CLI_DIR} envsubst
that the run script does at launch), resolves it through the --polar-task-template
path, and renders it against a fake SWE sample. Guards the docker/codex/swebench
wiring and the sole-placeholder -> dict rendering of `instance`. No NPU / network.

Run: TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=<vime repo root> \
     python -m pytest vime_bridge/tests/test_swe_task_template_cpu.py -q
"""
from __future__ import annotations

import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

from pathlib import Path
from types import SimpleNamespace

from vime_bridge.config import render_task_payload, resolve_polar_slime_config

TEMPLATE_IN = Path(__file__).resolve().parents[2] / "scripts" / "swe" / "swe_task_template.yaml.in"
FAKE_CLI_DIR = "/host/agent_cli"


def _load_template_to(tmp_path: Path) -> Path:
    # Mirror the run script's restricted envsubst '${AGENT_CLI_DIR}' (leaves $HOME literal).
    text = TEMPLATE_IN.read_text().replace("${AGENT_CLI_DIR}", FAKE_CLI_DIR)
    assert "$HOME" in text and "${AGENT_CLI_DIR}" not in text  # $HOME preserved, cli-dir substituted
    out = tmp_path / "swe_task_template.yaml"
    out.write_text(text)
    return out


def _args(template_path: Path):
    return SimpleNamespace(
        polar_url="http://polar:8080",
        polar_task_template=str(template_path),
        rollout_batch_size=1,
        n_samples_per_prompt=4,
        update_weights_interval=1,
    )


def _swe_sample():
    instance = {
        "instance_id": "getmoto__moto-4950",
        "base_commit": "deadbeef",
        "problem_statement": "Fix the S3 bug.",
        "test_patch": "diff --git ...",
    }
    return SimpleNamespace(
        prompt=[{"role": "user", "content": "Fix the S3 bug."}],
        response="",
        label=None,
        index=0,
        group_index=0,
        status=None,
        metadata={
            "instance_id": "getmoto__moto-4950",
            "docker_image": "swegym/sweb.eval.x86_64.getmoto_moto-4950:latest",
            "instance": instance,
        },
    )


def test_swe_template_loads_as_task_request(tmp_path):
    cfg = resolve_polar_slime_config(_args(_load_template_to(tmp_path)))
    assert cfg.submit_mode == "task_request"          # agent present -> task_request
    assert cfg.task_template["agent"]["harness"] == "codex"


def test_swe_template_renders_docker_codex_swebench(tmp_path):
    cfg = resolve_polar_slime_config(_args(_load_template_to(tmp_path)))
    args = _args(_load_template_to(tmp_path))
    sample = _swe_sample()

    payload = render_task_payload(
        args=args, config=cfg, sample=sample, instruction="Fix the S3 bug.",
        rollout_id=7, task_position=0, num_rollouts=4,
    )

    rt = payload["runtime"]
    assert rt["backend"] == "docker"
    # per-sample docker image rendered from metadata
    assert rt["image"] == "swegym/sweb.eval.x86_64.getmoto_moto-4950:latest"
    # ${AGENT_CLI_DIR} envsubst landed in the volume mount
    assert f"{FAKE_CLI_DIR}:/opt/node:ro" in rt["kwargs"]["volumes"]
    # $HOME preserved literally for in-container expansion (not host-expanded)
    assert "$HOME/.venv/bin" in rt["prepare"][0]["command"]

    assert payload["agent"]["harness"] == "codex"
    ev = payload["evaluator"]
    assert ev["strategy"] == "swebench_harness"
    assert ev["refresh_runtime"] is True
    # the crux: `instance` must render to the whole DICT, not a stringified one
    inst = ev["config"]["instance"]
    assert isinstance(inst, dict)
    assert inst["instance_id"] == "getmoto__moto-4950"
    assert inst["problem_statement"] == "Fix the S3 bug."

    # bridge-injected fields
    assert payload["num_samples"] == 4
    assert payload["task_id"]  # rendered from task_id_template
    assert payload["instruction"] == "Fix the S3 bug."
