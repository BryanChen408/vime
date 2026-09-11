from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from vime.ray.engine_roles import colocated_prefix_count, resolve_engine_roles
from vime.ray.resource_layout import load_resource_layout


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run-qwen36-35b-polar-multi-pd.sh"
SYNC_HYBRID = REPO_ROOT / "scripts" / "start_sync_hybrid.sh"
SYNC_SINGLE52 = REPO_ROOT / "scripts" / "start_sync_hybrid_single52.sh"


def _source_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _mode_block() -> str:
    source = _source_text(RUNNER)
    start = source.index('if [ "${FEAT_SYNC_ROLLOUT}" = "1" ]; then')
    end = source.index("\nROLLOUT_ARGS=(", start)
    return source[start:end]


def _gate_block() -> str:
    source = _source_text(RUNNER)
    start = source.index("# ─── 训推模式闸门 ───")
    end = source.index("# ─── polar 数据 / 端点 ───", start)
    return source[start:end]


def _shell_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def _evaluate_mode(*, sync: str = "0", tis_disabled: str = "0", factor: str = "1.0") -> subprocess.CompletedProcess[str]:
    script = f"""
set -e
FEAT_SYNC_ROLLOUT={sync}
POLAR_DISABLE_TIS={tis_disabled}
POLAR_SYNC_OVERSUBSCRIBE_FACTOR={factor}
{_mode_block()}
printf 'fn=%s\\n' "$ROLLOUT_FN"
printf 'sched=%s\\n' "${{SCHED_ARGS[*]-}}"
printf 'tis=%s\\n' "${{TIS_ARGS[*]-}}"
"""
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=_shell_env(),
        text=True,
        capture_output=True,
        check=False,
    )


def _evaluate_gate(
    *,
    sync: str = "0",
    colocate: str = "0",
    train_entry: str = "train_async.py",
    durable: str = "0",
    offload: str = "0",
) -> subprocess.CompletedProcess[str]:
    script = f"""
set -e
MASTER_ADDR=80.48.5.56
ROLLOUT_NODE_IP=80.48.5.64
RESOURCE_LAYOUT=layout.yaml
FEAT_COLOCATE={colocate}
TRAIN_ENTRY={train_entry}
FEAT_OFFLOAD={offload}
FEAT_SYNC_ROLLOUT={sync}
POLAR_POLICY_TRANSITION_ENABLED={durable}
{_gate_block()}
printf 'ok\\n'
"""
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=_shell_env(),
        text=True,
        capture_output=True,
        check=False,
    )


def _stdout_fields(result: subprocess.CompletedProcess[str]) -> dict[str, str]:
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def test_runner_is_shell_valid() -> None:
    result = subprocess.run(["bash", "-n", str(RUNNER)], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


def test_save_hf_default_does_not_leak_parameter_expansion_brace() -> None:
    source = _source_text(RUNNER)
    assert "SAVE_HF='/workspace/Qwen3.6-35B-A3B_vime_polar/rollout_{rollout_id}'" in source
    assert '--save-hf "${SAVE_HF}"' in source
    assert '${SAVE_HF:-/workspace/Qwen3.6-35B-A3B_vime_polar/rollout_{rollout_id}}' not in source


def test_async_is_the_default_and_keeps_session_pool() -> None:
    result = _evaluate_mode()
    assert result.returncode == 0, result.stderr
    fields = _stdout_fields(result)
    assert fields["fn"] == "vime_bridge.rollout.generate_rollout_polar_async"
    assert fields["sched"] == (
        "--rollout-max-async-level 1 --rollout-scheduler-mode session_pool "
        "--rollout-max-active-sessions 16 --rollout-release-on-postrun"
    )
    assert fields["tis"] == "--use-tis"


def test_sync_is_explicit_and_has_no_async_scheduler_state() -> None:
    result = _evaluate_mode(sync="1")
    assert result.returncode == 0, result.stderr
    fields = _stdout_fields(result)
    assert fields["fn"] == "vime_bridge.rollout.generate_rollout_polar_sync"
    assert fields["sched"] == "--rollout-sync-oversubscribe-factor 1.0"
    assert "session_pool" not in fields["sched"]
    assert fields["tis"] == "--use-tis"


@pytest.mark.parametrize("factor", ["1.0", "1.25", "1.5"])
def test_sync_factor_boundaries_are_forwarded(factor: str) -> None:
    result = _evaluate_mode(sync="1", factor=factor)
    assert result.returncode == 0, result.stderr
    assert _stdout_fields(result)["sched"] == f"--rollout-sync-oversubscribe-factor {factor}"


@pytest.mark.parametrize("factor", ["0.99", "1.51", "nan", "inf", "not-a-number"])
def test_sync_factor_out_of_contract_is_rejected(factor: str) -> None:
    result = _evaluate_mode(sync="1", factor=factor)
    assert result.returncode != 0
    assert "POLAR_SYNC_OVERSUBSCRIBE_FACTOR" in result.stderr


def test_tis_can_be_disabled_without_changing_rollout_mode() -> None:
    result = _evaluate_mode(tis_disabled="1")
    assert result.returncode == 0, result.stderr
    fields = _stdout_fields(result)
    assert fields["fn"] == "vime_bridge.rollout.generate_rollout_polar_async"
    assert fields["sched"].startswith("--rollout-max-async-level")
    assert fields["tis"] == ""


def test_mode_gate_rejects_sync_with_async_training_entry() -> None:
    result = _evaluate_gate(sync="1", train_entry="train_async.py")
    assert result.returncode != 0
    assert "requires TRAIN_ENTRY=train.py" in result.stderr


def test_mode_gate_rejects_colocate_without_sync() -> None:
    result = _evaluate_gate(colocate="1", train_entry="train.py")
    assert result.returncode != 0
    assert "FEAT_COLOCATE=1 requires FEAT_SYNC_ROLLOUT=1" in result.stderr


def test_durable_gate_requires_sync_train_entry_and_offload() -> None:
    async_entry = _evaluate_gate(durable="1", train_entry="train_async.py")
    assert async_entry.returncode != 0
    assert "durable Polar transitions require TRAIN_ENTRY=train.py" in async_entry.stderr

    without_offload = _evaluate_gate(durable="1", train_entry="train.py", offload="0")
    assert without_offload.returncode != 0
    assert "durable Polar transitions require FEAT_OFFLOAD=1" in without_offload.stderr

    valid = _evaluate_gate(durable="1", train_entry="train.py", offload="1")
    assert valid.returncode == 0, valid.stderr


def test_sync_launcher_pins_sync_durable_and_probe_contract() -> None:
    source = _source_text(SYNC_HYBRID)
    lines = {line.strip() for line in source.splitlines()}
    assert any(line.startswith("FEAT_SYNC_ROLLOUT=1") for line in lines)
    assert any(line.startswith("POLAR_SYNC_OVERSUBSCRIBE_FACTOR=1.0") for line in lines)
    assert any(line.startswith("POLAR_POLICY_TRANSITION_ENABLED=1") for line in lines)
    assert any(line.startswith("VIME_MEM_PROBE=1") for line in lines)
    for async_only in ("POLAR_MAX_ACTIVE_SESSIONS", "POLAR_DRAIN_SESSIONS", "POLAR_MAX_OFF_POLICY_STEPS"):
        assert async_only not in source


def test_single52_launcher_uses_local_layout_and_explicit_sync() -> None:
    source = _source_text(SYNC_SINGLE52)
    assert "resource_layout.single52_hybrid_colocate.yaml" in source
    assert 'FEAT_SYNC_ROLLOUT="${FEAT_SYNC_ROLLOUT:-1}"' in source
    assert 'POLAR_SYNC_OVERSUBSCRIBE_FACTOR="${POLAR_SYNC_OVERSUBSCRIBE_FACTOR:-1.0}"' in source
    assert 'VIME_MEM_PROBE="${VIME_MEM_PROBE:-1}"' in source


def test_launchers_do_not_reintroduce_online_mtp_or_yarn() -> None:
    source = "\n".join(_source_text(path) for path in (RUNNER, SYNC_HYBRID, SYNC_SINGLE52))
    assert re.search(r"(?i)\bmtp\b|yarn|rope_parameters", source) is None


@pytest.mark.parametrize(
    ("filename", "actor", "rollout", "per_engine", "shared", "dedicated", "prefix"),
    [
        ("resource_layout.dual56train57infer_pd.yaml", 16, 8, 2, 0, 8, 0),
        ("resource_layout.hybrid56cola64infer.yaml", 16, 24, 4, 16, 8, 4),
        ("resource_layout.single52_hybrid_colocate.yaml", 8, 12, 2, 8, 4, 4),
        ("resource_layout.single52_homo_colocate.yaml", 16, 12, 2, 12, 0, 6),
    ],
)
def test_resource_layout_topology_fingerprint(
    filename: str,
    actor: int,
    rollout: int,
    per_engine: int,
    shared: int,
    dedicated: int,
    prefix: int,
) -> None:
    layout = load_resource_layout(REPO_ROOT / "scripts" / filename)
    assert layout.actor_num_gpus == actor
    assert layout.rollout_num_gpus == rollout
    assert layout.rollout_num_gpus_per_engine == per_engine
    assert layout.rollout_shared_num_gpus == shared
    assert layout.rollout_dedicated_num_gpus == dedicated

    args = type(
        "Args",
        (),
        {"resource_layout_spec": layout, "rollout_num_gpus_per_engine": per_engine},
    )()
    roles = resolve_engine_roles(args)
    assert len(roles) == rollout // per_engine
    assert colocated_prefix_count(roles) == prefix
