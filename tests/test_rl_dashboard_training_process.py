from __future__ import annotations

import re
import subprocess

import pytest

import rl_dashboard


NUM_GPUS = 0


@pytest.mark.parametrize(
    "command",
    [
        "python3 train.py --rollout-backend vllm",
        "python3 /workspace/vime/train.py --rollout-backend vllm",
        "python3 /workspace/vime/train_async.py --rollout-backend vllm",
    ],
)
def test_training_process_pattern_matches_both_entrypoints(command):
    assert re.search(rl_dashboard.TRAIN_PROCESS_PATTERN, command)


@pytest.mark.parametrize(
    "command",
    [
        "python3 pretrain.py",
        "python3 train.py.bak",
        "python3 rl_dashboard.py",
    ],
)
def test_training_process_pattern_rejects_other_commands(command):
    assert re.search(rl_dashboard.TRAIN_PROCESS_PATTERN, command) is None


def test_training_pid_query_covers_sync_and_async_entrypoints(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="101\n202\n")

    monkeypatch.setattr(rl_dashboard.subprocess, "run", fake_run)

    assert rl_dashboard._training_pids() == ["101", "202"]
    assert calls == [
        (
            ["pgrep", "-f", rl_dashboard.TRAIN_PROCESS_PATTERN],
            {"capture_output": True, "text": True, "timeout": 3},
        )
    ]


def test_train_alive_uses_shared_training_pid_query(monkeypatch):
    monkeypatch.setattr(rl_dashboard, "_training_pids", lambda: ["101"])
    assert rl_dashboard.train_alive() is True

    monkeypatch.setattr(rl_dashboard, "_training_pids", lambda: [])
    assert rl_dashboard.train_alive() is False


def test_train_uptime_uses_oldest_matching_entrypoint(monkeypatch):
    monkeypatch.setattr(rl_dashboard, "_training_pids", lambda: ["101", "202"])

    def fake_run(command, **kwargs):
        elapsed = "45\n" if command[-1] == "101" else "120\n"
        return subprocess.CompletedProcess(command, 0, stdout=elapsed)

    monkeypatch.setattr(rl_dashboard.subprocess, "run", fake_run)

    assert rl_dashboard.train_uptime() == 120


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
