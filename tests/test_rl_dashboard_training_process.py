from __future__ import annotations

import re
import subprocess
import threading
import time

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


def test_resource_panel_combines_local_and_peer_cache_metrics():
    page = rl_dashboard.PAGE

    assert page.count("id=npu_cluster") == 1
    assert "id=npu_peer" not in page
    assert "id=local_vllm" not in page
    assert "renderEngines(ld.engines,'local')" in page
    assert "renderEngines(d.engines,'peer')" in page
    assert "Prefix命中" in page
    assert "KV=当前缓存块占用率" in page


def test_pointed_log_does_not_stat_nfs_target(tmp_path, monkeypatch):
    pointer = tmp_path / "active-log"
    target = "/mnt/pipeline-data/train_log/active.log"
    pointer.write_text(target + "\n")
    monkeypatch.setattr(rl_dashboard, "LOG_POINTER", str(pointer))

    def unexpected_target_io(*_args, **_kwargs):
        raise AssertionError("request path must not stat the pointed NFS log")

    monkeypatch.setattr(rl_dashboard.os.path, "isfile", unexpected_target_io)
    monkeypatch.setattr(rl_dashboard.os.path, "getsize", unexpected_target_io)

    assert rl_dashboard._pointed_log() == (True, target)


def test_log_update_scheduler_is_single_flight_and_nonblocking(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def blocked_update():
        started.set()
        release.wait(timeout=2)

    monkeypatch.setattr(rl_dashboard, "update_log", blocked_update)
    try:
        before = time.monotonic()
        assert rl_dashboard.schedule_log_update() is True
        assert time.monotonic() - before < 0.2
        assert started.wait(timeout=1)

        before = time.monotonic()
        assert rl_dashboard.schedule_log_update() is False
        assert time.monotonic() - before < 0.2
    finally:
        release.set()
        deadline = time.monotonic() + 1
        while rl_dashboard.LOG_IO_LOCK.locked() and time.monotonic() < deadline:
            time.sleep(0.01)

    assert not rl_dashboard.LOG_IO_LOCK.locked()


def test_log_frontend_retries_busy_background_reads():
    assert rl_dashboard.PAGE.count("if(!d||d.busy)") == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
