"""CPU regressions for the colocated NPU allocator handoff lifecycle."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import train as train_driver
import vime.ray.train_actor as train_actor
import vime.utils.memory_utils as memory_utils


NUM_GPUS = 0
GIB = 1024**3
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ("expandable_segments:True", True),
        ("max_split_size_mb:64, expandable_segments:1", True),
        ("expandable_segments:False", False),
        ("expandable_segments:0,max_split_size_mb:64", False),
        ("", False),
    ],
)
def test_expandable_segments_policy_is_read_from_actor_environment(monkeypatch, config, expected):
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", config)
    assert memory_utils.expandable_segments_enabled() is expected


def test_runtime_allocator_switch_uses_torch_npu_api(monkeypatch):
    settings = []
    fake_npu = SimpleNamespace(
        memory=SimpleNamespace(_set_allocator_settings=settings.append),
    )
    monkeypatch.setattr(memory_utils, "is_npu", lambda: True)
    monkeypatch.setattr(memory_utils.torch, "npu", fake_npu)

    memory_utils.set_expandable_segments(False)
    memory_utils.set_expandable_segments(True)

    assert settings == ["expandable_segments:False", "expandable_segments:True"]


def test_aggressive_empty_cache_retries_while_at_least_one_gib_is_reclaimed(monkeypatch):
    events = []
    reserved = iter([4 * GIB, 2 * GIB, 2 * GIB, int(1.5 * GIB)])
    allocated = iter([3 * GIB, 2 * GIB, 2 * GIB, int(1.75 * GIB)])
    fake_npu = SimpleNamespace(
        is_available=lambda: True,
        memory_reserved=lambda: next(reserved),
        memory_allocated=lambda: next(allocated),
        empty_cache=lambda: events.append("empty_cache"),
        synchronize=lambda: events.append("synchronize"),
    )
    monkeypatch.setattr(memory_utils, "is_npu", lambda: True)
    monkeypatch.setattr(memory_utils.torch, "npu", fake_npu)
    monkeypatch.setattr(memory_utils.gc, "collect", lambda: events.append("gc"))

    memory_utils.aggressive_empty_cache(force_sync=True, max_retries=3)

    assert events == [
        "gc",
        "empty_cache",
        "synchronize",
        "gc",
        "empty_cache",
        "synchronize",
    ]


def _shared_actor_state():
    return SimpleNamespace(
        args=SimpleNamespace(offload_train=True),
        _rollout_shares_actor_devices=True,
        _memory_handoff_active=False,
        _restore_expandable_segments=False,
    )


def test_actor_handoff_is_idempotent_and_restores_the_previous_policy(monkeypatch):
    actor = _shared_actor_state()
    events = []
    monkeypatch.setattr(train_actor, "is_npu", lambda: True)
    monkeypatch.setattr(train_actor, "expandable_segments_enabled", lambda: True)
    monkeypatch.setattr(train_actor, "set_expandable_segments", lambda enabled: events.append(("set", enabled)))
    monkeypatch.setattr(
        train_actor,
        "aggressive_empty_cache",
        lambda force_sync: events.append(("cleanup", force_sync)),
    )
    monkeypatch.setattr(train_actor, "print_memory", lambda label: events.append(("memory", label)))

    train_actor.TrainRayActor.prepare_memory_handoff(actor)
    train_actor.TrainRayActor.prepare_memory_handoff(actor)
    train_actor.TrainRayActor.finish_memory_handoff(actor)
    train_actor.TrainRayActor.finish_memory_handoff(actor)

    assert events == [
        ("set", False),
        ("cleanup", True),
        ("memory", "after memory handoff prepare (before rollout weights wake)"),
        ("memory", "after rollout KV cache wake"),
        ("set", True),
    ]
    assert actor._memory_handoff_active is False
    assert actor._restore_expandable_segments is False


def test_actor_handoff_does_not_enable_expandable_segments_when_originally_disabled(monkeypatch):
    actor = _shared_actor_state()
    settings = []
    monkeypatch.setattr(train_actor, "is_npu", lambda: True)
    monkeypatch.setattr(train_actor, "expandable_segments_enabled", lambda: False)
    monkeypatch.setattr(train_actor, "set_expandable_segments", settings.append)
    monkeypatch.setattr(train_actor, "aggressive_empty_cache", lambda **kwargs: None)
    monkeypatch.setattr(train_actor, "print_memory", lambda label: None)

    train_actor.TrainRayActor.prepare_memory_handoff(actor)
    train_actor.TrainRayActor.finish_memory_handoff(actor)

    assert settings == [False]


def test_actor_handoff_does_not_touch_disaggregated_async_actor(monkeypatch):
    actor = _shared_actor_state()
    actor._rollout_shares_actor_devices = False
    touched = []
    monkeypatch.setattr(train_actor, "is_npu", lambda: True)
    monkeypatch.setattr(train_actor, "set_expandable_segments", lambda enabled: touched.append(enabled))
    monkeypatch.setattr(train_actor, "aggressive_empty_cache", lambda **kwargs: touched.append("cleanup"))

    train_actor.TrainRayActor.prepare_memory_handoff(actor)

    assert touched == []
    assert actor._memory_handoff_active is False


def test_finish_restores_allocator_state_even_if_memory_probe_fails(monkeypatch):
    actor = _shared_actor_state()
    actor._memory_handoff_active = True
    actor._restore_expandable_segments = True
    restored = []
    monkeypatch.setattr(
        train_actor,
        "print_memory",
        lambda label: (_ for _ in ()).throw(RuntimeError("probe failed")),
    )
    monkeypatch.setattr(train_actor, "set_expandable_segments", restored.append)

    with pytest.raises(RuntimeError, match="probe failed"):
        train_actor.TrainRayActor.finish_memory_handoff(actor)

    assert restored == [True]
    assert actor._memory_handoff_active is False
    assert actor._restore_expandable_segments is False


class _RemoteCall:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def remote(self):
        self.events.append(self.name)
        return f"{self.name}-ref"


def test_driver_brackets_rollout_weights_and_kv_with_actor_handoff(monkeypatch):
    events = []
    args = SimpleNamespace(offload_rollout=True)
    actor = SimpleNamespace(
        prepare_memory_handoff=lambda: events.append("prepare_actor"),
        finish_memory_handoff=lambda: events.append("finish_actor"),
        probe_memory=lambda tag: events.append(f"probe:{tag}"),
    )
    rollout = SimpleNamespace(
        onload_weights=_RemoteCall("onload_weights", events),
        onload_kv=_RemoteCall("onload_kv", events),
    )
    monkeypatch.setattr(train_driver.ray, "get", lambda ref: events.append(f"wait:{ref}"))

    train_driver._prepare_rollout_memory_handoff(args, actor, rollout, "rollout 7")
    train_driver._finish_rollout_memory_handoff(args, actor, rollout, "rollout 7")

    # The probes sit inside the bracket, not around it: the onload_kv reading
    # has to be taken while the handoff allocator policy is still in force.
    assert events == [
        "prepare_actor",
        "onload_weights",
        "wait:onload_weights-ref",
        "probe:rollout 7 after onload_weights",
        "onload_kv",
        "wait:onload_kv-ref",
        "probe:rollout 7 after onload_kv",
        "finish_actor",
    ]


def test_driver_does_not_restore_actor_allocator_when_kv_wake_fails(monkeypatch):
    events = []
    args = SimpleNamespace(offload_rollout=True)
    actor = SimpleNamespace(
        finish_memory_handoff=lambda: events.append("finish_actor"),
        probe_memory=lambda tag: events.append(f"probe:{tag}"),
    )
    rollout = SimpleNamespace(onload_kv=_RemoteCall("onload_kv", events))

    def fail_wait(ref):
        raise RuntimeError("KV wake failed")

    monkeypatch.setattr(train_driver.ray, "get", fail_wait)

    with pytest.raises(RuntimeError, match="KV wake failed"):
        train_driver._finish_rollout_memory_handoff(args, actor, rollout, "rollout 7")

    # Neither the probe nor the allocator restore runs: a failed KV wake must
    # leave the handoff open so the caller stays fail-closed.
    assert events == ["onload_kv"]


def test_weight_sync_cleanup_runs_after_reloadable_process_groups_are_destroyed():
    tree = ast.parse((REPO_ROOT / "vime/backends/megatron_utils/actor.py").read_text())
    actor_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor"
    )
    update_method = next(
        node for node in actor_class.body if isinstance(node, ast.FunctionDef) and node.name == "update_weights"
    )

    calls = {}
    for node in ast.walk(update_method):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.setdefault(node.func.id, []).append(node.lineno)

    assert len(calls["destroy_process_groups"]) == 1
    assert len(calls["aggressive_empty_cache"]) == 1
    assert calls["destroy_process_groups"][0] < calls["aggressive_empty_cache"][0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
