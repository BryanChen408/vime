"""CPU regressions for the colocate hand-off memory probe.

The probe exists to attribute the rollout-window trainer residue — the number
that caps a colocated engine's ``gpu_memory_utilization`` — to a specific
hand-off. Two properties have to hold for it to be safe to leave in the loop:
it must cost nothing when disabled (``train.py`` calls it seven times per
rollout), and it must never be the thing that ends a run.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import vime.ray.actor_group as actor_group
import vime.ray.train_actor as train_actor
import vime.utils.memory_utils as memory_utils


NUM_GPUS = 0
REPO_ROOT = Path(__file__).resolve().parents[1]

# Every point in train.py where the training and rollout stacks trade HBM.
EXPECTED_HANDOFF_TAGS = (
    "after generate",
    "after rollout offload",
    "after train",
    "after train offload",
    "after onload_weights",
    "after update_weights",
    "after onload_kv",
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("0", False),
        ("1", True),
        # Deliberately strict. A looser parse would enable the hand-off probes
        # while leaving the train-step probes -- which have always compared
        # against "1" -- silent, splitting one diagnostic into two behaviors.
        ("true", False),
        ("on", False),
        ("", False),
    ],
)
def test_probe_gate_accepts_only_the_documented_value(monkeypatch, value, expected):
    monkeypatch.delenv("VIME_MEM_PROBE", raising=False)
    if value is not None:
        monkeypatch.setenv("VIME_MEM_PROBE", value)
    assert memory_utils.mem_probe_enabled() is expected


def test_lifted_probes_share_the_gate(monkeypatch):
    """The probes moved out of megatron_utils keep honoring the same switch."""
    monkeypatch.delenv("VIME_MEM_PROBE", raising=False)
    monkeypatch.setattr(
        memory_utils.torch,
        "npu",
        SimpleNamespace(is_available=lambda: pytest.fail("probe touched the device while disabled")),
        raising=False,
    )
    assert memory_utils._log_npu_mem("t") is None
    assert memory_utils._log_npu_expandable("t") is None


def test_actor_probe_is_inert_when_disabled(monkeypatch):
    monkeypatch.delenv("VIME_MEM_PROBE", raising=False)
    for name in ("_log_npu_mem", "_log_npu_expandable", "print_memory"):
        monkeypatch.setattr(train_actor, name, lambda *a, **k: pytest.fail(f"{name} ran while disabled"))

    assert train_actor.TrainRayActor.probe_memory(None, "after train offload") is None


def test_actor_probe_reports_device_and_host_together(monkeypatch):
    """All three readings are needed to say who the residue belongs to.

    ``_log_npu_mem`` splits device-used into torch's pool vs everything else
    (CANN/HCCL/AscendC workspace, which no offloader reaches), and
    ``_log_npu_expandable`` says how much of torch's own cache is still
    returnable. Dropping either one leaves the residue unattributable.
    """
    monkeypatch.setenv("VIME_MEM_PROBE", "1")
    seen = {}

    def _record(key, result=None):
        def _fake(tag):
            seen[key] = tag
            return result

        return _fake

    monkeypatch.setattr(train_actor, "_log_npu_mem", _record("mem"))
    monkeypatch.setattr(train_actor, "_log_npu_expandable", _record("exp"))
    monkeypatch.setattr(train_actor, "print_memory", _record("host", {"free_GB": 1.0}))

    result = train_actor.TrainRayActor.probe_memory(None, "after train offload")

    assert set(seen) == {"mem", "exp", "host"}
    assert all(tag == "handoff:after train offload" for tag in seen.values())
    assert result == {"free_GB": 1.0}


def test_actor_probe_survives_a_failing_reading(monkeypatch):
    monkeypatch.setenv("VIME_MEM_PROBE", "1")
    monkeypatch.setattr(train_actor, "_log_npu_mem", lambda tag: (_ for _ in ()).throw(RuntimeError("boom")))

    assert train_actor.TrainRayActor.probe_memory(None, "after generate") is None


def test_group_probe_makes_no_ray_round_trip_when_disabled(monkeypatch):
    monkeypatch.delenv("VIME_MEM_PROBE", raising=False)

    class _Tripwire:
        def __getattr__(self, name):
            pytest.fail(f"actor handle touched via {name!r} while the probe is disabled")

    group = actor_group.RayTrainGroup.__new__(actor_group.RayTrainGroup)
    group._actor_handlers = [_Tripwire(), _Tripwire()]

    assert group.probe_memory("after train offload") is None


def test_group_probe_fans_out_to_every_rank(monkeypatch):
    monkeypatch.setenv("VIME_MEM_PROBE", "1")
    calls = []

    class _Handle:
        def __init__(self, rank):
            self.rank = rank
            self.probe_memory = SimpleNamespace(remote=self._remote)

        def _remote(self, tag):
            calls.append((self.rank, tag))
            return (self.rank, tag)

    monkeypatch.setattr(actor_group.ray, "get", lambda refs: list(refs))
    group = actor_group.RayTrainGroup.__new__(actor_group.RayTrainGroup)
    group._actor_handlers = [_Handle(0), _Handle(1)]

    assert group.probe_memory("after onload_kv") == [(0, "after onload_kv"), (1, "after onload_kv")]
    assert calls == [(0, "after onload_kv"), (1, "after onload_kv")]


def _train_driver_source() -> str:
    return (REPO_ROOT / "train.py").read_text(encoding="utf-8")


def test_every_hand_off_in_the_train_loop_is_probed():
    """Guards against a hand-off being added without a probe next to it.

    The residue account is only readable if the probe set covers every point
    where the two stacks swap; a silent gap reads as "nothing happened there".
    """
    source = _train_driver_source()
    missing = [tag for tag in EXPECTED_HANDOFF_TAGS if tag not in source]
    assert not missing, f"train.py hand-off points without a probe tag: {missing}"

    tree = ast.parse(source)
    probe_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "probe_memory"
    ]
    assert len(probe_calls) == len(EXPECTED_HANDOFF_TAGS)


def test_startup_hand_off_is_probed_too():
    """The pre-loop weight sync trades memory the same way a rollout step does.

    Probing inside the two hand-off helpers rather than at their call sites is
    what makes the startup pass covered; a regression that moves the probes out
    to the loop body would silently drop it.
    """
    tree = ast.parse(_train_driver_source())
    helpers = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_prepare_rollout_memory_handoff", "_finish_rollout_memory_handoff"}
    }
    assert set(helpers) == {"_prepare_rollout_memory_handoff", "_finish_rollout_memory_handoff"}

    for name, node in helpers.items():
        probes = [
            child
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "probe_memory"
        ]
        assert probes, f"{name} must probe so the startup hand-off is covered"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
