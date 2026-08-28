"""CPU regressions for colocated NPU optimizer/scratch phase offload."""

from __future__ import annotations

import ast
import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vime.utils.npu_training_state_offloader import NPUTrainingStateOffloader


NUM_GPUS = 0
REPO_ROOT = Path(__file__).resolve().parents[1]


def _no_npu_cache(monkeypatch):
    fake_npu = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)


def _recording_cpu_mover(events):
    def move(tensor, device):
        events.append((id(tensor), str(device)))
        # Clone the storage to exercise the same object-identity contract as a
        # real CPU/NPU move while keeping this test CPU-only.
        tensor.data = tensor.data.clone()

    return move


def test_chained_hdo_roundtrip_preserves_values_aliases_and_tensor_identity(monkeypatch):
    _no_npu_cache(monkeypatch)
    master_param = torch.tensor([1.0, 2.0])
    exp_avg = torch.tensor([3.0, 4.0])
    exp_avg_sq = torch.tensor([5.0, 6.0])
    original_param = object()
    inner_param = object()
    sub_state = {
        inner_param: {
            "master_param": master_param,
            "exp_avg": exp_avg,
            "exp_avg_sq": exp_avg_sq,
        }
    }
    sub_optimizer = SimpleNamespace(state=sub_state)
    hdo = SimpleNamespace(
        sub_optimizers=[sub_optimizer],
        inner_param_to_orig_param={inner_param: original_param},
        state={original_param: sub_state[inner_param]},
    )
    leaf = SimpleNamespace(
        shard_fp32_from_float16_groups=[[master_param]],
        optimizer=hdo,
    )
    chained = SimpleNamespace(chained_optimizers=[leaf])
    scratch = torch.arange(8, dtype=torch.float32)
    global_buffer = SimpleNamespace(buffer={"mpu": scratch})
    events = []
    identities = {id(master_param), id(exp_avg), id(exp_avg_sq)}
    expected = {id(t): t.clone() for t in (master_param, exp_avg, exp_avg_sq)}

    offloader = NPUTrainingStateOffloader(
        global_memory_buffer_getter=lambda: global_buffer,
        # Treat CPU as the accelerator only inside this CPU regression.
        accelerator_device_types=frozenset({"cpu"}),
        tensor_mover=_recording_cpu_mover(events),
    )

    offloader.offload(chained, verbose=False)

    assert offloader.is_offloaded
    assert offloader.stats["optimizer_tensors"] == 3
    assert offloader.stats["scratch_discarded_mb"] > 0
    assert global_buffer.buffer == {}
    assert hdo.state[original_param] is sub_state[inner_param]
    assert {event[0] for event in events} == identities
    assert all(event[1] == "cpu" for event in events)

    restored = offloader.onload(chained, verbose=False)

    assert restored == sum(t.untyped_storage().nbytes() for t in (master_param, exp_avg, exp_avg_sq))
    assert not offloader.is_offloaded
    assert global_buffer.buffer == {}  # scratch is rebuilt lazily, not copied back
    assert len(events) == 6
    assert {id(master_param), id(exp_avg), id(exp_avg_sq)} == identities
    for tensor in (master_param, exp_avg, exp_avg_sq):
        assert torch.equal(tensor, expected[id(tensor)])


def test_full_cpu_hdo_state_is_not_duplicated_or_moved(monkeypatch):
    _no_npu_cache(monkeypatch)
    cpu_state = torch.tensor([7.0, 8.0])
    sub_optimizer = SimpleNamespace(state={object(): {"exp_avg": cpu_state}})
    hdo = SimpleNamespace(
        sub_optimizers=[sub_optimizer],
        inner_param_to_orig_param={},
        state={},
    )
    leaf = SimpleNamespace(shard_fp32_from_float16_groups=[], optimizer=hdo)
    events = []
    offloader = NPUTrainingStateOffloader(
        global_memory_buffer_getter=lambda: SimpleNamespace(buffer={}),
        tensor_mover=_recording_cpu_mover(events),
    )

    offloader.offload(leaf, verbose=False)
    offloader.onload(leaf, verbose=False)

    assert events == []
    assert torch.equal(cpu_state, torch.tensor([7.0, 8.0]))


def test_non_hdo_matches_verl_optimizer_state_keys(monkeypatch):
    _no_npu_cache(monkeypatch)
    exp_avg = torch.tensor([1.0])
    exp_avg_sq = torch.tensor([2.0])
    master_param = torch.tensor([3.0])
    unrelated_cache = torch.tensor([4.0])
    inner = SimpleNamespace(
        state={
            object(): {
                "exp_avg": exp_avg,
                "exp_avg_sq": exp_avg_sq,
                "master_param": master_param,
                "unrelated_cache": unrelated_cache,
            }
        }
    )
    leaf = SimpleNamespace(shard_fp32_from_float16_groups=[], optimizer=inner)
    events = []
    offloader = NPUTrainingStateOffloader(
        global_memory_buffer_getter=lambda: SimpleNamespace(buffer={}),
        accelerator_device_types=frozenset({"cpu"}),
        tensor_mover=_recording_cpu_mover(events),
    )

    offloader.offload(leaf, verbose=False)

    moved_ids = {tensor_id for tensor_id, _ in events}
    assert moved_ids == {id(exp_avg), id(exp_avg_sq), id(master_param)}
    assert id(unrelated_cache) not in moved_ids


def test_partial_offload_failure_rolls_back_already_moved_state(monkeypatch):
    _no_npu_cache(monkeypatch)
    first = torch.tensor([1.0])
    second = torch.tensor([2.0])
    leaf = SimpleNamespace(
        shard_fp32_from_float16_groups=[[first, second]],
        optimizer=None,
    )
    events = []

    def fail_second_move(tensor, device):
        events.append((id(tensor), str(device)))
        if tensor is second:
            raise RuntimeError("copy failed")
        tensor.data = tensor.data.clone()

    offloader = NPUTrainingStateOffloader(
        global_memory_buffer_getter=lambda: SimpleNamespace(buffer={}),
        accelerator_device_types=frozenset({"cpu"}),
        tensor_mover=fail_second_move,
    )

    with pytest.raises(RuntimeError, match="copy failed"):
        offloader.offload(leaf, verbose=False)

    assert events == [(id(first), "cpu"), (id(second), "cpu"), (id(first), "cpu")]
    assert not offloader.is_offloaded
    assert offloader.stats["optimizer_tensors"] == 0


def test_onload_rejects_a_rebuilt_optimizer(monkeypatch):
    _no_npu_cache(monkeypatch)
    optimizer = SimpleNamespace(shard_fp32_from_float16_groups=[], optimizer=None)
    replacement = SimpleNamespace(shard_fp32_from_float16_groups=[], optimizer=None)
    offloader = NPUTrainingStateOffloader(
        global_memory_buffer_getter=lambda: SimpleNamespace(buffer={}),
    )

    offloader.offload(optimizer, verbose=False)

    with pytest.raises(RuntimeError, match="optimizer was replaced"):
        offloader.onload(replacement, verbose=False)

    assert offloader.is_offloaded
    offloader.onload(optimizer, verbose=False)
    assert not offloader.is_offloaded


def test_model_forward_state_is_discarded_and_releases_its_autograd_graph(monkeypatch):
    """Repeated phase handoffs release graphs without touching useful state."""
    _no_npu_cache(monkeypatch)

    class MoEAlltoAllTokenDispatcher:
        def dispatch_preprocess(self, source, weight):
            # Match Megatron's lifecycle: every forward assigns all three
            # fields before combine reads them.
            self.probs = (source * weight).square()
            self.routing_map = torch.ones(source.numel(), dtype=torch.bool)
            self.reversed_local_input_permutation_mapping = torch.arange(source.numel())

        def combine_postprocess(self):
            assert self.probs is not None
            assert self.routing_map is not None
            assert self.reversed_local_input_permutation_mapping is not None
            return self.probs.sum()

    dispatcher = MoEAlltoAllTokenDispatcher()
    dispatcher.sort_input_by_local_experts = torch.arange(4)

    # Same-looking fields on another implementation are deliberately outside
    # this fix; class-specific cleanup prevents accidental cache destruction.
    class MoEAllGatherTokenDispatcher:
        pass

    untouched_dispatcher = MoEAllGatherTokenDispatcher()
    untouched_dispatcher.routing_map = torch.ones(2, dtype=torch.bool)

    class FakeGPTModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.token_dispatcher = dispatcher
            self.other_layer = torch.nn.Module()
            self.other_layer.token_dispatcher = untouched_dispatcher

    model = FakeGPTModel()
    optimizer = SimpleNamespace(shard_fp32_from_float16_groups=[], optimizer=None)
    offloader = NPUTrainingStateOffloader(
        global_memory_buffer_getter=lambda: SimpleNamespace(buffer={}),
    )
    weight_id = id(model.weight)
    weight_value = model.weight.detach().clone()
    persistent_id = id(dispatcher.sort_input_by_local_experts)
    untouched_id = id(untouched_dispatcher.routing_map)

    # Multiple cycles cover both leak accumulation and wake-up reuse. A None
    # dereference would fail in combine_postprocess on the next iteration.
    for _ in range(3):
        model.weight.grad = None
        source = torch.randn(8, requires_grad=True)
        dispatcher.dispatch_preprocess(source, model.weight)
        output = dispatcher.combine_postprocess()
        output.backward()
        weight_grad = model.weight.grad.clone()
        probs = dispatcher.probs
        source_ref = weakref.ref(source)
        probs_ref = weakref.ref(probs)
        direct_tensors = [
            probs,
            dispatcher.routing_map,
            dispatcher.reversed_local_input_permutation_mapping,
        ]
        expected_bytes = sum(t.untyped_storage().nbytes() for t in direct_tensors)
        del source, probs, output, direct_tensors

        offloader.offload(optimizer, verbose=False, model=[model])

        assert dispatcher.probs is None
        assert dispatcher.routing_map is None
        assert dispatcher.reversed_local_input_permutation_mapping is None
        assert offloader.stats["model_runtime_tensors"] == 3
        assert offloader.stats["model_runtime_discarded_mb"] == pytest.approx(
            expected_bytes / (1024 * 1024)
        )

        # ``probs.grad_fn`` was the Python root keeping the upstream graph
        # alive. Clearing it must release the graph rather than accumulate one
        # more copy on every train/rollout cycle.
        gc.collect()
        assert probs_ref() is None
        assert source_ref() is None

        # Parameters, gradients, persistent dispatcher configuration and
        # unvalidated dispatcher implementations must remain byte-for-byte and
        # object-for-object unchanged.
        assert id(model.weight) == weight_id
        assert torch.equal(model.weight, weight_value)
        assert torch.equal(model.weight.grad, weight_grad)
        assert id(dispatcher.sort_input_by_local_experts) == persistent_id
        assert id(untouched_dispatcher.routing_map) == untouched_id

        offloader.onload(optimizer, verbose=False)
        assert dispatcher.probs is None  # next forward assigns it before use


def _actor_method(tree, name):
    actor_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor"
    )
    return next(node for node in actor_class.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _call_lines(method):
    calls = {}
    for node in ast.walk(method):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        calls.setdefault(name, []).append(node.lineno)
    return calls


def test_actor_phase_order_and_shared_only_guard():
    tree = ast.parse((REPO_ROOT / "vime/backends/megatron_utils/actor.py").read_text())
    init_method = _actor_method(tree, "init")
    sleep_calls = _call_lines(_actor_method(tree, "sleep"))
    wake_calls = _call_lines(_actor_method(tree, "wake_up"))

    constructor = next(
        node
        for node in ast.walk(init_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "NPUTrainingStateOffloader"
    )
    enclosing_shared_guard = next(
        node
        for node in ast.walk(init_method)
        if isinstance(node, ast.If)
        and constructor in list(ast.walk(node))
        and "self._rollout_shares_actor_devices" in ast.unparse(node.test)
    )
    assert enclosing_shared_guard is not None

    # Optimizer state must move before DDP storage is invalidated. Restore is
    # the reverse: rebuild model views first, then restore optimizer devices.
    assert (
        sleep_calls["self._training_state_offloader.offload"][0]
        < sleep_calls["destroy_process_groups"][0]
        < sleep_calls["self._weight_offloader.offload"][0]
    )
    offload_call = next(
        node
        for node in ast.walk(_actor_method(tree, "sleep"))
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "self._training_state_offloader.offload"
    )
    assert [ast.unparse(arg) for arg in offload_call.args] == ["self.optimizer"]
    assert [(keyword.arg, ast.unparse(keyword.value)) for keyword in offload_call.keywords] == [
        ("model", "self.model")
    ]
    assert (
        wake_calls["self._weight_offloader.onload"][0]
        < wake_calls["self._training_state_offloader.onload"][0]
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
