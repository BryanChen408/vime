"""Unit tests for colocated vLLM IPC weight sync (UpdateWeightFromTensor)."""

from __future__ import annotations

import importlib
import sys
import types
import warnings
from argparse import Namespace
from contextlib import nullcontext
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest
import torch

MODULE_PATH = "vime.backends.megatron_utils.update_weight.update_weight_from_tensor"


def _install_stubs():
    mpu_stub = MagicMock()
    mpu_stub.get_data_parallel_rank.return_value = 0
    mpu_stub.get_tensor_model_parallel_rank.return_value = 0
    mpu_stub.get_tensor_model_parallel_world_size.return_value = 2
    mpu_stub.get_tensor_model_parallel_group.return_value = "tp_group"
    mpu_stub.get_pipeline_model_parallel_rank.return_value = 0

    megatron_core = types.ModuleType("megatron.core")
    megatron_core.mpu = mpu_stub
    megatron_mod = types.ModuleType("megatron")
    megatron_mod.core = megatron_core
    sys.modules.setdefault("megatron", megatron_mod)
    sys.modules.setdefault("megatron.core", megatron_core)

    ray_mod = types.ModuleType("ray")
    ray_mod.get = lambda refs: refs
    ray_mod.ObjectRef = object
    ray_mod.actor = types.ModuleType("ray.actor")
    ray_mod.actor.ActorHandle = object
    sys.modules.setdefault("ray", ray_mod)
    sys.modules.setdefault("ray.actor", ray_mod.actor)

    import torch.distributed as _dist

    dist_stub = MagicMock()
    dist_stub.get_rank.return_value = 0
    dist_stub.get_world_size.return_value = 1
    dist_stub.get_process_group_ranks.return_value = [0, 1]
    dist_stub.barrier = MagicMock()
    dist_stub.all_gather_object = MagicMock()
    _dist.get_rank = dist_stub.get_rank
    _dist.get_world_size = dist_stub.get_world_size
    _dist.get_process_group_ranks = dist_stub.get_process_group_ranks
    _dist.barrier = dist_stub.barrier
    _dist.all_gather_object = dist_stub.all_gather_object

    vime_utils = types.ModuleType("vime.utils.distributed_utils")
    vime_utils.get_gloo_group = MagicMock(return_value="gloo")
    sys.modules.setdefault("vime.utils.distributed_utils", vime_utils)

    hf_iter_stub = MagicMock()
    hf_iter_stub.get_hf_weight_chunks.return_value = iter([])

    hf_base_mod = types.ModuleType("vime.backends.megatron_utils.update_weight.hf_weight_iterator_base")
    hf_base_mod.HfWeightIteratorBase = MagicMock()
    hf_base_mod.HfWeightIteratorBase.create.return_value = hf_iter_stub

    upw_dist_mod = types.ModuleType("vime.backends.megatron_utils.update_weight.update_weight_from_distributed")
    upw_dist_mod.connect_rollout_engines_from_distributed = MagicMock(return_value="groups")
    upw_dist_mod.disconnect_rollout_engines_from_distributed = MagicMock()
    upw_dist_mod.post_process_weights = MagicMock()
    upw_dist_mod.update_weights_from_distributed = MagicMock(return_value=[])
    upw_dist_mod._begin_vllm_weight_update_session = MagicMock()
    upw_dist_mod._begin_vllm_draft_weight_update_session = MagicMock()
    upw_dist_mod._end_vllm_weight_update_session = MagicMock()
    upw_dist_mod._sync_mtp_draft_enabled = lambda args: bool(
        getattr(args, "enable_mtp_training", False)
        and (getattr(args, "vllm_speculative_config", None) or {}).get("method") == "mtp"
    )

    for key, mod in [
        ("vime.backends.megatron_utils.update_weight.hf_weight_iterator_base", hf_base_mod),
        ("vime.backends.megatron_utils.update_weight.update_weight_from_distributed", upw_dist_mod),
    ]:
        sys.modules.setdefault(key, mod)

    return hf_iter_stub, upw_dist_mod


# Placeholder iterator stored on freshly-built instances; every test that drives a real
# update overrides obj._hf_weight_iterator with its own MagicMock, so this only needs to be
# a non-None object.
_HF_ITER_STUB = MagicMock()
_HF_ITER_STUB.get_hf_weight_chunks.return_value = iter([])

# Modules stubbed by _install_stubs(), plus torch.distributed attributes it overwrites.
# These are installed ONLY for this module's tests (inside the fixture) and restored on
# teardown. Installing at import time leaked the stubs into sibling modules' COLLECTION (and
# left MagicMocks on torch.distributed), one source of the cross-test order-pollution.
_STUBBED_MODULES = (
    "megatron",
    "megatron.core",
    "ray",
    "ray.actor",
    "vime.utils.distributed_utils",
    "vime.backends.megatron_utils.update_weight.hf_weight_iterator_base",
    "vime.backends.megatron_utils.update_weight.update_weight_from_distributed",
)
_DIST_ATTRS = ("get_rank", "get_world_size", "get_process_group_ranks", "barrier", "all_gather_object")


@pytest.fixture(scope="module")
def upw_vllm():
    import torch.distributed as _dist

    saved_mods = {k: sys.modules.get(k) for k in (*_STUBBED_MODULES, MODULE_PATH)}
    saved_dist = {a: getattr(_dist, a, None) for a in _DIST_ATTRS}
    # Pop first so _install_stubs()'s setdefault() actually installs stubs (hermetic).
    for k in _STUBBED_MODULES:
        sys.modules.pop(k, None)
    _install_stubs()
    sys.modules.pop(MODULE_PATH, None)
    try:
        yield importlib.import_module(MODULE_PATH)
    finally:
        for k, original in saved_mods.items():
            if original is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = original
        for a, original in saved_dist.items():
            if original is not None:
                setattr(_dist, a, original)


@dataclass
class _RemoteCall:
    args: tuple
    kwargs: dict


class RecordingRemoteMethod:
    def __init__(self):
        self.calls: list[_RemoteCall] = []

    def remote(self, *args, **kwargs):
        self.calls.append(_RemoteCall(args=args, kwargs=kwargs))
        return "ref"


@dataclass
class RecordingVLLMEngine:
    release_memory_occupation: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    resume_memory_occupation: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    init_weight_transfer_engine: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    start_weight_update: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    start_draft_weight_update: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    finish_weight_update: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    update_weights_from_tensor: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    pause_generation: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    flush_cache: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    continue_generation: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)


def _default_args(**kwargs) -> Namespace:
    base = dict(
        actor_num_nodes=1,
        actor_num_gpus_per_node=4,
        rollout_num_gpus_per_engine=2,
        megatron_to_hf_mode="raw",
        update_weight_buffer_size=1 << 30,
        enable_mtp_training=False,
        vllm_speculative_config=None,
    )
    base.update(kwargs)
    return Namespace(**base)


def _make_instance(upw_vllm, args=None):
    obj = object.__new__(upw_vllm.UpdateWeightFromTensor)
    obj.args = args or _default_args()
    obj.model = []
    obj.weights_getter = lambda: {}
    obj.model_name = "test"
    obj.quantization_config = None
    obj.weight_version = 0
    obj._hf_weight_iterator = _HF_ITER_STUB
    obj.rollout_engines = []
    obj.distributed_rollout_engines = []
    obj.use_distribute = False
    obj._ipc_engine = None
    obj._ipc_gather_group = None
    obj._ipc_gather_src = None
    obj._model_update_groups = None
    obj._is_distributed_src_rank = False
    obj._group_name = "vime"
    obj._ipc_initialized = False
    return obj


def _bind_single_slot(obj, engine, *, src=0):
    """Bind ``obj`` to one colocated engine forming a slot whose leader rank is ``src``."""
    obj.rollout_engines = [engine]
    obj._ipc_engine = engine
    obj._ipc_gather_group = "slot_group"
    obj._ipc_gather_src = src


def _chunks(n=1):
    return [[(f"p.{i}", torch.zeros(2, 2)) for i in range(2)] for _ in range(n)]


def _run_update(obj, *, chunks=None, rank=0, slot_size=1) -> dict:
    """Drive ``update_weights`` with controlled rank / slot size.

    ``slot_size`` is what ``dist.get_world_size(self._ipc_gather_group)`` returns,
    so slot_size==1 takes the direct IPC path and slot_size>1 the gather path.
    Returns counters for barriers and ipc_collect calls.
    """
    chunks = chunks or _chunks(1)
    obj._hf_weight_iterator = MagicMock()
    obj._hf_weight_iterator.get_hf_weight_chunks.side_effect = lambda *args, **kwargs: iter(chunks)

    counters = {"barrier": 0, "ipc_collect": 0}

    def counting_barrier(*args, **kwargs):
        counters["barrier"] += 1

    def counting_ipc_collect(*args, **kwargs):
        counters["ipc_collect"] += 1

    with patch("torch.distributed.get_rank", return_value=rank), patch(
        "torch.distributed.get_world_size", return_value=slot_size
    ), patch("torch.distributed.barrier", side_effect=counting_barrier), patch(
        f"{MODULE_PATH}._device_module",
        return_value=types.SimpleNamespace(ipc_collect=counting_ipc_collect),
    ):
        obj.update_weights()
    return counters


@pytest.mark.unit
def test_colocated_lifecycle_uses_pause_flush_and_weight_transfer_apis(upw_vllm):
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    _bind_single_slot(obj, engine, src=0)

    dummy_info = {"names": ["w"], "dtype_names": ["bfloat16"], "shapes": [[2, 2]], "ipc_handles": [{"u": ("f", ())}]}
    with patch(f"{MODULE_PATH}._build_ipc_update_info_from_named_tensors", return_value=(dummy_info, [])):
        counters = _run_update(obj, chunks=_chunks(2))

    # Colocate quiesce: pause_generation + flush_cache only, no /sleep round-trip;
    # continue_generation resumes. No release/resume_memory_occupation.
    assert len(engine.pause_generation.calls) == 1
    assert len(engine.flush_cache.calls) == 1
    assert len(engine.release_memory_occupation.calls) == 0
    assert len(engine.resume_memory_occupation.calls) == 0
    # vLLM #39212: init runs in connect_rollout_engines, not update_weights.
    assert len(engine.init_weight_transfer_engine.calls) == 0
    assert len(engine.start_weight_update.calls) == 1
    assert engine.start_weight_update.calls[0].kwargs.get("is_checkpoint_format") is True
    assert len(engine.finish_weight_update.calls) == 1
    assert len(engine.continue_generation.calls) == 1
    # ipc_collect: one per HF chunk + one after the loop.
    assert counters["ipc_collect"] == 2 + 1
    # lifecycle barriers (no per-chunk barrier).
    assert counters["barrier"] >= 4


@pytest.mark.unit
def test_send_via_ipc_dispatches_update_weights_from_tensor_with_version(upw_vllm):
    """slot_size=1: every HF chunk fires
    ``engine.update_weights_from_tensor.remote(**fields, weight_version=...)`` —
    same name, parameterized fields, version travels with data (no piggyback onto
    ``finish_weight_update``)."""
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    _bind_single_slot(obj, engine, src=0)

    dummy_info = {"names": ["w"], "dtype_names": ["bfloat16"], "shapes": [[2, 2]], "ipc_handles": [{"u": ("f", ())}]}
    with patch(
        f"{MODULE_PATH}._build_ipc_update_info_from_named_tensors",
        return_value=(dummy_info, []),
    ):
        _run_update(obj, chunks=_chunks(2))

    # 2 HF chunks → 2 IPC RPCs
    assert len(engine.update_weights_from_tensor.calls) == 2
    kwargs = engine.update_weights_from_tensor.calls[0].kwargs
    # fields are passed as explicit kwargs (** expanded from local_info)
    assert kwargs["names"] == dummy_info["names"]
    assert kwargs["dtype_names"] == dummy_info["dtype_names"]
    assert kwargs["shapes"] == dummy_info["shapes"]
    assert kwargs["ipc_handles"] is dummy_info["ipc_handles"]
    # weight_version is the trainer's post-increment version (0 + 1 = 1) as a str
    assert kwargs["weight_version"] == "1"
    # finish_weight_update is a stateless bookend now — no kwargs
    assert len(engine.finish_weight_update.calls) == 1
    assert engine.finish_weight_update.calls[0].kwargs == {}


@pytest.mark.unit
def test_send_via_ipc_dispatches_update_weights_from_tensor_coordinator_multi_gpu(upw_vllm):
    """slot_size > 1: the slot leader (rank == _ipc_gather_src) gathers payloads from
    all slot ranks, merges them, and fires a single update_weights_from_tensor RPC per chunk."""
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    _bind_single_slot(obj, engine, src=0)

    dummy_info_0 = {
        "names": ["w"],
        "dtype_names": ["bfloat16"],
        "shapes": [[2, 2]],
        "ipc_handles": [{"uuid-gpu0": ("f", ())}],
    }
    dummy_info_1 = {
        "names": ["w"],
        "dtype_names": ["bfloat16"],
        "shapes": [[2, 2]],
        "ipc_handles": [{"uuid-gpu1": ("f", ())}],
    }

    def fake_all_gather_object(gathered_payloads, payload, group=None):
        gathered_payloads[0] = "payload0"
        gathered_payloads[1] = "payload1"

    with patch(
        f"{MODULE_PATH}._build_ipc_update_info_from_named_tensors",
        return_value=(dummy_info_0, []),
    ), patch(
        f"{MODULE_PATH}._serialize_ipc_update_info", return_value="payload0"
    ), patch(f"{MODULE_PATH}._deserialize_ipc_update_info", side_effect=[dummy_info_0, dummy_info_1] * 2), patch(
        "torch.distributed.all_gather_object", side_effect=fake_all_gather_object
    ):
        _run_update(obj, chunks=_chunks(2), rank=0, slot_size=2)

    assert len(engine.update_weights_from_tensor.calls) == 2
    kwargs = engine.update_weights_from_tensor.calls[0].kwargs
    assert kwargs["names"] == dummy_info_0["names"]
    assert kwargs["dtype_names"] == dummy_info_0["dtype_names"]
    assert kwargs["shapes"] == dummy_info_0["shapes"]
    assert len(kwargs["ipc_handles"]) == 1
    assert set(kwargs["ipc_handles"][0].keys()) == {"uuid-gpu0", "uuid-gpu1"}
    assert kwargs["weight_version"] == "1"


@pytest.mark.unit
def test_merge_ipc_update_infos_combines_gpu_uuids(upw_vllm):
    info0 = {
        "names": ["w"],
        "dtype_names": ["bfloat16"],
        "shapes": [[2, 2]],
        "ipc_handles": [{"uuid-gpu0": ("f0", ())}],
    }
    info1 = {
        "names": ["w"],
        "dtype_names": ["bfloat16"],
        "shapes": [[2, 2]],
        "ipc_handles": [{"uuid-gpu1": ("f1", ())}],
    }
    merged = upw_vllm._merge_ipc_update_infos([info0, info1])
    assert set(merged["ipc_handles"][0].keys()) == {"uuid-gpu0", "uuid-gpu1"}


@pytest.mark.unit
def test_connect_binds_engine_and_slot_leader_per_gpu_slot(upw_vllm):
    """Each rank binds to its slot's engine; the slot leader (== _ipc_gather_src,
    the lowest trainer rank in the engine GPU range) is the start/finish coordinator."""
    engines = [RecordingVLLMEngine() for _ in range(4)]
    for rank, engine_idx, expected_src in [
        (0, 0, 0),
        (1, 0, 0),
        (2, 1, 2),
        (3, 1, 2),
    ]:
        obj = _make_instance(
            upw_vllm,
            args=_default_args(actor_num_gpus_per_node=8, rollout_num_gpus_per_engine=2),
        )
        with patch("torch.distributed.get_rank", return_value=rank), patch(
            "megatron.core.mpu.get_tensor_model_parallel_rank", return_value=rank % 2
        ), patch("torch.distributed.new_group", return_value="slot_group"):
            obj.connect_rollout_engines(
                engines,
                rollout_engine_lock=MagicMock(),
                engine_gpu_counts=[2, 2, 2, 2],
                engine_gpu_offsets=[0, 2, 4, 6],
            )
        assert obj._ipc_engine is engines[engine_idx]
        assert obj._ipc_gather_src == expected_src
        is_coordinator = rank == obj._ipc_gather_src
        assert is_coordinator is (rank in (0, 2))
        assert obj.use_distribute is False
        assert obj.distributed_rollout_engines == []
        # vLLM #39212: init_weight_transfer_engine fires once during connect (rank 0 only).
        if rank == 0:
            assert len(engines[0].init_weight_transfer_engine.calls) == 1
            assert engines[0].init_weight_transfer_engine.calls[0].args[0] == {"init_info": {}}


@pytest.mark.unit
def test_non_leader_skips_start_finish_and_merged_rpc(upw_vllm):
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    # slot leader is rank 0; we drive update_weights as rank 1 (non-leader).
    _bind_single_slot(obj, engine, src=0)

    dummy_info = {"names": [], "dtype_names": [], "shapes": [], "ipc_handles": []}
    with patch(
        f"{MODULE_PATH}._build_ipc_update_info_from_named_tensors",
        return_value=(dummy_info, []),
    ), patch(
        f"{MODULE_PATH}._serialize_ipc_update_info", return_value="payload"
    ), patch("torch.distributed.all_gather_object") as all_gather_obj:
        _run_update(obj, chunks=_chunks(1), rank=1, slot_size=2)

    all_gather_obj.assert_called_once()
    # non-leader: no start/finish, and no merged update_weights_from_tensor RPC
    assert len(engine.start_weight_update.calls) == 0
    assert len(engine.finish_weight_update.calls) == 0
    assert len(engine.update_weights_from_tensor.calls) == 0


@pytest.mark.unit
def test_ipc_init_runs_once_in_connect(upw_vllm):
    """init_weight_transfer_engine fires once in connect_rollout_engines (rank 0),
    not in update_weights. A second connect call does not re-init."""
    engines = [RecordingVLLMEngine() for _ in range(2)]
    obj = _make_instance(
        upw_vllm,
        args=_default_args(actor_num_gpus_per_node=4, rollout_num_gpus_per_engine=2),
    )
    with patch("torch.distributed.get_rank", return_value=0), patch(
        "megatron.core.mpu.get_tensor_model_parallel_rank", return_value=0
    ), patch("torch.distributed.new_group", return_value="slot_group"):
        obj.connect_rollout_engines(
            engines,
            rollout_engine_lock=MagicMock(),
            engine_gpu_counts=[2, 2],
            engine_gpu_offsets=[0, 2],
        )
    assert obj._ipc_initialized is True
    assert len(engines[0].init_weight_transfer_engine.calls) == 1
    assert len(engines[1].init_weight_transfer_engine.calls) == 1

    # Second connect with _ipc_initialized=True does not re-init.
    engines2 = [RecordingVLLMEngine() for _ in range(2)]
    with patch("torch.distributed.get_rank", return_value=0), patch(
        "megatron.core.mpu.get_tensor_model_parallel_rank", return_value=0
    ), patch("torch.distributed.new_group", return_value="slot_group"):
        obj.connect_rollout_engines(
            engines2,
            rollout_engine_lock=MagicMock(),
            engine_gpu_counts=[2, 2],
            engine_gpu_offsets=[0, 2],
        )
    assert len(engines2[0].init_weight_transfer_engine.calls) == 0
    assert len(engines2[1].init_weight_transfer_engine.calls) == 0


@pytest.mark.unit
def test_mtp_update_replays_full_stream_to_draft(upw_vllm):
    obj = _make_instance(
        upw_vllm,
        args=_default_args(
            enable_mtp_training=True,
            vllm_speculative_config={"method": "mtp"},
        ),
    )
    engine = RecordingVLLMEngine()
    _bind_single_slot(obj, engine, src=0)
    chunks = _chunks(2)
    dummy_info = {
        "names": ["w"],
        "dtype_names": ["float32"],
        "shapes": [[2, 2]],
        "ipc_handles": [{"u": ("f", ())}],
    }

    with patch(
        f"{MODULE_PATH}._build_ipc_update_info_from_named_tensors",
        return_value=(dummy_info, []),
    ):
        _run_update(obj, chunks=chunks)

    assert len(engine.start_weight_update.calls) == 1
    assert len(engine.start_draft_weight_update.calls) == 1
    assert len(engine.finish_weight_update.calls) == 2
    assert len(engine.update_weights_from_tensor.calls) == 2 * len(chunks)
    assert obj._hf_weight_iterator.get_hf_weight_chunks.call_count == 2
    assert len(engine.continue_generation.calls) == 1


@pytest.mark.unit
def test_mtp_update_failure_does_not_resume_generation(upw_vllm):
    obj = _make_instance(
        upw_vllm,
        args=_default_args(
            enable_mtp_training=True,
            vllm_speculative_config={"method": "mtp"},
        ),
    )
    engine = RecordingVLLMEngine()
    _bind_single_slot(obj, engine, src=0)
    engine.start_draft_weight_update.remote = MagicMock(side_effect=RuntimeError("draft unavailable"))
    dummy_info = {
        "names": ["w"],
        "dtype_names": ["float32"],
        "shapes": [[2, 2]],
        "ipc_handles": [{"u": ("f", ())}],
    }

    with patch(
        f"{MODULE_PATH}._build_ipc_update_info_from_named_tensors",
        return_value=(dummy_info, []),
    ), pytest.raises(RuntimeError, match="draft unavailable"):
        _run_update(obj, chunks=_chunks(1))

    assert len(engine.continue_generation.calls) == 0


@pytest.mark.unit
def test_mtp_update_replays_to_both_ipc_and_distributed_engines(upw_vllm):
    obj = _make_instance(
        upw_vllm,
        args=_default_args(
            enable_mtp_training=True,
            vllm_speculative_config={"method": "mtp"},
        ),
    )
    ipc_engine = RecordingVLLMEngine()
    distributed_engine = RecordingVLLMEngine()
    _bind_single_slot(obj, ipc_engine, src=0)
    obj.use_distribute = True
    obj.distributed_rollout_engines = [distributed_engine]
    obj._is_distributed_src_rank = True
    obj._model_update_groups = "groups"
    chunks = _chunks(2)
    dummy_info = {
        "names": ["w"],
        "dtype_names": ["float32"],
        "shapes": [[2, 2]],
        "ipc_handles": [{"u": ("f", ())}],
    }
    upw_vllm.update_weights_from_distributed.reset_mock()
    upw_vllm._begin_vllm_weight_update_session.reset_mock()
    upw_vllm._begin_vllm_draft_weight_update_session.reset_mock()
    upw_vllm._end_vllm_weight_update_session.reset_mock()

    with patch(
        f"{MODULE_PATH}._build_ipc_update_info_from_named_tensors",
        return_value=(dummy_info, []),
    ):
        _run_update(obj, chunks=chunks)

    assert len(ipc_engine.update_weights_from_tensor.calls) == 2 * len(chunks)
    assert upw_vllm.update_weights_from_distributed.call_count == 2 * len(chunks)
    upw_vllm._begin_vllm_weight_update_session.assert_called_once_with([distributed_engine])
    upw_vllm._begin_vllm_draft_weight_update_session.assert_called_once_with([distributed_engine])
    assert upw_vllm._end_vllm_weight_update_session.call_count == 2
    assert len(ipc_engine.continue_generation.calls) == 1
    assert len(distributed_engine.continue_generation.calls) == 1


@pytest.mark.unit
def test_worker_draft_session_selects_drafter_model(upw_vllm):
    draft_model = torch.nn.Linear(2, 2, bias=False)
    target_model = torch.nn.Linear(2, 2, bias=False)
    target_config = object()
    draft_config = object()
    initialize_layerwise_reload = MagicMock()
    reload_module = types.ModuleType("vllm.model_executor.model_loader.reload")
    reload_module.initialize_layerwise_reload = initialize_layerwise_reload
    worker = types.SimpleNamespace(
        _check_weight_transfer_engine=lambda: None,
        _check_nz_disabled=lambda: None,
        _weight_update_active=False,
        device="cpu",
        model_runner=types.SimpleNamespace(
            model=target_model,
            drafter=types.SimpleNamespace(model=draft_model),
        ),
        model_config=target_config,
        vllm_config=types.SimpleNamespace(
            speculative_config=types.SimpleNamespace(draft_model_config=draft_config)
        ),
    )

    with patch.dict(
        sys.modules,
        {"vllm.model_executor.model_loader.reload": reload_module},
    ):
        upw_vllm.vLLMColocateWorkerExtension.start_draft_weight_update(worker)

    initialize_layerwise_reload.assert_called_once_with(draft_model)
    assert worker._weight_update_active is True
    assert worker._is_checkpoint_format is True
    assert worker._vime_weight_update_model is draft_model
    assert worker._vime_weight_update_model_config is draft_config
    assert worker._vime_weight_update_role == "draft"

    with upw_vllm._use_selected_vllm_weight_update_target(worker):
        assert worker.model_runner.model is draft_model
        assert worker.model_config is draft_config
    assert worker.model_runner.model is target_model
    assert worker.model_config is target_config


@pytest.mark.unit
def test_worker_draft_session_fails_without_drafter(upw_vllm):
    worker = types.SimpleNamespace(
        _check_weight_transfer_engine=lambda: None,
        _check_nz_disabled=lambda: None,
        _weight_update_active=False,
        model_runner=types.SimpleNamespace(model=torch.nn.Linear(2, 2), drafter=None),
        vllm_config=types.SimpleNamespace(speculative_config=None),
    )

    with pytest.raises(RuntimeError, match="no draft model"):
        upw_vllm.vLLMColocateWorkerExtension.start_draft_weight_update(worker)


@pytest.mark.unit
def test_worker_draft_loader_receives_complete_checkpoint_stream(upw_vllm):
    class RecordingLoader:
        def __init__(self):
            self.loaded: list[tuple[str, torch.Tensor]] = []

        def load_weights(self, *, weights):
            self.loaded.extend((name, weight.clone()) for name, weight in weights)

    target_model = RecordingLoader()
    draft_model = RecordingLoader()
    worker = types.SimpleNamespace(
        _weight_update_active=True,
        _is_checkpoint_format=True,
        _vime_weight_update_model=draft_model,
        device="cpu",
        model_runner=types.SimpleNamespace(model=target_model),
        vllm_config=object(),
    )
    names = [
        "model.embed_tokens.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ]
    physical_gpu_id = "gpu-0"

    def rebuild_tensor(value, *_args):
        return torch.full((2, 2), value)

    update_info = {
        "names": names,
        "shapes": [[2, 2]] * len(names),
        "ipc_handles": [
            {
                physical_gpu_id: (
                    rebuild_tensor,
                    (float(index + 1), None, None, None, None, None, 7),
                )
            }
            for index, _name in enumerate(names)
        ],
    }
    config_module = types.ModuleType("vllm.config")
    config_module.set_current_vllm_config = lambda _config: nullcontext()
    device_module = types.SimpleNamespace(
        current_device=lambda: 0,
        get_device_properties=lambda _index: types.SimpleNamespace(uuid=physical_gpu_id),
    )

    with patch.dict(sys.modules, {"vllm.config": config_module}), patch(
        f"{MODULE_PATH}._device_module", return_value=device_module
    ), patch.object(torch.accelerator, "synchronize"):
        upw_vllm.vLLMColocateWorkerExtension.update_weights_chunk(worker, update_info)

    assert [name for name, _weight in draft_model.loaded] == names
    for index, (_name, weight) in enumerate(draft_model.loaded):
        assert torch.equal(weight, torch.full((2, 2), float(index + 1)))
    assert target_model.loaded == []


@pytest.mark.unit
def test_restore_fused_moe_weight_loaders_after_ep_parameter_replacement(upw_vllm):
    class FakeFusedMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w13_weight = torch.nn.Parameter(torch.zeros(2, 2))
            self.w2_weight = torch.nn.Parameter(torch.zeros(2, 2))

        def weight_loader(self, *_args, **_kwargs):
            return True

    model = torch.nn.Module()
    moe = FakeFusedMoE()
    model.add_module("moe", moe)
    utils_module = types.ModuleType("vllm.model_executor.utils")

    def set_weight_attrs(weight, attrs):
        for name, value in attrs.items():
            setattr(weight, name, value)

    utils_module.set_weight_attrs = set_weight_attrs
    with patch.dict(sys.modules, {"vllm.model_executor.utils": utils_module}):
        assert upw_vllm._restore_fused_moe_weight_loaders(model) == 2
        assert moe.w13_weight.weight_loader.__self__ is moe
        assert moe.w2_weight.weight_loader.__self__ is moe
        assert upw_vllm._restore_fused_moe_weight_loaders(model) == 0


@pytest.mark.unit
def test_capture_vllm_param_attrs_skips_tensor_transpose_properties(upw_vllm):
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.zeros(2, 3, 4))
    model.weight.weight_loader = object()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        attrs = upw_vllm._capture_vllm_param_attrs(model)["weight"]

    assert attrs["weight_loader"] is model.weight.weight_loader
    assert not {"T", "H", "mT", "mH"}.intersection(attrs)


def test_restore_fused_moe_loader_adapts_ascend_runtime_layout(upw_vllm):
    """A post-process replacement must not feed canonical matrices to runtime
    transposed MoE storage.  This is the exact MTP/EP shape failure seen on NPU:
    the owner loader shards w1/w3 on dim 1 and receives a per-expert transpose.
    """

    class FakeFusedMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.moe_config = types.SimpleNamespace(hidden_dim=8)

            class AscendUnquantizedFusedMoEMethod:
                pass

            self.quant_method = AscendUnquantizedFusedMoEMethod()
            # Ascend runtime layout: w13 [experts, hidden, 2 * intermediate].
            self.w13_weight = torch.nn.Parameter(torch.zeros(2, 8, 4))
            # Ascend runtime layout: w2 [experts, intermediate, hidden].
            self.w2_weight = torch.nn.Parameter(torch.zeros(2, 2, 8))

        def weight_loader(
            self,
            param,
            loaded_weight,
            weight_name,
            shard_id,
            expert_id,
            return_success=False,
        ):
            shard_dim = {"w1": 0, "w2": 1, "w3": 0}[shard_id]
            if getattr(param, "is_transposed", False):
                shard_dim = int(not shard_dim)
            expert_data = param.data[expert_id]
            if shard_id == "w2":
                expert_data.copy_(loaded_weight)
                return True if return_success else None
            shard_size = expert_data.shape[shard_dim] // 2
            if shard_id == "w3":
                start = shard_size
            else:
                start = 0
            expert_data.narrow(shard_dim, start, shard_size).copy_(loaded_weight)
            return True if return_success else None

    model = torch.nn.Module()
    moe = FakeFusedMoE()
    model.add_module("moe", moe)
    utils_module = types.ModuleType("vllm.model_executor.utils")

    def set_weight_attrs(weight, attrs):
        for name, value in attrs.items():
            setattr(weight, name, value)

    with patch.dict(sys.modules, {"vllm.model_executor.utils": utils_module}):
        utils_module.set_weight_attrs = set_weight_attrs
        assert upw_vllm._restore_fused_moe_weight_loaders(model) == 2

    assert moe.w13_weight.is_transposed is True
    loaded = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    moe.w13_weight.weight_loader(
        moe.w13_weight,
        loaded,
        "experts.gate_up_proj.weight",
        shard_id="w1",
        expert_id=0,
        return_success=True,
    )
    assert torch.equal(moe.w13_weight[0, :, :2], loaded.t())

    loaded_w2 = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    moe.w2_weight.weight_loader(
        moe.w2_weight,
        loaded_w2,
        "experts.down_proj.weight",
        shard_id="w2",
        expert_id=0,
        return_success=True,
    )
    assert torch.equal(moe.w2_weight[0], loaded_w2.t())
