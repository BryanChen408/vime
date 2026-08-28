"""
Colocated vLLM weight sync (trainer + worker)
=============================================

Trainer: ``UpdateWeightFromTensor`` — Megatron → HF chunks → CUDA IPC (Ray).

Worker: ``vLLMColocateWorkerExtension`` — passed to ``vllm serve`` via
``--worker-extension-cls``; patches IPC receive before handle deserialisation.

https://docs.vllm.ai/en/stable/examples/rl/rlhf_ipc/
"""

from __future__ import annotations

import logging
import os
from argparse import Namespace
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle

from vime.utils.distributed_utils import get_gloo_group

from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import (
    _begin_vllm_weight_update_session,
    _end_vllm_weight_update_session,
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    post_process_weights,
    update_weights_from_distributed,
)


def _device_module() -> Any:
    """Return ``torch.npu`` if NPU is available, else ``torch.cuda``.

    This avoids importing ``mindspeed.megatron_adaptor`` in the vLLM
    subprocess (it breaks ``torch.compile``'s ``aot_compile``).
    """
    try:
        import torch_npu  # noqa: F401
        return torch.npu
    except ImportError:
        return torch.cuda


def _current_gpu_uuid() -> str:
    dev = _device_module()
    device_index = dev.current_device()
    props = dev.get_device_properties(device_index)
    return str(props.uuid)


def _build_ipc_update_info_from_named_tensors(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
) -> tuple[dict[str, list], list[torch.Tensor]]:
    """Build vLLM IPC ``update_info`` payload from tensors on this rank's GPU.

    Each handle is keyed by the physical GPU UUID of the producing rank rather
    than by a local device index. The coordinator gathers all ranks' dicts and
    merges them; the receiver looks up its own UUID to pick the matching handle,
    then vLLM unconditionally overwrites ``args[6]`` (device_index) with its own
    local index before ``rebuild_cuda_tensor``. This UUID-keyed routing makes
    the path correct under any ``CUDA_VISIBLE_DEVICES`` ordering without
    relying on a torch reductions monkey-patch.

    Return the contiguous tensor refs alongside the payload. ``reduce_tensor``
    only exports CUDA IPC metadata, so the producer storage must stay alive
    until the receiver opens the handle.
    """
    from torch.multiprocessing.reductions import reduce_tensor

    names: list[str] = []
    dtype_names: list[str] = []
    shapes: list[list[int]] = []
    ipc_handles: list[dict[str, tuple]] = []
    weight_refs: list[torch.Tensor] = []
    gpu_uuid = _current_gpu_uuid()

    for name, tensor in named_tensors:
        names.append(name)
        dtype_names.append(str(tensor.dtype).split(".")[-1])
        shapes.append(list(tensor.shape))
        weight = tensor.detach().contiguous()
        weight_refs.append(weight)
        rebuild_func, ipc_args = reduce_tensor(weight)
        ipc_handles.append({gpu_uuid: (rebuild_func, ipc_args)})

    return (
        {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "ipc_handles": ipc_handles,
        },
        weight_refs,
    )


def _serialize_ipc_update_info(info: dict[str, list]) -> str:
    """Pickle IPC handles for cross-rank gather (Gloo ``all_gather_object`` cannot carry them)."""
    import base64

    import cloudpickle

    return base64.b64encode(cloudpickle.dumps(info)).decode("ascii")


def _deserialize_ipc_update_info(payload: str) -> dict[str, list]:
    import base64

    import cloudpickle

    return cloudpickle.loads(base64.b64decode(payload.encode("ascii")))


def _merge_ipc_update_infos(infos: Sequence[dict[str, list]]) -> dict[str, list]:
    """Merge per-rank IPC payloads so each weight has handles for every GPU UUID in the slot."""
    if not infos:
        raise ValueError("no IPC update_info payloads to merge")
    base = infos[0]
    merged_handles: list[dict[str, tuple]] = []
    num_params = len(base["names"])
    for i in range(num_params):
        combined: dict[str, tuple] = {}
        for info in infos:
            combined.update(info["ipc_handles"][i])
        merged_handles.append(combined)
    return {
        "names": base["names"],
        "dtype_names": base["dtype_names"],
        "shapes": base["shapes"],
        "ipc_handles": merged_handles,
    }


def count_colocated_engines(
    engine_gpu_offsets: Sequence[int], engine_gpu_counts: Sequence[int], total_actor_gpus: int
) -> int:
    """共卡(IPC)引擎数:GPU 槽位完全落在 actor 卡范围内的前缀引擎个数。

    槽位在范围内的引擎走 IPC(与 actor 同节点);第一个越界的引擎及其后全部走
    HCCL。专用段必须从越界处开始且连续(布局校验保证共卡段在前)。
    """
    colocate_engine_nums = 0
    for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
        if gpu_offset + gpu_count > total_actor_gpus:
            break
        colocate_engine_nums += 1
    return colocate_engine_nums


def _resolve_colocated_engine_count(
    args, engine_gpu_offsets: Sequence[int], engine_gpu_counts: Sequence[int]
) -> int:
    """共卡(IPC)引擎数,优先取自单一真源 ``resolve_engine_roles``。

    ``count_colocated_engines`` 按「槽位是否 < 总 actor 卡数」判定,该判据隐含
    「share 段覆盖 actor 全部卡」。部分共卡布局(share 8 < actor 16、专用段在另一
    节点)下远程引擎的槽位仍小于 actor 卡数 → 被误判成共卡 → 对远程引擎尝试 IPC
    直传。真源按 ``(node, device)`` 集合包含判定,不受此影响。

    按 ``gpu_slot`` 对齐:``ServerGroup.engine_gpu_offsets`` 与真源的 ``gpu_slot``
    用同一个公式(``gpu_offset + j * num_gpus_per_engine``),故键天然对应,且
    placeholder 组占掉的槽位不会错位。

    真源用单一 ``args.rollout_num_gpus_per_engine`` 铺槽位,而 PD 的多 group 每组
    引擎卡数可以不同 —— 那种布局下槽位对不上,覆盖不全时退回原判据。这是安全的:
    per-group 卡数不一致的布局都是纯分离(无 share),原判据在其上本就正确。
    """
    from vime.ray.engine_roles import (
        EngineRole,
        EngineRoleError,
        colocated_prefix_count,
        resolve_engine_roles,
    )

    total_actor_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
    try:
        colocation = {role.gpu_slot: role.colocated for role in resolve_engine_roles(args)}
    except EngineRoleError as e:
        logger.warning("resolve_engine_roles failed (%s); falling back to slot-range heuristic", e)
        return count_colocated_engines(engine_gpu_offsets, engine_gpu_counts, total_actor_gpus)

    verdicts = [colocation.get(offset) for offset in engine_gpu_offsets]
    if any(verdict is None for verdict in verdicts):
        missing = [o for o, v in zip(engine_gpu_offsets, verdicts, strict=True) if v is None]
        logger.info(
            "engine roles do not cover GPU slots %s (heterogeneous per-group engine sizes); "
            "falling back to slot-range heuristic",
            missing,
        )
        return count_colocated_engines(engine_gpu_offsets, engine_gpu_counts, total_actor_gpus)

    roles = tuple(
        EngineRole(index=i, gpu_slot=offset, placement=(), colocated=bool(verdict))
        for i, (offset, verdict) in enumerate(zip(engine_gpu_offsets, verdicts, strict=True))
    )
    return colocated_prefix_count(roles)


class UpdateWeightFromTensor:
    """
    Update rollout engines from tensor dict:
    gather TP(GPU NCCL) → convert HF(GPU) → send.
    Colocated: build CUDA IPC handles → all_gather_object(Gloo CPU, over the engine
    slot ranks) → Ray IPC to engine.  Distributed: GPU NCCL broadcast to remote engines.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        """
        Compute param buckets.  IPC Gloo groups are created later in
        ``connect_rollout_engines`` once ``engine_gpu_counts`` is known.
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.update_weight_metrics: dict[str, float] = {}

        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args, model=model, model_name=model_name, quantization_config=quantization_config
        )

        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_engine = None
        self._model_update_groups = None
        # 混合分发(共卡 IPC + 远程 HCCL)的分布式半边状态;connect_rollout_engines 填充。
        self.distributed_rollout_engines: list = []
        self.use_distribute = False
        self._is_distributed_src_rank = False
        # vLLM #39212 IPC transfer-engine init runs once per set of colocated engines.
        self._ipc_initialized = False
        # vLLM IPC handle payloads may use cloudpickle on the Ray/HTTP bridge.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # ------------------------------------------------------------------
    # connect / disconnect
    # ------------------------------------------------------------------

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            # Fallback: assume engines are densely packed (no placeholder gaps).
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        colocate_engine_nums = _resolve_colocated_engine_count(
            self.args, engine_gpu_offsets, engine_gpu_counts
        )

        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
            )
            self._group_name = "vime"
            if self._is_distributed_src_rank:
                if self._model_update_groups is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, self._model_update_groups, self.distributed_rollout_engines
                    )
                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self._group_name,
                    self.distributed_rollout_engines,
                    engine_gpu_counts=distributed_gpu_counts,
                )

        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]

        # Create IPC Gloo gather groups (only on first call; partitioning is
        # fixed across reconnects).
        if self._ipc_gather_group is None:
            for i in range(colocate_engine_nums):
                group_ranks = list(range(colocate_gpu_offsets[i], colocate_gpu_offsets[i] + colocate_gpu_counts[i]))
                new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                if dist.get_rank() in group_ranks:
                    self._ipc_gather_group = new_group
                    self._ipc_gather_src = colocate_gpu_offsets[i]

        # Map training ranks to colocated engine actors.
        for i, engine in enumerate(self.rollout_engines):
            start = colocate_gpu_offsets[i]
            end = start + colocate_gpu_counts[i]
            if start <= dist.get_rank() < end:
                self._ipc_engine = engine

        # vLLM #39212: one-time IPC transfer-engine init on each colocated engine.
        if dist.get_rank() == 0 and self.rollout_engines and not self._ipc_initialized:
            ray.get([engine.init_weight_transfer_engine.remote({"init_info": {}}) for engine in self.rollout_engines])
            self._ipc_initialized = True

    def pop_metrics(self) -> dict[str, float]:
        """
        Return and clear ``update_weight_metrics``. Empty under colocate today;
        kept symmetric with UpdateWeightFromDistributed so the actor can drain unconditionally.
        """
        out, self.update_weight_metrics = self.update_weight_metrics, {}
        return out

    # ------------------------------------------------------------------
    # weight update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        self.weight_version += 1

        # 混合分发:pause/flush/continue 与会话管理必须同时覆盖共卡(IPC)与远程(HCCL)
        # 引擎 —— connect_rollout_engines 之后 self.rollout_engines 只剩共卡子集。
        all_engines = list(self.rollout_engines) + list(self.distributed_rollout_engines)

        rank = dist.get_rank()
        if rank == 0:
            ray.get([engine.pause_generation.remote() for engine in all_engines])
            ray.get([engine.flush_cache.remote() for engine in all_engines])
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=all_engines,
                )
        dist.barrier(group=get_gloo_group())

        # vLLM #39212: enter weight-update mode on each slot leader.
        if self._ipc_engine is not None and rank == self._ipc_gather_src:
            ray.get(self._ipc_engine.start_weight_update.remote(is_checkpoint_format=True))
        if self.use_distribute and self.distributed_rollout_engines:
            _begin_vllm_weight_update_session(self.distributed_rollout_engines)
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = self.weights_getter()

        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
            ray.get(refs)
            # Free GPU tensors so the caching allocator can reuse the blocks,
            # then release CUDA IPC cache entries whose consumers (vLLM engines)
            # have already closed their IPC handles.
            del long_lived_tensors, hf_named_tensors
            _device_module().ipc_collect()

        dist.barrier(group=get_gloo_group())
        # After the barrier all engines have returned, so every rank's last-chunk
        # IPC handles are now released by the consumers.  Clean them up.
        _device_module().ipc_collect()

        # vLLM #39212: exit weight-update mode.
        if self.use_distribute and self.distributed_rollout_engines:
            _end_vllm_weight_update_session(self.distributed_rollout_engines)
        if self._ipc_engine is not None and rank == self._ipc_gather_src:
            ray.get(self._ipc_engine.finish_weight_update.remote())
        dist.barrier(group=get_gloo_group())

        # int4/fp4 post_process
        if rank == 0:
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=all_engines,
                )
            ray.get([engine.continue_generation.remote() for engine in all_engines])
        dist.barrier(group=get_gloo_group())

    def _send_hf_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        all_refs = []

        refs_colocated, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            weight_version=self.weight_version,
        )
        all_refs.extend(refs_colocated)

        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                hf_named_tensors,
                packed=False,
            )
            if refs_distributed:
                all_refs.extend(refs_distributed)

        return all_refs, long_lived_tensors


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version,
) -> tuple[list[ObjectRef], Any]:
    # Placeholder ranks (GPU slots reserved but no engine) have no gather group.
    # all_gather_object is only collective among group members, so we skip entirely.
    if ipc_gather_group is None:
        return [], None

    slot_size = dist.get_world_size(ipc_gather_group)
    if slot_size <= 1:
        local_info, weight_refs = _build_ipc_update_info_from_named_tensors(hf_named_tensors)
        ref = ipc_engine.update_weights_from_tensor.remote(**local_info, weight_version=str(weight_version))
        return [ref], weight_refs

    local_info, weight_refs = _build_ipc_update_info_from_named_tensors(hf_named_tensors)
    payload = _serialize_ipc_update_info(local_info)

    # all_gather_object is monkey-patched for ReloadableProcessGroup; gather_object
    # is not (it fails after a Megatron reload).
    gathered_payloads = [None] * slot_size
    dist.all_gather_object(gathered_payloads, payload, group=ipc_gather_group)

    refs = []
    if dist.get_rank() == ipc_gather_src:
        if any(p is None for p in gathered_payloads):
            raise RuntimeError(f"Missing IPC payloads in slot {ipc_gather_src}; got {gathered_payloads!r}")
        slot_infos = [_deserialize_ipc_update_info(p) for p in gathered_payloads]
        merged = _merge_ipc_update_infos(slot_infos)
        refs.append(ipc_engine.update_weights_from_tensor.remote(**merged, weight_version=str(weight_version)))

    return refs, weight_refs


# ---------------------------------------------------------------------------
# vLLM worker extension (loaded by ``--worker-extension-cls`` in colocate mode)
# ---------------------------------------------------------------------------


class _VLLMHijack:
    """Monkey-patch vLLM IPC receive so CUDA IPC handles deserialize on the correct GPU."""

    @staticmethod
    def hijack() -> None:
        from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferEngine

        if getattr(IPCWeightTransferEngine, "_vime_receive_patched", False):
            return

        _orig = IPCWeightTransferEngine.receive_weights

        def _vime_receive_weights(self, update_info, load_weights, _orig=_orig):
            _orig(self, update_info, load_weights)

        IPCWeightTransferEngine.receive_weights = _vime_receive_weights
        IPCWeightTransferEngine._vime_receive_patched = True  # type: ignore[attr-defined]

        # vllm-ascend-023 的 NPUWorker 原生 start_weight_update 缺 weight_loader 重补丁:
        # CaMem wake_up 的 w2/w13 transpose 重接线会造出丢失 weight_loader 的裸
        # Parameter,layerwise reload 的 load_weights 会在 MoE 专家权重上炸
        # AttributeError。在引擎子进程内包一层:先 capture/restore attrs,再委托原生。
        try:
            from vllm_ascend.worker.worker import NPUWorker
        except Exception:
            NPUWorker = None
        if NPUWorker is not None and not getattr(NPUWorker, "_vime_start_patched", False):
            _orig_start = NPUWorker.start_weight_update

            def _vime_start_weight_update(self, is_checkpoint_format: bool = True, _orig=_orig_start):
                model = self.model_runner.model
                _capture_vllm_param_attrs(model)  # 首次后 no-op
                patched = _restore_vllm_param_attrs(model)
                if patched:
                    logger.debug(
                        "Re-patched weight_loader attrs on %d params: %s",
                        len(patched),
                        ", ".join(sorted(patched)[:10]),
                    )
                return _orig(self, is_checkpoint_format)

            NPUWorker.start_weight_update = _vime_start_weight_update
            NPUWorker._vime_start_patched = True  # type: ignore[attr-defined]


def _copy_vllm_param_attrs(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Copy vLLM custom attrs (set via ``set_weight_attrs``) from *src* to *dst*.

    ``torch.nn.Parameter(data)`` creates a fresh tensor that drops every
    non-standard attribute, so this must be called whenever a param is
    re-created during post-weight-sync transpose (was in vllm-ascend
    worker.wake_up, now runs after :meth:`finish_weight_update`).
    """
    _SKIP = frozenset(
        {
            "data", "dtype", "device", "grad", "grad_fn", "layout",
            "name", "names", "ndim", "output_nr", "requires_grad",
            "retains_grad", "shape", "size",
        }
    )
    for key in dir(src):
        if key.startswith("_") or key in _SKIP:
            continue
        try:
            val = getattr(src, key)
        except (AttributeError, RuntimeError):
            continue
        if callable(val) and key not in ("weight_loader",):
            continue
        try:
            setattr(dst, key, val)
        except (AttributeError, TypeError, RuntimeError):
            pass


# Per-model cache of vLLM parameter attributes, keyed by ``id(model)``.
# Captured once on first start_weight_update so that weight_loader and
# friends can be re-applied after sleep level 2 (which discards param
# objects via CaMem pool reset) or after layerwise reload unfuses params.
_VLLM_PARAM_ATTRS_CACHE: dict[int, dict[str, dict[str, object]]] = {}


def _capture_vllm_param_attrs(model) -> dict[str, dict[str, object]]:
    """Iterate all named parameters of *model* and capture every non-standard
    attribute (``weight_loader``, ``weight_loader_impl``, ``output_dim``,
    ``input_dim``, etc.) into a nested dict ``{param_name: {attr: value}}``.

    The result is cached in the module-level ``_VLLM_PARAM_ATTRS_CACHE`` keyed
    by ``id(model)`` so subsequent calls are cheap.

    Returns the captured dict (same as the cached value).
    """
    model_id = id(model)
    if model_id in _VLLM_PARAM_ATTRS_CACHE:
        return _VLLM_PARAM_ATTRS_CACHE[model_id]

    _SKIP = frozenset(
        {
            "data", "dtype", "device", "grad", "grad_fn", "layout",
            "name", "names", "ndim", "output_nr", "requires_grad",
            "retains_grad", "shape", "size",
        }
    )
    captured: dict[str, dict[str, object]] = {}
    for name, param in model.named_parameters():
        attrs: dict[str, object] = {}
        for key in dir(param):
            if key.startswith("_") or key in _SKIP:
                continue
            try:
                val = getattr(param, key)
            except (AttributeError, RuntimeError):
                continue
            # Skip non-weight_loader callables — they are bound methods that
            # won't survive a param re-creation anyway.
            if callable(val) and key not in ("weight_loader", "weight_loader_impl"):
                continue
            attrs[key] = val
        if attrs:
            captured[name] = attrs

    _VLLM_PARAM_ATTRS_CACHE[model_id] = captured
    return captured


def _restore_vllm_param_attrs(model, captured: dict[str, dict[str, object]] | None = None) -> set[str]:
    """Re-apply vLLM custom attributes to model parameters.

    Uses the cached attribute map (see :func:`_capture_vllm_param_attrs`) so
    that weight syncing works after sleep level 2 (which discards param
    objects) or after layerwise reload unfuses parameters.

    Already-present attributes are NOT overwritten (idempotent), so repeated
    calls are safe.

    Returns the set of parameter names that were re-patched.
    """
    if captured is None:
        model_id = id(model)
        if model_id not in _VLLM_PARAM_ATTRS_CACHE:
            _capture_vllm_param_attrs(model)
        captured = _VLLM_PARAM_ATTRS_CACHE[model_id]

    patched: set[str] = set()
    params_dict = dict(model.named_parameters())
    for name, attrs in captured.items():
        param = params_dict.get(name)
        if param is None:
            continue
        needs_patch = False
        for key in attrs:
            if not hasattr(param, key):
                needs_patch = True
                break
        if not needs_patch:
            continue
        for key, val in attrs.items():
            if not hasattr(param, key):
                try:
                    setattr(param, key, val)
                except (AttributeError, TypeError, RuntimeError):
                    pass
        patched.add(name)
    return patched


class vLLMColocateWorkerExtension:
    """vLLM ``--worker-extension-cls`` entry for colocated IPC weight sync."""

    def __new__(cls, **kwargs):
        _VLLMHijack.hijack()
        return super().__new__(cls)

    # ── Three-phase weight update protocol ────────────────────────────────────
    # Mirrors SkyRL's NewInferenceWorkerWrap. Callable via /collective_rpc from
    # VLLMEngine.update_weights_chunk / update_weights_chunk on the trainer side.

    def update_weights_chunk(self, update_info: dict) -> None:
        """Receive and load a single chunk of weights via CUDA IPC.

        Accepts the ``update_info`` dict produced by
        ``VLLMEngine.update_weights`` / ``update_weights``, which
        carries ``ipc_handles_pickled`` (cloudpickle + base64 serialised CUDA
        IPC handles assembled by the trainer's
        ``IPCWeightTransferEngine.trainer_send_weights``).

        Deserialises IPC handles inline (the same pattern as SkyRL's
        NewInferenceWorkerWrap) and reconstructs each weight tensor before
        loading into the model — no dependency on
        ``weight_transfer_engine.receive_weights``.

        Args:
            update_info: Dict with keys:
                - names: list[str]
                - dtype_names: list[str]
                - shapes: list[list[int]]
                - ipc_handles_pickled: base64(cloudpickle({gpu_uuid: (func, args)}))
        """
        if not getattr(self, "_weight_update_active", False):
            raise RuntimeError("start_weight_update must be called before update_weights.")

        import base64

        import cloudpickle

        # Deserialise cloudpickle+b64 encoded IPC handles back to raw callables.
        inner = dict(update_info)
        if "ipc_handles_pickled" in inner:
            inner["ipc_handles"] = cloudpickle.loads(base64.b64decode(inner.pop("ipc_handles_pickled")))

        names: list[str] = inner["names"]
        shapes: list[list[int]] = inner["shapes"]
        ipc_handles: list[dict] = inner["ipc_handles"]

        device_index = _device_module().current_device()
        physical_gpu_id = str(_device_module().get_device_properties(device_index).uuid)

        # Reconstruct weights from per-tensor IPC handles (one handle per
        # parameter — the vLLM IPCWeightTransferEngine.trainer_send_weights
        # convention, which differs from SkyRL's single-packed-buffer approach).
        weights: list[tuple[str, torch.Tensor]] = []
        for name, _shape, ipc_handle in zip(names, shapes, ipc_handles, strict=True):
            if physical_gpu_id not in ipc_handle:
                raise ValueError(
                    f"IPC handle not found for GPU UUID {physical_gpu_id}. "
                    f"Available UUIDs: {list(ipc_handle.keys())}"
                )
            func, args = ipc_handle[physical_gpu_id]
            # Index 6 is the device_index in torch's rebuild_cuda_tensor tuple.
            # Remap to the local (receiver-side) device index.
            list_args = list(args)
            list_args[6] = device_index
            # 克隆成引擎自有 tensor:IPC 重建的是 trainer 显存的借用视图,而 layerwise
            # reload 会把未完整层的 loaded_weight 引用缓冲到后续 chunk(layerwise.py
            # loaded_weights.append(bound_args) 不拷贝);trainer 在 ray.get 返回后即
            # del + ipc_collect 并复用该显存 → 缓冲引用读到被覆写的内容(2026-08-21
            # 共享专家一致性校验炸点,仅 IPC 路径;HCCL 路径 tensor 引擎自有,无此问题)。
            # 末尾的 torch.accelerator.synchronize() 保证克隆在 RPC 返回前完成。
            weight: torch.Tensor = func(*list_args).clone()
            weights.append((name, weight))

        # Load weights into the model.
        from vllm.config import set_current_vllm_config

        model = self.model_runner.model
        with set_current_vllm_config(self.vllm_config), torch.device(self.device):
            if self._is_checkpoint_format:
                model.load_weights(weights=iter(weights))
            else:
                for name, weight in weights:
                    param = model.get_parameter(name)
                    param.copy_(weight)

        # Ensure the receiver has finished consuming the IPC tensors before
        # the sender drops its reference on the next barrier.
        torch.accelerator.synchronize()

    # ── IPC weight-update lifecycle (init / start / finish) ───────────────────
    # 当前 vllm-ascend-023 的 NPUWorker 已原生提供这三个方法(worker.py:289-375),
    # 扩展类若再定义会触发 vLLM 的"属性冲突"断言导致 WorkerProc 起不来(2026-08-21
    # 实爆)。故生命周期走原生方法;原生 start_weight_update 缺的 weight_loader
    # 重补丁由 _VLLMHijack 在引擎子进程内包装注入(见 hijack)。
