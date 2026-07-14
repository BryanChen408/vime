import logging
import os
import re
from pathlib import Path

# TODO: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import get_args

from vime.utils import megatron_bridge_utils

try:
    # Here we patch out the `validate_non_overlapping_shards_metadata` in both functions
    # because it is really slow for large models with many shards.
    # TODO: find a less hacky way to do this.
    import torch.distributed as dist
    import torch.distributed._shard.sharding_spec as shard_spec
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
    from torch.distributed._shard.sharded_tensor.shard import Shard
    from torch.distributed._shard.sharded_tensor.utils import _parse_and_validate_remote_device
    from torch.distributed._shard.sharding_spec.api import EnumerableShardingSpec

    def __post_init__(self):
        pass

    EnumerableShardingSpec.__post_init__ = __post_init__

    @classmethod
    def _init_from_local_shards_and_global_metadata(  # type: ignore[override]
        cls,
        local_shards: list[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None,
        init_rrefs=False,
        sharding_spec=None,
    ) -> ShardedTensor:
        """
        Initialize a ShardedTensor with local shards and a global
        ShardedTensorMetadata built on each rank.

        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
        process_group = cls._normalize_pg(process_group)
        current_rank = dist.get_rank()  # intentional to get global rank

        shards_metadata = sharded_tensor_metadata.shards_metadata

        local_shard_metadatas = []

        # collect local shard metadatas from the global sharded_tensor_metadata
        for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
            rank, local_device = _parse_and_validate_remote_device(process_group, shard_metadata.placement)

            if current_rank == rank:
                local_shard_metadatas.append(shard_metadata)

        shards_metadata = sharded_tensor_metadata.shards_metadata
        tensor_properties = sharded_tensor_metadata.tensor_properties

        if sharding_spec is None:
            spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
        else:
            spec = sharding_spec

        sharded_tensor = ShardedTensor.__new__(
            ShardedTensor,
            spec,
            sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )

        # done validation, add local_shards
        sharded_tensor._local_shards = local_shards
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        # run post initialization, i.e. map registration, rpc initialization
        sharded_tensor._post_init()
        return sharded_tensor

    ShardedTensor._init_from_local_shards_and_global_metadata = _init_from_local_shards_and_global_metadata

except ImportError:
    pass

logger = logging.getLogger(__name__)

__all__ = ["save_checkpoint"]


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    # ref: how megatron `load_checkpoint` gets directory
    args = get_args()
    load_path = args.load

    assert Path(load_path).exists() and _is_dir_nonempty(
        load_path
    ), f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"

    if _is_megatron_checkpoint(load_path):
        # [GDN 校验 fail-loud] GDN verify(_verify_gdn_load)只挂在 HF-load 路径 _load_checkpoint_hf,
        #   与 slime 一致(slime 的 verify 只在 mbridge HF-online-load,torch_dist 路径无 verify)。
        #   torch_dist/megatron ckpt 走这里 → verify 不触发。若设了 QWEN36_VERIFY_LOAD 却走本路径,
        #   大声提示,避免静默空跑(曾误导:env 设了但 log 无校验输出)。
        if os.environ.get("QWEN36_VERIFY_LOAD"):
            hf_hint = getattr(args, "hf_checkpoint", None) or "<HF 目录>"
            print(
                f"[VERIFY_LOAD][WARN] QWEN36_VERIFY_LOAD 已设,但 --load={load_path} 是 megatron/torch_dist ckpt "
                f"→ 走 _load_checkpoint_megatron,GDN verify **不触发**(verify 仅在 HF-load 路径 _load_checkpoint_hf)。"
                f"要跑校验:临时把 --load 指向 HF 目录(如 {hf_hint})再设 QWEN36_VERIFY_LOAD=1;"
                f"该校验验的是 bridge 的 HF↔megatron GDN 映射,即产出此 torch_dist 的同一套转换逻辑。",
                flush=True,
            )
        return _load_checkpoint_megatron(
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        )
    else:
        return _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )


def _is_megatron_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7}", Path(path).name)
    )


def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
    from megatron.bridge import AutoBridge

    import vime_plugins.megatron_bridge  # noqa: F401

    logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={load_path})")

    with megatron_bridge_utils.patch_megatron_model(ddp_model):
        bridge = megatron_bridge_utils.patch_auto_bridge_hf_config(
            AutoBridge.from_hf_pretrained(load_path, trust_remote_code=True)
        )
        bridge.load_hf_weights(ddp_model)

    # [GDN 加载校验,移植 slime checkpoint._verify_gdn_load] GDN(linear_attn:qkvz/ba 分组 +
    #   conv1d TP 交织)是全模型最易错的权重布局,megatron.bridge 加载后无兜底。设 QWEN36_VERIFY_LOAD
    #   触发一次比对(actor 本 rank 分片 == merge(HF) 切片,逐 in_proj/conv1d/dt_bias/A_log/out_proj/norm)。
    #   未设 QWEN36_VERIFY_PREPUSH 则校验后 sys.exit(纯验证);默认 env 不设 → 零开销、不影响正常训练。
    if os.environ.get("QWEN36_VERIFY_LOAD"):
        _verify_gdn_load(ddp_model, load_path, args, exit_after=not os.environ.get("QWEN36_VERIFY_PREPUSH"))

    # Copied from Megatron-core :: load_checkpoint (with simplifications)
    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    # We can see `successfully loaded checkpoint from ... [ t 1/2, p 1/1 ] at iteration 0`
    # when loading Megatron, thus it is 0
    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far


def _verify_gdn_load(models, load_path, args=None, tag="VERIFY_LOAD", exit_after=True):
    """[GDN 加载校验,移植 slime checkpoint._verify_gdn_load(LOAD 部分)]

    校验 actor 当前 GDN 权重本 rank 分片 == merge(HF) 切片,逐张量比对
    in_proj / conv1d / dt_bias / A_log / out_proj / norm。GDN(linear_attn:qkvz/ba 分组融合 +
    conv1d TP 交织 + 各张量不同切分轴)是全模型最易错的权重布局,megatron.bridge 加载后无兜底。
    QWEN36_VERIFY_LOAD 触发;未设 QWEN36_VERIFY_PREPUSH(即 exit_after=True)则校验后 sys.exit。
    仅移植 LOAD 部分;slime 的 CONVERT(update_weights 转换链路)依赖 slime 专属 convert_qwen3_5_to_hf。
    """
    import json
    import re
    import struct
    import sys

    import torch
    from megatron.bridge.models.conversion.param_mapping import (
        _fuse_gdn_separate_to_grouped,
        merge_gdn_linear_weights,
    )
    from megatron.core import mpu

    from vime_plugins.mbridge.qwen3_5 import interleave_gdn_conv1d as _interleave_gdn_conv1d

    tp = mpu.get_tensor_model_parallel_world_size()
    tpr = mpu.get_tensor_model_parallel_rank()
    pp = mpu.get_pipeline_model_parallel_world_size()
    ppr = mpu.get_pipeline_model_parallel_rank()
    rank = torch.distributed.get_rank()

    cfg = json.load(open(f"{load_path}/config.json"))
    tc = cfg.get("text_config", cfg)
    layers_per_stage = tc["num_hidden_layers"] // pp

    class _NS:
        pass

    ns = _NS()
    ns.hidden_size = tc["hidden_size"]
    ns.linear_key_head_dim = tc["linear_key_head_dim"]
    ns.linear_value_head_dim = tc["linear_value_head_dim"]
    ns.linear_num_key_heads = tc["linear_num_key_heads"]
    ns.linear_num_value_heads = tc["linear_num_value_heads"]

    idx = json.load(open(f"{load_path}/model.safetensors.index.json"))["weight_map"]

    def _load_hf(name):
        fn = idx[name]
        with open(f"{load_path}/{fn}", "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            meta = hdr[name]
            off0, off1 = meta["data_offsets"]
            f.seek(8 + n + off0)
            buf = f.read(off1 - off0)
        t = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16 if meta["dtype"] == "BF16" else torch.float32)
        return t.reshape(meta["shape"]).float()

    model0 = models[0] if isinstance(models, (list, tuple)) else models
    actor = {n: p.detach().float().cpu() for n, p in model0.named_parameters() if "linear_attn." in n}
    local_layer = None
    for n in actor:
        m = re.search(r"layers\.(\d+)\.self_attention\.linear_attn\.in_proj\.weight", n)
        if m:
            local_layer = int(m.group(1))
            break
    if local_layer is None:
        print(f"[{tag} rank{rank}] 本 rank 无 GDN in_proj,跳过", flush=True)
        torch.distributed.barrier()
        if exit_after:
            sys.exit(0)
        return
    gl = ppr * layers_per_stage + local_layer  # GLOBAL 层号(修 PP 索引)
    pre = f"model.language_model.layers.{gl}.linear_attn"

    def _find(suffix):
        for n, p in actor.items():
            if n.endswith(f"layers.{local_layer}.self_attention.linear_attn.{suffix}"):
                return p
        return None

    def _diff(aw, exp):
        return (aw - exp).abs().max().item() if aw is not None and aw.shape == exp.shape else 9.9

    results = []
    # in_proj:qkvz/ba 分组融合 + TP 切分(dim=0)
    qkv = _load_hf(f"{pre}.in_proj_qkv.weight")
    z = _load_hf(f"{pre}.in_proj_z.weight")
    b = _load_hf(f"{pre}.in_proj_b.weight")
    a = _load_hf(f"{pre}.in_proj_a.weight")
    qkvz, ba = _fuse_gdn_separate_to_grouped(ns, qkv, z, b, a)
    exp = merge_gdn_linear_weights(ns, qkvz, ba, tp_size=tp).chunk(tp, 0)[tpr]
    results.append(("in_proj", _diff(_find("in_proj.weight"), exp)))
    # conv1d:TP 交织
    conv_hf = _load_hf(f"{pre}.conv1d.weight")
    exp_c = _interleave_gdn_conv1d(conv_hf, ns, tp).chunk(tp, 0)[tpr]
    results.append(("conv1d", _diff(_find("conv1d.weight"), exp_c)))
    # dt_bias / A_log:按头 chunk(dim=0)
    for leaf in ("dt_bias", "A_log"):
        exp_d = _load_hf(f"{pre}.{leaf}").chunk(tp, 0)[tpr]
        results.append((leaf, _diff(_find(leaf), exp_d)))
    # out_proj:RowParallel,partition_dim=1(沿输入 value_dim 切,与上面不同轴)
    exp_op = _load_hf(f"{pre}.out_proj.weight").chunk(tp, 1)[tpr]
    results.append(("out_proj", _diff(_find("out_proj.weight"), exp_op)))
    # norm:复制(不切)
    nm_hf = _load_hf(f"{pre}.norm.weight")
    results.append(("norm", _diff(_find("norm.weight"), nm_hf)))

    summary = " ".join(f"{nm}={'PASS' if d < 1e-2 else f'FAIL({d:.3g})'}" for nm, d in results)
    print(f"[{tag} rank{rank} tp{tpr} pp{ppr} gl{gl}] GDN LOAD: {summary}", flush=True)
    torch.distributed.barrier()
    if exit_after:
        sys.exit(0)


def _is_dir_nonempty(path):
    with os.scandir(path) as it:
        return any(it)
