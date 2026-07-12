import gc
import json
import os
import shutil

import torch
import torch.distributed as dist

# [NPU] Register torch.npu + transfer_to_npu BEFORE any megatron import. The GDN-patched
# megatron (megatron.core.ssm.gated_delta_net) imports mindspeed triton ops at module-load
# time, which dereference torch.npu. Mirrors slime-ascend's convert + vime train.py ordering.
try:
    import torch_npu  # noqa: F401
    import mindspeed.megatron_adaptor  # noqa: F401
except ImportError:
    pass

from megatron.core.enums import ModelType
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, save_checkpoint
from megatron.training.training import get_model

import vime_plugins.mbridge  # noqa: F401
import vime_plugins.mbridge.qwen3_5  # noqa: F401
from mbridge import AutoBridge
from vime.backends.megatron_utils.arguments import set_default_megatron_args
from vime.backends.megatron_utils.initialize import init
from vime.backends.megatron_utils.model_provider import get_model_provider_func
from vime.utils.logging_utils import configure_logger
from vime.utils.memory_utils import print_memory
from vime.utils.common import is_npu


def add_convertion_args(parser):
    """Add conversion arguments to the parser"""
    parser.add_argument("--hf-checkpoint", type=str, required=True, help="HuggingFace model path")
    parser.add_argument(
        "--megatron-to-hf-mode",
        choices=["raw", "bridge"],
        default="raw",
        help="The method to convert megatron weights to hugging face weights for vLLM.",
    )
    # [NPU/GDN] mirror vime's full-parser arg; the model reads getattr(args,"qwen_gdn_backend","fla").
    # On NPU this must be "npu" so get_chunk_gated_delta_rule/get_causal_conv1d route to mindspeed ops
    # instead of fla (which is not installed on Ascend).
    parser.add_argument(
        "--qwen-gdn-backend",
        type=str,
        choices=["fla", "flashqla", "npu"],
        default="fla",
        help="GDN implementation backend for Qwen linear-attention layers.",
    )
    try:
        parser.add_argument("--padded-vocab-size", type=int, default=None)
    except Exception:
        pass
    parser.add_argument(
        "--no-moe-permute-fusion",
        action="store_true",
        help="Accept vime model arg bundles that explicitly disable MoE permute fusion during conversion.",
    )
    return parser


def _get_text_config_dict(hf_checkpoint: str) -> dict:
    config_path = os.path.join(hf_checkpoint, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    return config.get("text_config", config)


def _validate_qwen36_gdn_args(args) -> None:
    text_config = _get_text_config_dict(args.hf_checkpoint)
    required_keys = {
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
    }
    if not required_keys.issubset(text_config):
        return

    expected = {
        "linear_conv_kernel_dim": text_config["linear_conv_kernel_dim"],
        "linear_key_head_dim": text_config["linear_key_head_dim"],
        "linear_value_head_dim": text_config["linear_value_head_dim"],
        "linear_num_key_heads": text_config["linear_num_key_heads"],
        "linear_num_value_heads": text_config["linear_num_value_heads"],
        "mtp_num_layers": text_config.get("mtp_num_hidden_layers", 0),
    }
    actual = {
        "linear_conv_kernel_dim": args.linear_conv_kernel_dim,
        "linear_key_head_dim": args.linear_key_head_dim,
        "linear_value_head_dim": args.linear_value_head_dim,
        "linear_num_key_heads": args.linear_num_key_heads,
        "linear_num_value_heads": args.linear_num_value_heads,
        "mtp_num_layers": args.mtp_num_layers or 0,
    }
    mismatches = [
        f"{name}: expected {expected[name]}, got {actual[name]}"
        for name in expected
        if actual[name] != expected[name]
    ]
    if mismatches:
        raise ValueError(
            "Qwen3.6 config mismatch between HF checkpoint and Megatron args:\n- "
            + "\n- ".join(mismatches)
        )

    layer_types = text_config.get("layer_types")
    interval = text_config.get("full_attention_interval")
    if layer_types is None and interval != 4:
        raise ValueError(
            f"Expected full_attention_interval=4 when layer_types is absent, got {interval!r}"
        )


def get_args():
    args = parse_args(add_convertion_args)
    args = set_default_megatron_args(args)

    text_config = _get_text_config_dict(args.hf_checkpoint)
    if getattr(args, "mtp_num_layers", None) is None and text_config.get("mtp_num_hidden_layers") is not None:
        args.mtp_num_layers = text_config["mtp_num_hidden_layers"]

    # set to pass megatron validate_args
    args.save_interval = 1
    args.micro_batch_size = 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.global_batch_size = int(os.environ.get("WORLD_SIZE", "1"))

    assert world_size <= args.num_layers, (
        f"World size {world_size} must be less than or equal to number of layers {args.num_layers}. "
        "You are using too many GPUs for this conversion."
    )

    def ceildiv(a, b):
        return -(a // -b)

    if args.pipeline_model_parallel_size == 1 and world_size > 1:
        pp_size = world_size
        while True:
            args.pipeline_model_parallel_size = pp_size
            args.decoder_last_pipeline_num_layers = args.num_layers - ceildiv(
                args.num_layers, args.pipeline_model_parallel_size
            ) * (args.pipeline_model_parallel_size - 1)

            if args.decoder_last_pipeline_num_layers > 0:
                break

            if pp_size % 2 == 0:
                pp_size //= 2
            else:
                raise ValueError(
                    f"Cannot find a valid pipeline model parallel size for {args.num_layers} layers and {world_size} GPUs."
                )
    print(
        f"Using pipeline model parallel size: {args.pipeline_model_parallel_size}, decoder last pipeline num layers: {args.decoder_last_pipeline_num_layers}"
    )

    validate_args(args)
    _validate_qwen36_gdn_args(args)
    return args


def main():
    configure_logger()

    # Initialize distributed environment
    world_size = int(os.getenv("WORLD_SIZE") or os.getenv("SLURM_NTASKS") or 1)
    local_rank = int(os.getenv("LOCAL_RANK") or os.getenv("SLURM_LOCALID") or 0)
    global_rank = int(os.getenv("RANK") or os.getenv("SLURM_PROCID") or 0)

    torch.cuda.set_device(local_rank)
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(global_rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    if is_npu():
        # [NPU] hccl must NOT receive device_id: transfer_to_npu maps cuda:n->npu:n, and a
        # npu device_id on the default PG makes gloo sub-group creation in
        # mpu.initialize_model_parallel hit `_get_backend(npu).supports_splitting`
        # -> "No backend type associated with device type npu". Mirrors slime-ascend convert.
        dist.init_process_group(
            backend="hccl",
            world_size=world_size,
            rank=global_rank,
        )
    else:
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=global_rank,
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    args = get_args()
    init(args)

    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)

    # Load model
    hf_model_path = args.hf_checkpoint
    bridge = AutoBridge.from_pretrained(hf_model_path, trust_remote_code=True)
    bridge.load_weights(model, hf_model_path, memory_efficient=True)
    print(f"Model loaded: {hf_model_path}")

    if args.use_cpu_initialization:
        model[0] = model[0].cpu()

    print_memory("after loading model")
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    save_checkpoint(1, model, None, None, 0)

    if dist.get_rank() == 0:
        # change to release ckpt
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, "w") as f:
            f.write("release")
        source_dir = get_checkpoint_name(args.save, 1, False, return_base_dir=True)
        target_dir = get_checkpoint_name(args.save, -1, True, return_base_dir=True)
        shutil.move(source_dir, target_dir)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
