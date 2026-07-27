import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers.activations import ACT2FN

# _split_tensor_factory is private Megatron API; no public equivalent exists.
from megatron.core.ssm.gated_delta_net import _split_tensor_factory
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)

# The GDN kernel and the causal conv are selected through --qwen-gdn-backend so this module
# stays importable on machines without the Ascend kernels.
from .gdn_cp_utils import (
    get_parameter_local_cp,
    redo_attention_load_balancing_thd,
    tensor_a2a_cp2hp,
    tensor_a2a_hp2cp,
    undo_attention_load_balancing_thd,
)
from .hf_attention import HuggingfaceAttention, _load_hf_config
from .qwen_gdn_backend import get_causal_conv1d, get_chunk_gated_delta_rule

try:
    from fla.modules import FusedRMSNormGated
except ImportError:
    FusedRMSNormGated = None

try:
    import torch_npu as _torch_npu

    _IS_NPU_AVAILABLE = True
except ImportError:
    _torch_npu = None
    _IS_NPU_AVAILABLE = False


def _mark_tp(t, dim=0):
    """Mark a plain parameter as TP-sharded so the loader and dist-checkpoint split it."""
    setattr(t, "tensor_model_parallel", True)
    setattr(t, "partition_dim", dim)
    setattr(t, "partition_stride", 1)
    return t


def _get_text_config(hf_config):
    """Extract text config from a VLM config if needed."""
    if hasattr(hf_config, "text_config"):
        return hf_config.text_config
    return hf_config


class Qwen3_5MoeRMSNormGated(nn.Module):
    """Gated RMSNorm: rms_norm(x) * silu(gate). Fallback for when fla is unavailable."""

    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        if _IS_NPU_AVAILABLE:
            hidden_states = _torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
            hidden_states = hidden_states * F.silu(gate)
        else:
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            hidden_states = self.weight * hidden_states.to(input_dtype)
            hidden_states = hidden_states * F.silu(gate.to(torch.float32)).to(input_dtype)
        return hidden_states


# Adapted from Qwen3NextGatedDeltaNet: one fused in_proj ([q|k|v|z|beta|alpha]) and one depthwise
# conv1d, split over TP by head, with CP and dist-checkpoint support.
class Qwen3_5GatedDeltaNet(nn.Module):
    """
    Qwen3.5 GatedDeltaNet with varlen support.
    Single fused in_proj produces [q|k|v|z|beta|alpha]; a single depthwise causal conv1d acts on
    the [q|k|v] block. TP splits heads; CP routes through _forward_cp.
    """

    def __init__(self, config, layer_idx: int, args=None, mcore_config=None, pg_collection=None):
        super().__init__()
        self.layer_idx = layer_idx
        self._init_dims(config)
        proj_config = self._init_parallel(mcore_config, pg_collection)
        self._init_kernels(args)
        self._build_projections(config, proj_config)

    def _init_dims(self, config):
        """Head and projection dimensions, all taken from the HF config."""
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.in_proj_dim = self.key_dim * 2 + self.value_dim * 2 + self.num_v_heads * 2

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps

    def _init_parallel(self, mcore_config, pg_collection):
        """Resolve the TP and CP groups, derive the per-rank dimensions and return the
        Megatron config the projections must be built with."""
        self.pg_collection = pg_collection
        self.tp_size = pg_collection.tp.size() if pg_collection is not None else 1

        cp_group = None
        if pg_collection is not None and getattr(pg_collection, "cp", None) is not None:
            cp_group = pg_collection.cp
        elif mpu.is_initialized() and mpu.get_context_parallel_world_size() > 1:
            cp_group = mpu.get_context_parallel_group()
        self._cp_group = cp_group
        self.cp_size = cp_group.size() if cp_group is not None else 1

        assert self.num_v_heads % self.tp_size == 0 and self.num_k_heads % self.tp_size == 0, (
            f"TP={self.tp_size} must divide num_v_heads={self.num_v_heads} "
            f"and num_k_heads={self.num_k_heads}"
        )
        assert self.cp_size == 1 or self.num_v_heads % (self.tp_size * self.cp_size) == 0, (
            f"num_v_heads={self.num_v_heads} must be divisible by "
            f"tp*cp={self.tp_size * self.cp_size}"
        )

        self.num_v_heads_local = self.num_v_heads // self.tp_size
        self.num_k_heads_local = self.num_k_heads // self.tp_size
        self.key_dim_local = self.key_dim // self.tp_size
        self.value_dim_local = self.value_dim // self.tp_size
        self.conv_dim_local = self.conv_dim // self.tp_size

        # The conv and the recurrence need the whole sequence, so the projections below must not
        # be sequence-parallel; the caller gathers and re-scatters around this module instead.
        if mcore_config is not None and getattr(mcore_config, "sequence_parallel", False):
            mcore_config = copy.copy(mcore_config)
            mcore_config.sequence_parallel = False
        self._tp_linear = mcore_config is not None and pg_collection is not None
        return mcore_config

    def _init_kernels(self, args):
        """Select the GDN and causal-conv kernels for the configured backend."""
        self.gdn_backend = getattr(args, "qwen_gdn_backend", "fla")
        self.chunk_gated_delta_rule = get_chunk_gated_delta_rule(self.gdn_backend)
        # None means no external kernel is available; _causal_conv then falls back to eager conv1d.
        self.causal_conv1d_fn = get_causal_conv1d(
            self.gdn_backend, getattr(args, "qwen_gdn_conv1d_impl", "triton")
        )

    def _build_projections(self, config, proj_config):
        # One depthwise conv over the concatenated [q|k|v] block. Segment boundaries line up
        # because the loader interleaves the weight per TP rank (see the mbridge converter).
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim_local,
            out_channels=self.conv_dim_local,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim_local,
            padding=self.conv_kernel_size - 1,
        )
        _mark_tp(self.conv1d.weight, dim=0)

        # ColumnParallelLinear splits dim 0; the loader interleaves the weight per TP rank
        # (merge_gdn_linear_weights) so each rank's contiguous slice is [q|k|v|z|beta|alpha].
        if self._tp_linear:
            self.in_proj = ColumnParallelLinear(
                self.hidden_size,
                self.in_proj_dim,
                config=proj_config,
                init_method=proj_config.init_method,
                bias=False,
                gather_output=False,
                skip_bias_add=True,
                tp_group=self.pg_collection.tp,
            )
        else:  # no Megatron config: fall back to a plain Linear
            self.in_proj = nn.Linear(self.hidden_size, self.in_proj_dim, bias=False)

        # time step projection, sharded over TP like the heads
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads_local))
        _mark_tp(self.dt_bias, dim=0)

        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads_local).uniform_(0, 16)))
        _mark_tp(self.A_log, dim=0)

        if self.gdn_backend == "npu" or FusedRMSNormGated is None:
            self.norm = Qwen3_5MoeRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        else:
            self.norm = FusedRMSNormGated(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                activation=self.activation,
                device=torch.cuda.current_device(),
                dtype=config.dtype if config.dtype is not None else torch.get_default_dtype(),
            )

        # out_proj takes the TP-local value dim and all-reduces back to the full hidden size
        if self._tp_linear:
            self.out_proj = RowParallelLinear(
                self.value_dim,
                self.hidden_size,
                config=proj_config,
                init_method=proj_config.output_layer_init_method,
                bias=False,
                input_is_parallel=True,
                skip_bias_add=True,
                tp_group=self.pg_collection.tp,
            )
        else:
            self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    @property
    def in_proj_sections(self):
        """Widths of the fused in_proj segments [q, k, v, z, beta, alpha] on the local TP shard."""
        return [
            self.key_dim_local,
            self.key_dim_local,
            self.value_dim_local,
            self.value_dim_local,
            self.num_v_heads_local,
            self.num_v_heads_local,
        ]

    @property
    def conv_sections(self):
        """Widths of the conv1d segments [q, k, v] on the local TP shard."""
        return [self.key_dim_local, self.key_dim_local, self.value_dim_local]

    def _apply_in_proj(self, hidden_states):
        # ColumnParallelLinear returns (out, bias); the non-TP nn.Linear fallback returns a tensor.
        out = self.in_proj(hidden_states)
        return out[0] if isinstance(out, tuple) else out

    def _apply_out_proj(self, core_attn_out):
        out = self.out_proj(core_attn_out)
        return out[0] if isinstance(out, tuple) else out

    def _causal_conv(self, x, weight, bias, cu_seqlens):
        """Depthwise causal conv over the fused [q|k|v] block.

        `weight` is [channels, 1, k]; the external kernel wants it transposed to [k, channels].
        """
        if self.causal_conv1d_fn is not None:
            out, _ = self.causal_conv1d_fn(
                x=x,
                weight=weight.squeeze(1).transpose(-1, -2).contiguous(),
                bias=bias,
                activation=self.activation,
                cu_seqlens=cu_seqlens,
            )
            return out
        out = F.conv1d(
            x.transpose(1, 2),
            weight=weight,
            bias=bias,
            padding=self.conv_kernel_size - 1,
            groups=weight.shape[0],
        )[:, :, : x.shape[1]]
        return self.act(out).transpose(1, 2)

    def _gated_delta_attention(self, qkvzba, sections, conv_weight, conv_bias, a_log, dt_bias, cu_seqlens):
        """Shared core: split the fused projection, conv the [q|k|v] block, run the gated
        delta rule and apply the gated norm.

        `sections` are the segment widths of `qkvzba` (TP-local, and CP-local under CP).
        The parameters are passed in because the CP path feeds CP-sliced copies of them.
        """
        batch_size, seq_len, _ = qkvzba.shape
        qkv, z, beta_raw, alpha_raw = torch.split(qkvzba, [sum(sections[:3]), *sections[3:]], dim=-1)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        qkv = self._causal_conv(qkv, conv_weight, conv_bias, cu_seqlens)
        query, key, value = torch.split(qkv, sections[:3], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        repeat = self.num_v_heads // self.num_k_heads
        if repeat > 1:
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        beta = beta_raw.sigmoid()
        # Without the .float() A can become -inf when the model is loaded in fp16.
        g = -a_log.float().exp() * F.softplus(alpha_raw.float() + dt_bias)

        # The kernels require contiguous inputs; .contiguous() is a no-op when already so.
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        g, beta = g.contiguous(), beta.contiguous()

        core_attn_out, _ = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )

        z_shape = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        core_attn_out = self.norm(core_attn_out, z.reshape(-1, z.shape[-1]))
        return core_attn_out.reshape(z_shape).reshape(batch_size, seq_len, -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
    ):
        # Under CP the input stays CP-scattered and the all-to-all happens in _forward_cp.
        if self.cp_size > 1:
            return self._forward_cp(hidden_states, cu_seqlens)
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(torch.int64)

        core_attn_out = self._gated_delta_attention(
            self._apply_in_proj(hidden_states),
            self.in_proj_sections,
            self.conv1d.weight,
            self.conv1d.bias,
            self.A_log,
            self.dt_bias,
            cu_seqlens,
        )
        return self._apply_out_proj(core_attn_out)

    def _forward_cp(self, hidden_states, cu_seqlens=None):
        """Ulysses-style CP forward. Input is CP-scattered [b, s_local, h]; the fused
        projection is exchanged to head-parallel, the core runs on the full sequence,
        and the output is exchanged back.

        The all-to-all helpers only know how to undo the zigzag load balancing for a whole
        segment, which would reorder across sequence boundaries when packed, so the packed
        path disables that and reorders per sequence instead.
        """
        cp = self._cp_group
        cp_size = self.cp_size
        packed = cu_seqlens is not None
        if packed:
            cu_seqlens = cu_seqlens.to(torch.int64)

        # seq-first, because the all-to-all expects seq_dim=0
        qkvzba = self._apply_in_proj(hidden_states).transpose(0, 1).contiguous()
        qkvzba = tensor_a2a_cp2hp(
            qkvzba,
            seq_dim=0,
            head_dim=-1,
            cp_group=cp,
            split_sections=self.in_proj_sections,
            undo_attention_load_balancing=not packed,
        )
        if packed:
            assert qkvzba.shape[1] == 1, "packed CP expects batch=1 (variable-length packing)"
            qkvzba = undo_attention_load_balancing_thd(qkvzba, cu_seqlens, cp_size)
        qkvzba = qkvzba.transpose(0, 1)

        conv_weight = get_parameter_local_cp(
            self.conv1d.weight, dim=0, cp_group=cp, split_sections=self.conv_sections
        )
        conv_bias = (
            get_parameter_local_cp(self.conv1d.bias, dim=0, cp_group=cp, split_sections=self.conv_sections)
            if self.conv1d.bias is not None
            else None
        )
        core_attn_out = self._gated_delta_attention(
            qkvzba,
            [s // cp_size for s in self.in_proj_sections],
            conv_weight,
            conv_bias,
            get_parameter_local_cp(self.A_log, dim=0, cp_group=cp),
            get_parameter_local_cp(self.dt_bias, dim=0, cp_group=cp),
            cu_seqlens,
        )

        core_attn_out = core_attn_out.transpose(0, 1).contiguous()
        if packed:
            core_attn_out = redo_attention_load_balancing_thd(core_attn_out, cu_seqlens, cp_size)
        core_attn_out = tensor_a2a_hp2cp(
            core_attn_out,
            seq_dim=0,
            head_dim=-1,
            cp_group=cp,
            redo_attention_load_balancing=not packed,
        )
        return self._apply_out_proj(core_attn_out.transpose(0, 1))

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, tp_group=None):
        """Distributed-checkpoint sharding for the fused in_proj and the conv1d."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        tp_group = tp_group if tp_group is not None else self.pg_collection.tp
        sharded_sd = {}
        # own parameters: dt_bias and A_log are sharded on axis 0
        self._save_to_state_dict(sharded_sd, "", keep_vars=True)
        sharded_sd = make_sharded_tensors_for_checkpoint(
            sharded_sd,
            prefix,
            tensor_parallel_layers_axis_map={"A_log": 0, "dt_bias": 0},
            sharded_offsets=sharded_offsets,
            tp_group=tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )
        # submodules: conv1d is sharded on axis 0, the rest use their own defaults
        for name, module in self.named_children():
            if name == "conv1d":
                module_sd = module.state_dict(prefix="", keep_vars=True)
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd,
                    f"{prefix}{name}.",
                    {"weight": 0},
                    sharded_offsets,
                    tp_group=tp_group,
                    dp_cp_group=metadata["dp_cp_group"],
                )
            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata, tp_group=tp_group
                )
            sharded_sd.update(module_sharded_sd)

        # Declare each fused segment as its own ShardedTensor so resharding across a different
        # TP size stays correct per segment.
        assert sharded_sd[f"{prefix}in_proj.weight"].data.size(0) == sum(self.in_proj_sections)
        sharded_sd[f"{prefix}in_proj.weight"] = _split_tensor_factory(
            sharded_sd[f"{prefix}in_proj.weight"],
            self.in_proj_sections,
            ["query", "key", "value", "z", "beta", "alpha"],
            0,
        )
        assert sharded_sd[f"{prefix}conv1d.weight"].data.size(0) == sum(self.conv_sections)
        sharded_sd[f"{prefix}conv1d.weight"] = _split_tensor_factory(
            sharded_sd[f"{prefix}conv1d.weight"],
            self.conv_sections,
            ["query", "key", "value"],
            0,
        )
        return sharded_sd


class Attention(HuggingfaceAttention):
    # The gated delta net exchanges the sequence itself (see Qwen3_5GatedDeltaNet._forward_cp),
    # so the base class must not gather it.
    gathers_cp_in_base = False

    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        pg_collection=None,
    ):
        super().__init__(
            args,
            config,
            layer_number,
            cp_comm_type,
            pg_collection,
        )
        # Qwen3.5 is a VLM model with nested text_config
        self.hf_config = _get_text_config(self.hf_config)
        self.hf_config._attn_implementation = "flash_attention_2"

        # Pass the Megatron config and process groups down to the linear-attention module
        self.pg_collection = pg_collection
        self.linear_attn = Qwen3_5GatedDeltaNet(
            self.hf_config,
            self.hf_layer_idx,
            args=args,
            mcore_config=self.config,
            pg_collection=pg_collection,
        )

        # Use a simple RMSNorm
        try:
            from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm

            self.input_layernorm = Qwen3NextRMSNorm(self.hf_config.hidden_size, eps=self.hf_config.rms_norm_eps)
        except ImportError:
            from torch.nn import RMSNorm

            self.input_layernorm = RMSNorm(self.hf_config.hidden_size, eps=self.hf_config.rms_norm_eps)

    def hf_forward(self, hidden_states, packed_seq_params):
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            # GDN needs the padded leading-0 cu_seqlens; cu_seqlens_q is repointed to the
            # MindSpeed ring convention (no leading 0) under CP>1, so prefer the stashed copy.
            cu_seqlens=getattr(packed_seq_params, "cu_seqlens_gdn", packed_seq_params.cu_seqlens_q),
        )
        return hidden_states


def get_qwen3_5_spec(args, config, vp_stage):
    # always use the moe path for MoE models
    if not args.num_experts:
        config.moe_layer_freq = [0] * config.num_layers

    # Define the decoder block spec
    kwargs = {
        "use_transformer_engine": True,
    }
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)

    assert config.pipeline_model_parallel_layout is None, "not support this at the moment"

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

    hf_config = _load_hf_config(args.hf_checkpoint)
    text_config = _get_text_config(hf_config)

    # Compute layer_types if the config class doesn't expose it
    if not hasattr(text_config, "layer_types"):
        interval = getattr(text_config, "full_attention_interval", 4)
        n = text_config.num_hidden_layers
        text_config.layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n)
        ]

    for layer_id in range(num_layers_to_build):
        if text_config.layer_types[layer_id + offset] == "linear_attention":
            layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
            layer_specs.submodules.self_attention = ModuleSpec(
                module=Attention,
                params={"args": args},
            )
            transformer_layer_spec.layer_specs[layer_id] = layer_specs
    return transformer_layer_spec
