import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers.activations import ACT2FN

# [Route B / longctx] TP 按头切:in_proj→ColumnParallel、out_proj→RowParallel。
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

# qwen3.6 GDN migration — mirror the verified longctx (qwen36-longctx) path:
#   GDN core  = AscendC-hybrid op in MindSpeed (== slime-ascend flash_chunk_gated_delta_rule)
#   conv1d    = Triton causal_conv1d ported from MindSpeed-MM SFT (eager F.silu(F.conv1d) fallback)
#   gated norm= fla FusedRMSNormGated if present, else torch fallback Qwen3_5MoeRMSNormGated
# Unlike longctx (which hardcodes the mindspeed imports), the two hot kernels are routed
# through vime's --qwen-gdn-backend dispatch: backend=npu returns the exact mindspeed ops
# (bit-identical to longctx) while keeping this module importable on non-NPU machines.
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
    """标记一个非 Megatron-层的参数为 TP 分片(conv/dt_bias/A_log),供 mbridge 按 partition_dim 切 + dist-ckpt sharded。"""
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
    """Gated RMSNorm — torch fallback, ported from MindSpeed-MM SFT (modeling_qwen3_5_moe.py:328).
    forward(x, gate) == rms_norm(x) * silu(gate). Used when fla's FusedRMSNormGated is not
    available (e.g. on Ascend NPU)."""

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


# Adapted from Qwen3NextGatedDeltaNet. [Route B] fused in_proj ([q|k|v|z|beta|alpha]) + single
# depthwise conv1d, TP head-split (ColumnParallel in_proj / RowParallel out_proj), CP, sharded_state_dict.
class Qwen3_5GatedDeltaNet(nn.Module):
    """
    Qwen3.5 GatedDeltaNet with varlen support — Route B (longctx) structure.
    Single fused in_proj produces [q|k|v|z|beta|alpha]; a single depthwise causal conv1d acts on
    the [q|k|v] block. TP splits heads; CP (cp_size>1, ulysses) routes through _forward_cp.
    """

    def __init__(self, config, layer_idx: int, args=None, mcore_config=None, pg_collection=None):
        super().__init__()
        # [vime] kernel dispatch via --qwen-gdn-backend (npu => mindspeed ops, == longctx).
        self.gdn_backend = getattr(args, "qwen_gdn_backend", "fla")
        self.chunk_gated_delta_rule = get_chunk_gated_delta_rule(self.gdn_backend)

        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps

        # [Route B] TP 按头切(SP=off 首版;CP 后续).mcore_config=Megatron TransformerConfig(init_method等),
        # pg_collection.tp=TP 进程组.维度仍用 hf_config(config),并行基建用 mcore_config.
        self.mcore_config = mcore_config
        self.pg_collection = pg_collection
        self.tp_size = pg_collection.tp.size() if pg_collection is not None else 1
        # [Phase3 CP] CP 组大小;cp=1 时所有 CP helper 均 no-op,forward 退化成已验 Phase1 路径。
        # cp group 鲁棒解析:优先 pg_collection.cp;缺失则回退 mpu.get_context_parallel_group()。
        _cp_group = None
        if pg_collection is not None and getattr(pg_collection, "cp", None) is not None:
            _cp_group = pg_collection.cp
        else:
            try:
                from megatron.core import mpu as _mpu

                if _mpu.get_context_parallel_world_size() > 1:
                    _cp_group = _mpu.get_context_parallel_group()
            except Exception:
                _cp_group = None
        self._cp_group = _cp_group
        self.cp_size = _cp_group.size() if _cp_group is not None else 1
        # [Phase3 CP] SP 已在 hf_attention 上游 gather 还原,GDN 内 SP 不再切 → sp_size=1。
        self.sp_size = 1
        assert self.num_v_heads % self.tp_size == 0 and self.num_k_heads % self.tp_size == 0, (
            f"TP={self.tp_size} 必须整除 num_v_heads={self.num_v_heads} 和 num_k_heads={self.num_k_heads}"
        )
        assert self.cp_size == 1 or (self.num_v_heads % (self.tp_size * self.cp_size) == 0), (
            f"CP={self.cp_size}: num_v_heads={self.num_v_heads} 须被 tp*cp={self.tp_size*self.cp_size} 整除"
        )
        self.num_v_heads_local = self.num_v_heads // self.tp_size
        self.num_k_heads_local = self.num_k_heads // self.tp_size
        self.key_dim_local = self.key_dim // self.tp_size
        self.value_dim_local = self.value_dim // self.tp_size

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_dim_local = self.conv_dim // self.tp_size
        # [Route B·照抄原生 megatron core/ssm/gated_delta_net.py] 单个 depthwise conv1d,本地 TP 维 conv_dim_local.
        # 作用于 in_proj 切出的 [q|k|v] 拼接块.depthwise=每通道独立,段边界靠 loader 交错对齐(见 mbridge converter).
        # weight 布局 [conv_dim_local, 1, k] 与 fla ShortConvolution(亦 nn.Conv1d 子类)一致 → 转换器/ckpt 兼容.
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim_local,
            out_channels=self.conv_dim_local,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim_local,
            padding=self.conv_kernel_size - 1,
        )
        _mark_tp(self.conv1d.weight, dim=0)
        # conv backend: npu+triton => mindspeed causal_conv1d (SFT-verified); 否则 None => eager F.silu(F.conv1d).
        # get_causal_conv1d 已处理 QWEN36_CAUSAL_CONV1D_IMPL=eager 与非 npu 后端 (均返回 None).
        self.causal_conv1d_fn = get_causal_conv1d(self.gdn_backend)
        self.causal_conv1d_implementation = (
            "triton" if self.causal_conv1d_fn is not None else "eager"
        )

        # [Route B·照抄原生] 单个 fused in_proj(ColumnParallel),宽 in_proj_dim = qk*2+v*2+nvh*2.
        # 合并 [q|k|v|z|beta|alpha];ColumnParallel dim0 列切,靠 loader 把权重按 TP rank 交错
        # (merge_gdn_linear_weights),使每 rank 的连续切片恰好是 [q_r|k_r|v_r|z_r|b_r|a_r].
        self.in_proj_dim = self.key_dim * 2 + self.value_dim * 2 + self.num_v_heads * 2
        self._tp_linear = self.mcore_config is not None and self.pg_collection is not None
        # [A.3 SP] GDN 是 conv+recurrence,需连续全序列,不能按序列切。开 --sequence-parallel 时,
        # 全序列由 hf_attention 的手动 gather/scatter 提供。故 GDN in_proj/out_proj 必须 sequence_parallel=False。
        if self._tp_linear:
            mcore_config_no_sp = self.mcore_config
            if getattr(self.mcore_config, "sequence_parallel", False):
                mcore_config_no_sp = copy.copy(self.mcore_config)
                mcore_config_no_sp.sequence_parallel = False
            self._mcore_config_no_sp = mcore_config_no_sp
            self.in_proj = ColumnParallelLinear(
                self.hidden_size,
                self.in_proj_dim,
                config=mcore_config_no_sp,
                init_method=mcore_config_no_sp.init_method,
                bias=False,
                gather_output=False,
                skip_bias_add=True,
                tp_group=self.pg_collection.tp,
            )
        else:  # 非 TP 回退(无 mcore_config 时)
            self.in_proj = nn.Linear(self.hidden_size, self.in_proj_dim, bias=False)

        # time step projection — 局部 TP 维 + 标 TP 分片(dim0)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads_local))
        _mark_tp(self.dt_bias, dim=0)

        A = torch.empty(self.num_v_heads_local).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        _mark_tp(self.A_log, dim=0)

        # gated norm: torch fallback on NPU (or when fla absent); fla FusedRMSNormGated on GPU.
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

        # out_proj 输入是局部 value_dim/tp(TP 分片),RowParallel all-reduce 出全 hidden.
        if self._tp_linear:
            self.out_proj = RowParallelLinear(
                self.value_dim,
                self.hidden_size,
                config=self._mcore_config_no_sp,
                init_method=self._mcore_config_no_sp.output_layer_init_method,
                bias=False,
                input_is_parallel=True,
                skip_bias_add=True,
                tp_group=self.pg_collection.tp,
            )
        else:
            self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
    ):
        # [Phase3 CP] cp>1 且 ulysses 模式 → 真 CP 前向(_forward_cp,输入 CP-scattered,内部 a2a)。
        # gather_dup 模式:hf_attention 已 all-gather 成全序列喂入 → 走下方正常 forward。cp=1:两模式都走下方。
        if self.cp_size > 1 and os.environ.get("QWEN36_CP_MODE", "ulysses") == "ulysses":
            return self._forward_cp(hidden_states, cu_seqlens)
        batch_size, seq_len, _ = hidden_states.shape

        # [Route B·照抄原生 forward split] 单个 fused in_proj → 局部 [q|k|v|z|b|a],各段 //tp.
        # ColumnParallel 返回 (out, bias) 元组;统一解包(非 TP 的 nn.Linear 直接返回张量).
        out = self.in_proj(hidden_states)
        qkvzba = out[0] if isinstance(out, tuple) else out  # [b, s, in_proj_dim_local]

        qkv, z, b, a = torch.split(
            qkvzba,
            [
                self.conv_dim_local,  # q|k|v 合块 = (key_dim*2+value_dim)//tp
                self.value_dim_local,  # z (gate)
                self.num_v_heads_local,  # beta
                self.num_v_heads_local,  # alpha
            ],
            dim=-1,
        )
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(torch.int64)

        # 单个 depthwise causal conv 作用于整个 [q|k|v] 合块(每通道独立,等价于分开做但 1× launch).
        qkv = self._conv(self.conv1d, qkv, cu_seqlens)  # [b, s, conv_dim_local]
        # conv 后再 split 成 query/key/value(各局部维)
        query, key, value = torch.split(
            qkv, [self.key_dim_local, self.key_dim_local, self.value_dim_local], dim=-1
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if self.gdn_backend == "flashqla":
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            g = g.contiguous()
            beta = beta.contiguous()

        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
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

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        out = self.out_proj(core_attn_out)
        output = out[0] if isinstance(out, tuple) else out
        return output

    def _conv(self, conv, x, cu_seqlens):
        """Depthwise causal conv on the flat [q|k|v] block (local TP dim).

        npu+triton: route through the external causal_conv1d kernel (weight [k, conv_dim_local]);
        else: eager F.silu(F.conv1d). The nn.Conv1d only holds the weight.
        """
        w = conv.weight  # [conv_dim_local, 1, k]
        if self.causal_conv1d_implementation == "triton" and self.causal_conv1d_fn is not None:
            cout, _ = self.causal_conv1d_fn(
                x=x,
                weight=w.squeeze(1).transpose(-1, -2).contiguous(),  # [k, conv_dim_local]
                bias=conv.bias,
                activation=self.activation,
                cu_seqlens=cu_seqlens,
            )
            return cout
        # eager fallback: F.silu(F.conv1d(...))
        xt = x.transpose(1, 2)
        xt = F.silu(
            F.conv1d(xt, weight=w, bias=conv.bias, padding=self.conv_kernel_size - 1, groups=w.shape[0])[
                :, :, : x.shape[1]
            ]
        )
        return xt.transpose(1, 2)

    def _conv_cp(self, x, conv_w, conv_b, cu_seqlens):
        """[Phase3 CP] CP 下的 depthwise causal conv,收**已按 CP 切过的** conv 权重
        (`get_parameter_local_cp(self.conv1d.weight, dim=0, ...)` 产出,shape [conv_dim/cp, 1, k])。
        triton 路把切过的 [conv_dim/cp,1,k] 转成 [k,conv_dim/cp] 喂进去;eager 兜底用内联 F.silu(F.conv1d)。
        x: [b, s, conv_dim/cp]  -> 返回 [b, s, conv_dim/cp]。
        """
        if self.causal_conv1d_implementation == "triton" and self.causal_conv1d_fn is not None:
            cout, _ = self.causal_conv1d_fn(
                x=x,
                weight=conv_w.squeeze(1).transpose(-1, -2).contiguous(),  # [conv_dim/cp,1,k] -> [k, conv_dim/cp]
                bias=conv_b,
                activation=self.activation,
                cu_seqlens=cu_seqlens,
            )
            return cout
        # eager fallback: F.silu(F.conv1d(...))(weight 已是切过的 [conv_dim/cp,1,k],depthwise groups=该维)
        dim = conv_w.shape[0]
        xt = x.transpose(1, 2)
        xt = F.conv1d(xt, weight=conv_w, bias=conv_b, padding=self.conv_kernel_size - 1, groups=dim)[
            :, :, : x.shape[1]
        ]
        if self.activation in ("silu", "swish"):
            xt = F.silu(xt)
        return xt.transpose(1, 2)

    def _forward_cp(self, hidden_states, cu_seqlens=None):
        """[Phase3 CP / dev_10 §5] cp_size>1 的 native-mirrored CP 前向。
        输入 [b, s_local, h](cp-scattered);内部转 seq-first 做 a2a,算完转回。
        非 packed:CP helper 整段 zigzag undo/redo;packed:undo=False + gdn_cp_utils thd 重排。
        CP helper 在 vendored gdn_cp_utils(底层走老 Megatron 现成 _all_to_all_*/undo/redo primitive)。
        """
        from .gdn_cp_utils import (
            get_parameter_local_cp,
            tensor_a2a_cp2hp,
            tensor_a2a_hp2cp,
        )

        cp = self._cp_group
        cp_size = self.cp_size
        b, s_local, _ = hidden_states.shape
        packed = cu_seqlens is not None
        if packed:
            # [dev_10 §7] packed×CP:CP helper 内部 undo 是非 packed 版(整段 reorder,会跨序列错排),
            # 故 packed 必须 a2a undo=False + 用 gdn_cp_utils 的 thd 版按序列边界重排。
            from .gdn_cp_utils import (
                redo_attention_load_balancing_thd,
                undo_attention_load_balancing_thd,
            )

            cu_seqlens = cu_seqlens.to(torch.int64)
            # ⚠️ TODO(dev_10 §8.4):thd 要求每条序列长被 2*cp 整除 → 须传 cu_seqlens_padded;此处假设已 padded。

        out = self.in_proj(hidden_states)
        qkvzba = out[0] if isinstance(out, tuple) else out  # [b, s_local, in_proj_dim_local]
        qkvzba = qkvzba.transpose(0, 1).contiguous()  # → [s_local, b, *] seq-first(a2a 要 seq_dim=0)
        qkvzba = tensor_a2a_cp2hp(  # CP→HP:Ulysses a2a(zigzag undo 见下)
            qkvzba,
            seq_dim=0,
            head_dim=-1,
            cp_group=cp,
            split_sections=[
                self.key_dim_local,
                self.key_dim_local,
                self.value_dim_local,
                self.value_dim_local,
                self.num_v_heads_local,
                self.num_v_heads_local,
            ],
            undo_attention_load_balancing=not packed,  # 非packed:整段undo;packed:↓ thd undo
        )
        if packed:
            assert qkvzba.shape[1] == 1, "packed×CP 约定 batch=1(变长拼接)"
            qkvzba = undo_attention_load_balancing_thd(qkvzba, cu_seqlens, cp_size)  # zigzag→自然序(按序列边界)
        qkvzba = qkvzba.transpose(0, 1)  # → [b, seq, *_/cp]
        seq_len = qkvzba.shape[1]
        qkv, z, b_raw, a_raw = torch.split(
            qkvzba,
            [
                self.conv_dim_local // cp_size,  # q|k|v 合块 = (key_dim*2+value_dim)/tp/cp
                self.value_dim_local // cp_size,
                self.num_v_heads_local // cp_size,
                self.num_v_heads_local // cp_size,
            ],
            dim=-1,
        )
        z = z.reshape(b, seq_len, -1, self.head_v_dim)

        qkv_split = [self.key_dim_local, self.key_dim_local, self.value_dim_local]  # conv 权重按 CP 切,3 段 [q,k,v]
        conv_w = get_parameter_local_cp(self.conv1d.weight, dim=0, cp_group=cp, split_sections=qkv_split)
        conv_b = (
            get_parameter_local_cp(self.conv1d.bias, dim=0, cp_group=cp, split_sections=qkv_split)
            if self.conv1d.bias is not None
            else None
        )
        qkv = self._conv_cp(qkv, conv_w, conv_b, cu_seqlens)  # [b, s, conv_dim/cp]

        query, key, value = torch.split(
            qkv,
            [self.key_dim_local // cp_size, self.key_dim_local // cp_size, self.value_dim_local // cp_size],
            dim=-1,
        )
        query = query.reshape(b, seq_len, -1, self.head_k_dim)
        key = key.reshape(b, seq_len, -1, self.head_k_dim)
        value = value.reshape(b, seq_len, -1, self.head_v_dim)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()

        A_log_cp = get_parameter_local_cp(self.A_log, dim=0, cp_group=cp)
        dt_bias_cp = get_parameter_local_cp(self.dt_bias, dim=0, cp_group=cp)
        beta = b_raw.contiguous().sigmoid()
        g = -A_log_cp.float().exp() * F.softplus(a_raw.float().contiguous() + dt_bias_cp)

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
        )  # packed:全局自然序 cu_seqlens;非packed:None

        z_shape = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        core_attn_out = self.norm(core_attn_out, z.reshape(-1, z.shape[-1]))
        core_attn_out = core_attn_out.reshape(z_shape).reshape(b, seq_len, -1)
        core_attn_out = core_attn_out.transpose(0, 1).contiguous()  # → [seq, b, v_dim/cp]
        if packed:
            core_attn_out = redo_attention_load_balancing_thd(core_attn_out, cu_seqlens, cp_size)  # 自然序→zigzag
        core_attn_out = tensor_a2a_hp2cp(  # HP→CP → [s_local, b, value_dim_local]
            core_attn_out,
            seq_dim=0,
            head_dim=-1,
            cp_group=cp,
            redo_attention_load_balancing=not packed,
        )  # 非packed:整段redo;packed:↑ thd redo
        core_attn_out = core_attn_out.transpose(0, 1)  # → [b, s_local, value_dim_local]
        out = self.out_proj(core_attn_out)
        return out[0] if isinstance(out, tuple) else out

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, tp_group=None):
        """[Route B·照抄原生 gated_delta_net.py:428-514] fused in_proj + 单 conv1d 的 dist-ckpt 切分声明。
        dt_bias/A_log axis0;in_proj(ColumnParallel)切后再 _split_tensor_factory 拆 6 段
        [query,key,value,z,beta,alpha];conv1d 拆 3 段 [query,key,value]——保证跨 TP-size resharding 各段独立正确。"""
        from megatron.core.ssm.gated_delta_net import _split_tensor_factory
        from megatron.core.transformer.utils import (
            ensure_metadata_has_dp_cp_group,
            make_sharded_tensors_for_checkpoint,
            sharded_state_dict_default,
        )

        metadata = ensure_metadata_has_dp_cp_group(metadata)
        tp_group = tp_group if tp_group is not None else self.pg_collection.tp
        sharded_sd = {}
        # 直接参数:dt_bias / A_log → TP axis 0
        self._save_to_state_dict(sharded_sd, "", keep_vars=True)
        sharded_sd = make_sharded_tensors_for_checkpoint(
            sharded_sd,
            prefix,
            tensor_parallel_layers_axis_map={"A_log": 0, "dt_bias": 0},
            sharded_offsets=sharded_offsets,
            tp_group=tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )
        # 子模块:conv1d weight axis0;in_proj/out_proj/norm 走各自 default(Column=0/Row=1)
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

        # in_proj.weight 再拆 6 段(每段独立 ShardedTensor,跨 TP-size resharding 正确)
        in_proj_dim_local = self.in_proj_dim // self.tp_size
        assert sharded_sd[f"{prefix}in_proj.weight"].data.size(0) == in_proj_dim_local
        sharded_sd[f"{prefix}in_proj.weight"] = _split_tensor_factory(
            sharded_sd[f"{prefix}in_proj.weight"],
            [
                self.key_dim // self.tp_size,
                self.key_dim // self.tp_size,
                self.value_dim // self.tp_size,
                self.value_dim // self.tp_size,
                self.num_v_heads // self.tp_size,
                self.num_v_heads // self.tp_size,
            ],
            ["query", "key", "value", "z", "beta", "alpha"],
            0,
        )
        # conv1d.weight 再拆 3 段
        assert sharded_sd[f"{prefix}conv1d.weight"].data.size(0) == self.conv_dim_local
        sharded_sd[f"{prefix}conv1d.weight"] = _split_tensor_factory(
            sharded_sd[f"{prefix}conv1d.weight"],
            [self.key_dim // self.tp_size, self.key_dim // self.tp_size, self.value_dim // self.tp_size],
            ["query", "key", "value"],
            0,
        )
        return sharded_sd


class Attention(HuggingfaceAttention):
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

        # [Route B] 把 Megatron config(self.config,带 init_method/sequence_parallel)+ pg_collection
        # 下传给 GDN,使 in_proj/out_proj 走 TP 按头切;args 提供 --qwen-gdn-backend.
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
            # [fix 2026-08-20] 环境无 NVIDIA transformer_engine 时,Megatron extensions 把
            # core_attention 兜底成 MagicMock,deepcopy 会炸("cannot pickle 'cell' object")。
            # 这些 GDN 层的 self_attention 随后整体被替换(下方 ModuleSpec(module=Attention)),
            # 原 core_attention 不会被使用 —— deepcopy 前换成可用的本地实现即可。
            _sa = transformer_layer_spec.layer_specs[layer_id].submodules.self_attention
            _ca = getattr(_sa.submodules, "core_attention", None)
            if _ca is not None and "Mock" in type(_ca).__name__:
                from megatron.core.transformer.dot_product_attention import DotProductAttention

                _sa.submodules.core_attention = DotProductAttention
            layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
            layer_specs.submodules.self_attention = ModuleSpec(
                module=Attention,
                params={"args": args},
            )
            transformer_layer_spec.layer_specs[layer_id] = layer_specs
    return transformer_layer_spec
