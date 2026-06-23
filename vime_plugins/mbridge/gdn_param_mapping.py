"""[Route B / longctx mbridge 依赖] GDN 参数映射工具(单一可信源 single source of truth)。

vime 的 qwen3_5 mbridge(_weight_to_mcore_format / _weight_to_hf_format)与模型
sharded_state_dict 共用这里的 6 个 helper。分两类:

A) in_proj 融合/拆分(4 个,逐字 vendored from Megatron-Bridge,零改动):
  - merge_gdn_linear_weights:      head-grouped (qkvz,ba) → megatron fused in_proj(TP 交错)
  - split_gdn_linear_weights:      megatron fused in_proj → head-grouped (qkvz,ba)(去交错)
  - _fuse_gdn_separate_to_grouped: HF 4 分离张量(qkv,z,b,a)→ (qkvz,ba) 中间格式
  - _split_gdn_grouped_to_separate:(qkvz,ba) → HF 4 分离张量
  这 4 个在新 Megatron-Bridge(megatron.bridge.models.conversion.param_mapping)里,但导入该
  模块会传递性拉入 transformer_engine(本 NPU 环境未装)→ 崩溃。故 vendor 进来,使 mbridge 自给自足。
  原始来源:Megatron-Bridge-slime/src/megatron/bridge/models/conversion/param_mapping.py L2450-L2603。

B) conv1d 段交错/去交错(2 个,vime 自研,非 vendored):
  - interleave_gdn_conv1d:   HF conv1d[conv_dim,1,k]([q|k|v] 平铺)→ TP 交错全局张量
  - deinterleave_gdn_conv1d: 上者之逆(mcore→HF)
  这 2 个与 merge_gdn_linear_weights 的 qkv 段交错方式严格对齐(同一 TP rank 上 conv 通道与
  fused in_proj 的 q/k/v 头一一对应)。它们在 Megatron-Bridge 里没有对应物 —— 不要拿去 diff 上游。

全部 6 个均为纯 torch(只读 config 的 GDN 维度),已用真实 Qwen3.6 维度做 merge↔split、
conv interleave↔deinterleave 往返 + 逐 rank chunk 一致性数值校验(tp=1/2/4 全 exact)。
"""
from typing import Tuple

import torch


def merge_gdn_linear_weights(
    provider,  # TransformerConfig,但此处仅读属性,不强依赖类型
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    tp_size: int = 1,
) -> torch.Tensor:
    """Merge GDN linear weights into in_proj.

    将 head-grouped 中间格式(qkvz,ba)按 TP 交错成 megatron fused in_proj 全局张量,
    使后续 mbridge chunk(tp, dim0) 正好落每 rank [q_r|k_r|v_r|z_r|b_r|a_r]。

    Args:
        provider: 配置对象(需属性 hidden_size/linear_key_head_dim/linear_value_head_dim/
                  linear_num_key_heads/linear_num_value_heads)。
        qkvz: [num_qk_heads * (qk_dim*2+v_dim*2), hidden] grouped。
        ba: [num_qk_heads * (v_per_group*2), hidden] grouped。
        tp_size: TP 并行度。

    Returns:
        in_proj: [in_proj_dim, hidden] 全局张量(后续 mbridge 按 dim0 chunk(tp))。
    """
    assert tp_size >= 1, f"tp_size must be greater than 0, but got {tp_size=}"

    hidden_size = provider.hidden_size
    qk_head_dim = provider.linear_key_head_dim
    v_head_dim = provider.linear_value_head_dim
    num_qk_heads = provider.linear_num_key_heads
    num_v_heads = provider.linear_num_value_heads
    qk_dim = qk_head_dim * num_qk_heads
    v_dim = v_head_dim * num_v_heads

    # Reshape to expose head dimension
    qkvz_reshaped = qkvz.reshape(num_qk_heads, (qk_dim * 2 + v_dim * 2) // num_qk_heads, hidden_size)
    ba_reshaped = ba.reshape(num_qk_heads, 2 * num_v_heads // num_qk_heads, hidden_size)
    q, k, v, z = torch.split(
        qkvz_reshaped,
        [
            qk_head_dim,
            qk_head_dim,
            num_v_heads // num_qk_heads * v_head_dim,
            num_v_heads // num_qk_heads * v_head_dim,
        ],
        dim=1,
    )
    b, a = torch.split(
        ba_reshaped,
        [
            num_v_heads // num_qk_heads,
            num_v_heads // num_qk_heads,
        ],
        dim=1,
    )

    q, k, v, z, b, a = [weight.reshape(tp_size, -1, hidden_size) for weight in [q, k, v, z, b, a]]
    in_proj = torch.cat([q, k, v, z, b, a], dim=1)
    in_proj = in_proj.reshape(-1, hidden_size)

    assert in_proj.numel() == qkvz.numel() + ba.numel(), (
        f"QKVZBA weights are not correctly merged, {qkvz.numel()=}, {ba.numel()=}, {in_proj.numel()=}, {tp_size=}"
    )

    return in_proj


def split_gdn_linear_weights(provider, in_proj: torch.Tensor, tp_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split GDN linear weights into QKVZ and BA.

    merge_gdn_linear_weights 的逆:megatron fused in_proj(全局,gather 后)→ (qkvz,ba) grouped。

    Args:
        provider: 配置对象。
        in_proj: [in_proj_dim, hidden] 全局 fused。
        tp_size: TP 并行度。

    Returns:
        (qkvz, ba): head-grouped 中间格式。
    """
    assert tp_size >= 1, f"tp_size must be greater than 0, but got {tp_size=}"

    hidden_size = provider.hidden_size
    qk_head_dim = provider.linear_key_head_dim
    v_head_dim = provider.linear_value_head_dim
    num_qk_heads = provider.linear_num_key_heads
    num_qk_heads_local_tp = provider.linear_num_key_heads // tp_size
    num_v_heads_local_tp = provider.linear_num_value_heads // tp_size
    qk_dim_local_tp = qk_head_dim * num_qk_heads_local_tp
    v_dim_local_tp = v_head_dim * num_v_heads_local_tp

    in_proj = in_proj.reshape(tp_size, -1, hidden_size)
    q, k, v, z, b, a = torch.split(
        in_proj,
        [
            qk_dim_local_tp,
            qk_dim_local_tp,
            v_dim_local_tp,
            v_dim_local_tp,
            num_v_heads_local_tp,
            num_v_heads_local_tp,
        ],
        dim=1,
    )

    q, k, v, z, b, a = [weight.reshape(num_qk_heads, -1, hidden_size) for weight in [q, k, v, z, b, a]]
    qkvz = torch.cat([q, k, v, z], dim=1)
    ba = torch.cat([b, a], dim=1)

    qkvz = qkvz.reshape(-1, hidden_size)
    ba = ba.reshape(-1, hidden_size)

    assert qkvz.numel() + ba.numel() == in_proj.numel(), (
        f"QKVZBA weights are not correctly split, {qkvz.numel()=}, {ba.numel()=}, {in_proj.numel()=}"
    )

    return qkvz, ba


def _fuse_gdn_separate_to_grouped(
    config,
    qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert four separate (flat) GDN projection tensors into the head-grouped
    ``qkvz`` and ``ba`` format expected by :func:`merge_gdn_linear_weights`.

    Args:
        config: 配置对象。
        qkv: Flat ``[Q; K; V]`` tensor of shape ``(qk_dim*2 + v_dim, hidden)``.
        z:   Z projection of shape ``(v_dim, hidden)``.
        b:   B projection of shape ``(num_v_heads, hidden)``.
        a:   A projection of shape ``(num_v_heads, hidden)``.

    Returns:
        Tuple of (qkvz, ba) in head-grouped layout that
        :func:`merge_gdn_linear_weights` can consume directly.
    """
    hidden_size = config.hidden_size
    qk_head_dim = config.linear_key_head_dim
    v_head_dim = config.linear_value_head_dim
    num_qk_heads = config.linear_num_key_heads
    num_v_heads = config.linear_num_value_heads
    qk_dim = qk_head_dim * num_qk_heads
    v_dim = v_head_dim * num_v_heads
    v_per_group = num_v_heads // num_qk_heads

    expected_qkv = (qk_dim * 2 + v_dim, hidden_size)
    expected_z = (v_dim, hidden_size)
    expected_ba = (num_v_heads, hidden_size)
    if tuple(qkv.shape) != expected_qkv:
        raise ValueError(f"qkv shape mismatch: expected {expected_qkv}, got {tuple(qkv.shape)}")
    if tuple(z.shape) != expected_z:
        raise ValueError(f"z shape mismatch: expected {expected_z}, got {tuple(z.shape)}")
    if tuple(b.shape) != expected_ba:
        raise ValueError(f"b shape mismatch: expected {expected_ba}, got {tuple(b.shape)}")
    if tuple(a.shape) != expected_ba:
        raise ValueError(f"a shape mismatch: expected {expected_ba}, got {tuple(a.shape)}")

    # --- Split flat QKV into individual components ---
    q_flat, k_flat, v_flat = torch.split(qkv, [qk_dim, qk_dim, v_dim], dim=0)

    # --- Reshape every component to (num_qk_heads, per_group_dim, hidden) ---
    q_g = q_flat.reshape(num_qk_heads, qk_head_dim, hidden_size)
    k_g = k_flat.reshape(num_qk_heads, qk_head_dim, hidden_size)
    v_g = v_flat.reshape(num_qk_heads, v_per_group * v_head_dim, hidden_size)
    z_g = z.reshape(num_qk_heads, v_per_group * v_head_dim, hidden_size)
    b_g = b.reshape(num_qk_heads, v_per_group, hidden_size)
    a_g = a.reshape(num_qk_heads, v_per_group, hidden_size)

    # --- Assemble grouped qkvz and ba ---
    qkvz = torch.cat([q_g, k_g, v_g, z_g], dim=1).reshape(-1, hidden_size)
    ba = torch.cat([b_g, a_g], dim=1).reshape(-1, hidden_size)

    return qkvz, ba


def _split_gdn_grouped_to_separate(
    config,
    qkvz: torch.Tensor,
    ba: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert head-grouped ``qkvz`` and ``ba`` tensors (as produced by
    :func:`split_gdn_linear_weights`) back into four flat tensors.

    Returns:
        Tuple of (qkv, z, b, a) where each tensor has a flat per-component layout.
    """
    hidden_size = config.hidden_size
    qk_head_dim = config.linear_key_head_dim
    v_head_dim = config.linear_value_head_dim
    num_qk_heads = config.linear_num_key_heads
    num_v_heads = config.linear_num_value_heads
    v_per_group = num_v_heads // num_qk_heads

    expected_qkvz_dim0 = num_qk_heads * (qk_head_dim * 2 + v_per_group * v_head_dim * 2)
    expected_ba_dim0 = num_qk_heads * v_per_group * 2
    if qkvz.ndim != 2 or qkvz.shape[0] != expected_qkvz_dim0 or qkvz.shape[1] != hidden_size:
        raise ValueError(
            f"qkvz shape mismatch: expected ({expected_qkvz_dim0}, {hidden_size}), got {tuple(qkvz.shape)}"
        )
    if ba.ndim != 2 or ba.shape[0] != expected_ba_dim0 or ba.shape[1] != hidden_size:
        raise ValueError(f"ba shape mismatch: expected ({expected_ba_dim0}, {hidden_size}), got {tuple(ba.shape)}")

    # --- Split grouped QKVZ ---
    qkvz_g = qkvz.reshape(num_qk_heads, -1, hidden_size)
    q_g, k_g, v_g, z_g = torch.split(
        qkvz_g,
        [qk_head_dim, qk_head_dim, v_per_group * v_head_dim, v_per_group * v_head_dim],
        dim=1,
    )
    q_flat = q_g.reshape(-1, hidden_size)
    k_flat = k_g.reshape(-1, hidden_size)
    v_flat = v_g.reshape(-1, hidden_size)
    z_flat = z_g.reshape(-1, hidden_size)
    qkv = torch.cat([q_flat, k_flat, v_flat], dim=0)

    # --- Split grouped BA ---
    ba_g = ba.reshape(num_qk_heads, -1, hidden_size)
    b_g, a_g = torch.split(ba_g, [v_per_group, v_per_group], dim=1)
    b_flat = b_g.reshape(-1, hidden_size)
    a_flat = a_g.reshape(-1, hidden_size)

    return qkv, z_flat, b_flat, a_flat


def interleave_gdn_conv1d(conv: torch.Tensor, config, tp_size: int) -> torch.Tensor:
    """HF conv1d ``[conv_dim, 1, k]`` (flat ``[q|k|v]`` segments) -> TP-interleaved global tensor.

    After this, mbridge's ``chunk(tp_size, dim=0)`` lands ``[q_r|k_r|v_r]`` on each rank,
    matching the qkv-segment interleave of :func:`merge_gdn_linear_weights` so the conv
    channels stay aligned head-for-head with the fused in_proj on every rank.
    """
    qk_dim = config.linear_key_head_dim * config.linear_num_key_heads
    v_dim = config.linear_value_head_dim * config.linear_num_value_heads
    kk = conv.shape[-1]
    q, k_seg, v_seg = torch.split(conv, [qk_dim, qk_dim, v_dim], dim=0)
    q, k_seg, v_seg = [w.reshape(tp_size, -1, 1, kk) for w in (q, k_seg, v_seg)]
    return torch.cat([q, k_seg, v_seg], dim=1).reshape(-1, 1, kk).contiguous()


def deinterleave_gdn_conv1d(conv: torch.Tensor, config, tp_size: int) -> torch.Tensor:
    """Inverse of :func:`interleave_gdn_conv1d`: TP-interleaved global conv -> flat ``[q|k|v]``.

    Used on the mcore->HF path (update_weights -> rollout).
    """
    qk_dim = config.linear_key_head_dim * config.linear_num_key_heads
    v_dim = config.linear_value_head_dim * config.linear_num_value_heads
    kk = conv.shape[-1]
    qk_local, v_local = qk_dim // tp_size, v_dim // tp_size
    conv = conv.reshape(tp_size, -1, 1, kk)
    q, k_seg, v_seg = torch.split(conv, [qk_local, qk_local, v_local], dim=1)
    q, k_seg, v_seg = [w.reshape(-1, 1, kk) for w in (q, k_seg, v_seg)]
    return torch.cat([q, k_seg, v_seg], dim=0).contiguous()
