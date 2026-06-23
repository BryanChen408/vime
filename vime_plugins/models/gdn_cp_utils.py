"""[Phase3 CP] GDN context-parallel 工具(从新 MA 栈 dev_10 §13.4 移植到老 slime-mindspeed 栈)。

包含两部分:
  A. packed×CP 的纯 torch thd 重排(原样从新栈拷):TEnpu 没有 `tex.thd_get_partitioned_indices`,
     用纯 torch 复刻,供 GDN 在 packed(THD)+ CP>1 下做 undo/redo attention load-balancing。
  B. **[老栈适配 gap#2] 高层 CP helper 的本地 vendored 实现**:
     `tensor_a2a_cp2hp` / `tensor_a2a_hp2cp` / `get_parameter_local_cp`。
     新栈这三个函数在 `megatron.core.ssm.gated_delta_net` 里(NVIDIA 新版 Megatron),
     **老栈 `/home/docker/slime-mindspeed/Megatron-LM` 没有这三个函数**(老栈 CP 是 `MambaContextParallel`
     类 + 模块级 `_all_to_all_cp2hp/hp2cp` + `_undo/_redo_attention_load_balancing(input_, cp_size)`)。
     算法层面老栈的底层 primitive(`all_to_all`、`_all_to_all_cp2hp/hp2cp`、非 packed 的 undo/redo)
     与新栈逐字一致,故这里照新栈高层 helper 在老栈 primitive 上重写,签名与新栈对齐,
     使 `_forward_cp` 无需改动 import 之外的逻辑、也无需碰老 Megatron。

A 部分语义照 megatron `get_batch_on_this_cp_rank` / `get_thd_batch_on_this_cp_rank`:
每条序列切 2*cp 段,cp_rank 取段 cp_rank 与段 (2*cp-1-cp_rank),返回这些 token 在全 buffer 的下标
(段内[早,晚]、跨序列拼接)。仅 packed × CP>1 用;CP=1 或非 packed 不走 thd(走整段 reorder)。
"""
from typing import List, Optional

import torch

# [老栈适配 gap#2] 复用老 Megatron 已有的底层 primitive(算法与新栈逐字一致,经核对)。
#   - all_to_all:CP 组 all-to-all 通信
#   - _all_to_all_cp2hp / _all_to_all_hp2cp:序列<->头维 a2a(seq-first [s,b,h])
#   - _undo_attention_load_balancing / _redo_attention_load_balancing:非 packed 整段 zigzag undo/redo
from megatron.core.tensor_parallel import all_to_all  # noqa: F401  (primitive 用于校验可用)
from megatron.core.ssm.mamba_context_parallel import (
    _all_to_all_cp2hp,
    _all_to_all_hp2cp,
    _undo_attention_load_balancing,
    _redo_attention_load_balancing,
)


# ============================================================================
# A. packed×CP thd 索引重排(纯 torch,原样从新栈 gdn_cp_utils.py 拷)
# ============================================================================
def thd_get_partitioned_indices_torch(cu_seqlens, total_tokens: int, cp_size: int, cp_rank: int):
    """返回 cp_rank 在全 packed buffer([total_tokens, ...] 的 dim0)持有的 token 原始下标。

    Args:
        cu_seqlens: [N+1] long,各序列累积边界(应为 *padded* 版,每条长度可被 2*cp 整除)。
        total_tokens: buffer 的 dim0 总长。
        cp_size, cp_rank: CP 组大小 / 本 rank。
    与 native `_undo/_redo_attention_load_balancing` 的 packed 分支配合:
      undo: output[idx] = input_[start:end];  redo: index[start:end] = idx; input_.index_select(0, index)。
    """
    if isinstance(cu_seqlens, torch.Tensor):
        cu = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    else:
        cu = [int(x) for x in cu_seqlens]
    two_cp = 2 * cp_size
    idx = []
    for i in range(len(cu) - 1):
        s, e = cu[i], cu[i + 1]
        L = e - s
        if L <= 0:
            continue
        assert L % two_cp == 0, (
            f"序列 {i} 长度 {L} 不能被 2*cp={two_cp} 整除;packed×CP 需用 cu_seqlens_padded(每条 pad 到 2*cp 整数倍)"
        )
        cs = L // two_cp  # 段长
        lo = s + cp_rank * cs                       # 早段起点
        hi = s + (two_cp - 1 - cp_rank) * cs        # 晚段起点
        idx.append(torch.arange(lo, lo + cs, dtype=torch.long))
        idx.append(torch.arange(hi, hi + cs, dtype=torch.long))
    out = torch.cat(idx) if idx else torch.empty(0, dtype=torch.long)
    return out


def undo_attention_load_balancing_thd(input_, cu_seqlens, cp_size):
    """packed 版 undo:把 zigzag 负载均衡布局还原成自然顺序(纯 torch,替 native 的 tex 分支)。
    input_: [total_tokens, ...](全 buffer,各 rank 切片按 rank 顺序拼接)。"""
    if cp_size == 1:
        return input_
    total = input_.size(0)
    assert total % cp_size == 0
    per_rank = total // cp_size
    output = torch.empty_like(input_)
    for r in range(cp_size):
        idx = thd_get_partitioned_indices_torch(cu_seqlens, total, cp_size, r).to(input_.device)
        output[idx] = input_[r * per_rank:(r + 1) * per_rank]
    return output


def redo_attention_load_balancing_thd(input_, cu_seqlens, cp_size):
    """packed 版 redo:自然顺序 → zigzag 负载均衡布局(undo 的逆)。"""
    if cp_size == 1:
        return input_
    total = input_.size(0)
    assert total % cp_size == 0
    per_rank = total // cp_size
    index = torch.empty(total, dtype=torch.long, device=input_.device)
    for r in range(cp_size):
        idx = thd_get_partitioned_indices_torch(cu_seqlens, total, cp_size, r).to(input_.device)
        index[r * per_rank:(r + 1) * per_rank] = idx
    return input_.index_select(0, index)


# ---------------- cu_seqlens 末尾补齐到 2*cp 整数倍(真实变长 packed×CP 必需,dev_10 §8.4)----------------
def compute_cp_seqlen_padding(cu_seqlens, cp_size):
    """给 packed `cu_seqlens`([N+1]),每条序列**末尾补**到 `2*cp` 整数倍,返回:
      cu_seqlens_padded([N+1])、total_padded(int)、scatter_idx([total_real] long)。
    scatter_idx[i] = 第 i 个真实 token(自然 packed 序)在 padded buffer 的位置(pad 位不在其中)。
    因果 GDN:pad 补在每条序列末尾,causal 下不影响真实 token 输出;unpad 时按 scatter_idx 取回即可。
    """
    two_cp = 2 * cp_size
    cu = [int(x) for x in (cu_seqlens.tolist() if isinstance(cu_seqlens, torch.Tensor) else cu_seqlens)]
    real_lens = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
    padded_lens = [((L + two_cp - 1) // two_cp) * two_cp for L in real_lens]
    cu_padded = [0]
    for pl in padded_lens:
        cu_padded.append(cu_padded[-1] + pl)
    total_padded = cu_padded[-1]
    scatter = []
    for i, L in enumerate(real_lens):
        ps = cu_padded[i]
        scatter.extend(range(ps, ps + L))           # 真实 token 占每条 padded 序列的前 L 位
    cu_padded_t = torch.tensor(cu_padded, dtype=torch.long)
    scatter_idx = torch.tensor(scatter, dtype=torch.long)
    return cu_padded_t, total_padded, scatter_idx


def pad_packed_for_cp(x, scatter_idx, total_padded):
    """x[total_real, ...] → padded[total_padded, ...](真实 token 散入 scatter_idx 位,其余补零)。"""
    padded = x.new_zeros((total_padded,) + tuple(x.shape[1:]))
    padded[scatter_idx.to(x.device)] = x
    return padded


def unpad_packed_for_cp(padded, scatter_idx):
    """padded[total_padded, ...] → x[total_real, ...](按 scatter_idx 取回真实 token,丢弃 pad)。"""
    return padded.index_select(0, scatter_idx.to(padded.device))


# ============================================================================
# B. [老栈适配 gap#2] 高层 CP helper（vendored;新栈在 megatron gated_delta_net.py,老栈缺,故本地实现）
#    签名与新栈 megatron.core.ssm.gated_delta_net 的同名函数逐字对齐:
#      get_parameter_local_cp(param, dim, cp_group, split_sections=None)
#      tensor_a2a_cp2hp(tensor, seq_dim, head_dim, cp_group, split_sections=None,
#                       undo_attention_load_balancing=True)
#      tensor_a2a_hp2cp(tensor, seq_dim, head_dim, cp_group, split_sections=None,
#                       redo_attention_load_balancing=True)
#    底层用老栈现成 primitive(_all_to_all_cp2hp/hp2cp、_undo/_redo_attention_load_balancing)。
# ============================================================================
def get_parameter_local_cp(
    param: torch.Tensor,
    dim: int,
    cp_group: "torch.distributed.ProcessGroup",
    split_sections: Optional[List[int]] = None,
) -> torch.Tensor:
    """取本 CP rank 的本地参数切片(照新栈 gated_delta_net.get_parameter_local_cp 逐行)。

    cp_size==1 → 原样返回(no-op);split_sections!=None → 先按 dim 切段、每段各自 CP 切、再拼回。
    """
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()

    if cp_size == 1:
        return param

    if split_sections is not None:
        inputs = torch.split(param, split_sections, dim=dim)
        outputs = []
        for p in inputs:
            p = get_parameter_local_cp(p, dim, cp_group)
            outputs.append(p)
        return torch.cat(outputs, dim=dim)

    slices = [slice(None)] * param.dim()
    dim_size = param.size(dim=dim)
    slices[dim] = slice(cp_rank * dim_size // cp_size, (cp_rank + 1) * dim_size // cp_size)
    param = param[tuple(slices)]
    return param


def tensor_a2a_cp2hp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group: "torch.distributed.ProcessGroup",
    split_sections: Optional[List[int]] = None,
    undo_attention_load_balancing: bool = True,
):
    """CP→HP all-to-all(照新栈 gated_delta_net.tensor_a2a_cp2hp 逐行,底层走老栈 _all_to_all_cp2hp)。

    输入 (seq_len, batch, head_dim);seq_dim 仅支持 0,head_dim 仅支持 -1/2。
    split_sections!=None → 按 head_dim 分段、每段单独 a2a(段内不 undo)、再拼;最后整体 undo(若需要)。
    """
    cp_size = cp_group.size()
    if cp_size == 1:
        return tensor

    assert seq_dim == 0, f"tensor_a2a_cp2hp only supports seq_dim == 0 for now, but got {seq_dim=}"
    assert (
        head_dim == -1 or head_dim == 2
    ), f"tensor_a2a_cp2hp only supports head_dim == -1 or 2 for now, but got {head_dim=}"
    assert tensor.dim() == 3, f"tensor_a2a_cp2hp only supports 3-d input tensor, but got {tensor.dim()=}"

    if split_sections is not None:
        inputs = torch.split(tensor, split_sections, dim=head_dim)
        outputs = []
        for x in inputs:
            x = tensor_a2a_cp2hp(
                x, seq_dim=seq_dim, head_dim=head_dim, cp_group=cp_group,
                undo_attention_load_balancing=False,
            )
            outputs.append(x)
        tensor = torch.cat(outputs, dim=head_dim)
    else:
        tensor = _all_to_all_cp2hp(tensor, cp_group)

    if undo_attention_load_balancing:
        tensor = _undo_attention_load_balancing(tensor, cp_size)
    return tensor


def tensor_a2a_hp2cp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group: "torch.distributed.ProcessGroup",
    split_sections: Optional[List[int]] = None,
    redo_attention_load_balancing: bool = True,
):
    """HP→CP all-to-all(照新栈 gated_delta_net.tensor_a2a_hp2cp 逐行,底层走老栈 _all_to_all_hp2cp)。

    先整体 redo(若需要),再按 head_dim 分段单独 a2a(段内不 redo)拼回 / 或整体 a2a。
    """
    cp_size = cp_group.size()
    if cp_size == 1:
        return tensor

    assert seq_dim == 0, f"tensor_a2a_hp2cp only supports seq_dim == 0 for now, but got {seq_dim=}"
    assert (
        head_dim == -1 or head_dim == 2
    ), f"tensor_a2a_hp2cp only supports head_dim == -1 or 2 for now, but got {head_dim=}"
    assert tensor.dim() == 3, f"tensor_a2a_hp2cp only supports 3-d input tensor, but got {tensor.dim()=}"

    if redo_attention_load_balancing:
        tensor = _redo_attention_load_balancing(tensor, cp_size)

    if split_sections is not None:
        inputs = torch.split(tensor, split_sections, dim=head_dim)
        outputs = []
        for x in inputs:
            x = tensor_a2a_hp2cp(
                x, seq_dim=seq_dim, head_dim=head_dim, cp_group=cp_group,
                redo_attention_load_balancing=False,
            )
            outputs.append(x)
        tensor = torch.cat(outputs, dim=head_dim)
    else:
        tensor = _all_to_all_hp2cp(tensor, cp_group)

    return tensor
