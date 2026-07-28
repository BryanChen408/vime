"""MTP+CP roll 的 cu_seqlens 约定修复单测(见 docs/design/chunked_mtp_lmhead.md)。

锁定 mtp_cp_roll_patch 的契约,不依赖 megatron / torch.distributed:
  1. select_roll_cu_seqlens:优先取 origin 约定 cu_seqlens_q_padded,缺失回退 cu_seqlens_q。
  2. make_patched_roll:调用 orig 前把 cu_seqlens_q 临时换成选中的 origin 约定,调用后还原;
     且能重现"若不修 → roll 拿到 ring 约定(÷cp、无前导0)→ 迭代错位/切片退化"的对比。

CP>1 分支真正的 isend/irecv 端到端等价由 NPU run(C6 同一入口)覆盖;这里锁定字段选择/换还原本身。
"""
import torch

from vime.backends.megatron_utils.mtp_cp_roll_patch import (
    make_patched_roll,
    select_roll_cu_seqlens,
)


class _FakePSP:
    """模拟 vime data.py 的 ring 布局:cu_seqlens_q 被重指向成 ring 约定,origin 留在 *_padded。"""

    def __init__(self, cu_seqlens_q, cu_seqlens_q_padded=None):
        self.cu_seqlens_q = cu_seqlens_q
        if cu_seqlens_q_padded is not None:
            self.cu_seqlens_q_padded = cu_seqlens_q_padded


def _ring_and_origin(cp_size=4):
    """还原 data.py:113-144 的两份 cu_seqlens。两段序列,本地各 2*chunk_size。"""
    # local 累计(带前导0),两段本地长 8、6 → *cp_size = origin 约定(roll 期望)
    origin = torch.tensor([0, 8, 14], dtype=torch.int) * cp_size  # cu_seqlens_q_padded
    ring = (origin // cp_size)[1:].contiguous()                    # ring:÷cp、去前导0 = [8,14]
    return ring, origin


def test_select_prefers_padded_origin():
    ring, origin = _ring_and_origin()
    psp = _FakePSP(cu_seqlens_q=ring, cu_seqlens_q_padded=origin)
    chosen = select_roll_cu_seqlens(psp)
    assert torch.equal(chosen, origin), "有 cu_seqlens_q_padded 时必须选 origin 约定"
    # origin 带前导0 且值可被 cp_size 整除还原本地索引;ring 无前导0(第一个元素非 0)
    assert chosen[0].item() == 0, "origin 约定必须带前导 0"
    assert ring[0].item() != 0, "ring 约定无前导 0(这正是喂给 roll 会错位的原因)"


def test_select_falls_back_when_no_padded():
    # cp_size==1:data.py 跳过 ring 重指向,不设 cu_seqlens_q_padded → 回退 cu_seqlens_q
    origin = torch.tensor([0, 8, 14], dtype=torch.int)
    psp = _FakePSP(cu_seqlens_q=origin)  # 无 _padded 属性
    assert torch.equal(select_roll_cu_seqlens(psp), origin), "缺 _padded 时回退 cu_seqlens_q"


def test_patched_roll_swaps_then_restores():
    ring, origin = _ring_and_origin()
    psp = _FakePSP(cu_seqlens_q=ring, cu_seqlens_q_padded=origin)

    seen = {}

    def _orig(tensor, shifts, dims, packed_seq_params, cp_group=None):
        # orig 内部读的正是被换过的 cu_seqlens_q;记录它当时看到的值
        seen["cu"] = packed_seq_params.cu_seqlens_q
        return "ok"

    patched = make_patched_roll(_orig)
    ret = patched(torch.zeros(1, 14), -1, -1, psp, cp_group=None)

    assert ret == "ok"
    assert torch.equal(seen["cu"], origin), "orig 调用期间应看到 origin 约定(已换入)"
    assert torch.equal(psp.cu_seqlens_q, ring), "调用后必须还原成 ring 约定(供 ring attention)"


def test_patched_roll_restores_on_exception():
    ring, origin = _ring_and_origin()
    psp = _FakePSP(cu_seqlens_q=ring, cu_seqlens_q_padded=origin)

    def _orig_raises(*a, **k):
        raise RuntimeError("boom")

    patched = make_patched_roll(_orig_raises)
    try:
        patched(torch.zeros(1, 14), -1, -1, psp, cp_group=None)
    except RuntimeError:
        pass
    assert torch.equal(psp.cu_seqlens_q, ring), "即使 orig 抛异常也必须还原 cu_seqlens_q"


def test_patched_roll_noop_when_no_padded():
    # 无 _padded(cp==1)时不换字段,直接透传 orig
    origin = torch.tensor([0, 8, 14], dtype=torch.int)
    psp = _FakePSP(cu_seqlens_q=origin)
    seen = {}

    def _orig(tensor, shifts, dims, packed_seq_params, cp_group=None):
        seen["cu"] = packed_seq_params.cu_seqlens_q
        return "ok"

    make_patched_roll(_orig)(torch.zeros(1, 14), -1, -1, psp, cp_group=None)
    assert torch.equal(seen["cu"], origin) and torch.equal(psp.cu_seqlens_q, origin)


if __name__ == "__main__":
    test_select_prefers_padded_origin()
    test_select_falls_back_when_no_padded()
    test_patched_roll_swaps_then_restores()
    test_patched_roll_restores_on_exception()
    test_patched_roll_noop_when_no_padded()
    print("MTP+CP roll cu_seqlens 约定修复单测通过 ✅")
