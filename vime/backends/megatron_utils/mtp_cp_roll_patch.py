"""[MTP + CP fix] 修正 MTP 的 packed-seq roll 在 Context-Parallel 下读错 cu_seqlens 约定的问题。

背景(见 docs/design/chunked_mtp_lmhead.md;根因 commit e19530af 的 CP ring fix):
  vime 直接驱动 Megatron core 的 GPTModel.forward(非 MindSpeed 的 gpt_forward_wrapper),
  ring attention 需要的 packed_seq_params 字段由 vime 在 data.get_batch 里手工填。ring kernel
  要求 `cu_seqlens_q` 是 **CP-local、无前导 0** 的约定(带前导 0 会产生零长段 → npu_fusion_attention
  161001),于是 data.py 把 `cu_seqlens_q` 重指向成 ring 约定,同时把 **origin(×cp_size、带前导 0)**
  的那份保留在 `cu_seqlens_q_padded`(供 RoPE)和 `cu_seqlens_gdn`(供 GDN)。

  该 commit 只盘点了当时的三个消费者(ring / RoPE / GDN)。MTP 的 `_roll_tensor_packed_seq` 是**第四个
  消费者**,当时还没进场:它硬读 `packed_seq_params.cu_seqlens_q`,却按 **origin 约定**处理 ——
    for i in range(len(cu_seqlens)-1):            # 需要前导 0
        local_start = cu_seqlens[i] // cp_size    # 需要值是 ×cp_size 的
  拿到 ring 约定(已 ÷cp、无前导 0)后:迭代错位 + 二次除法 → tensor_slice 退化成极短切片 →
  `tensor_slice.chunk(2)` 只返回 1 块 → `tensor_recv_list[1]` 越界 IndexError(cp_size>1 分支)。

方案:MTP roll 想要的正是 data.py 特意保留的 origin 约定 = `cu_seqlens_q_padded`。故在调用原
  `_roll_tensor_packed_seq` 前,把 `packed_seq_params.cu_seqlens_q` 临时换成 `cu_seqlens_q_padded`,
  调用后立刻还原 —— ring attention 在后续 layer forward 才读 `cu_seqlens_q`,还原后不受影响。
  这样 roll 每段本地切片 = 2*chunk_size(≥2,front+镜像 tail),chunk(2) 出对称两块,不崩且数学正确。

作用域:
  - cp_size==1:data.py 跳过重指向,`cu_seqlens_q_padded` 不存在 → select 回退到 `cu_seqlens_q`
    (此时它本就是 origin+前导 0)→ 本 patch 为 no-op。这也解释了为何 CP=1 从不崩。
  - 未开 CP / 未开 MTP:`_roll_tensor_packed_seq` 只被 MTP 代码路径调用 → 无 MTP 即不触发。
  不动 Megatron 源码;幂等。
"""


def select_roll_cu_seqlens(packed_seq_params):
    """返回 packed-seq roll 应使用的 cu_seqlens(纯逻辑,可单测,不依赖 megatron/distributed)。

    优先用 data.py 保留的 origin 约定 `cu_seqlens_q_padded`(×cp_size、带前导 0,正是 roll 写死
    的处理约定);缺失时(cp_size==1,未走 ring 重指向)回退 `cu_seqlens_q`(此时它本就是 origin)。
    """
    padded = getattr(packed_seq_params, "cu_seqlens_q_padded", None)
    return padded if padded is not None else packed_seq_params.cu_seqlens_q


def make_patched_roll(orig):
    """构造包装后的 roll(纯逻辑,可注入 stub orig 单测 swap/restore 行为)。

    _roll_tensor_packed_seq 内部只读 `packed_seq_params.cu_seqlens_q` 一次(取到局部变量),故
    临时换字段 → 调 orig → finally 还原 是安全且最小的改动,不必复制整段含 distributed 的实现。
    """

    def _patched(tensor, shifts, dims, packed_seq_params, cp_group=None):
        chosen = select_roll_cu_seqlens(packed_seq_params)
        saved = packed_seq_params.cu_seqlens_q
        if chosen is saved:
            return orig(tensor, shifts, dims, packed_seq_params, cp_group)
        packed_seq_params.cu_seqlens_q = chosen
        try:
            return orig(tensor, shifts, dims, packed_seq_params, cp_group)
        finally:
            packed_seq_params.cu_seqlens_q = saved

    return _patched


def apply_mtp_cp_roll_patch():
    """monkey-patch megatron 的 _roll_tensor_packed_seq(幂等)。在开 MTP 训练时调用。

    该函数在 multi_token_prediction 模块内以模块全局名被调用(_get_embeddings 里),故重指向模块
    属性即可在调用点生效(Python 调用时按模块全局解析)。
    """
    from megatron.core.transformer import multi_token_prediction as mtp_mod

    if getattr(mtp_mod, "_mtp_cp_roll_patched", False):
        return
    mtp_mod._roll_tensor_packed_seq = make_patched_roll(mtp_mod._roll_tensor_packed_seq)
    mtp_mod._mtp_cp_roll_patched = True
