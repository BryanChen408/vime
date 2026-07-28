"""[MTP chunked CE] 把 MTP 头的 CE 纳入分块,避免 NPU 上全量 [seq, vocab] logits OOM。

背景(见 docs/design/chunked_mtp_lmhead.md):
  MTP 训练时 Megatron `_postprocess` 对每个 MTP 层调
  `LanguageModule.compute_output_layer_and_language_model_loss(hidden, labels=mtp_labels, ...)`
  得到逐 token CE `[b,s]`,再经 roll/scale/`MTPLossAutoScaler`/`MTPLossLoggingHelper` 注入反向与日志。
  该函数的 fused 分支(linear_cross_entropy)是 blackwell/CUDA kernel,NPU 没有;NPU 走非融合分支
  `functional_call(output_layer, hidden)` → 全量 `[seq, vocab/tp]` logits → OOM。

方案:仅拦截**非融合分支**,按序列分块算 CE —— SP 先 gather 一次,再按 seq 切块,每块用真 weight
  做 matmul 出 `[chunk, b, vocab/tp]`,复用 `self.compute_language_model_loss` 走 vocab-parallel CE,
  拼回 `[b,s]`。峰值从 `[seq, vocab]` 降到 `[chunk, vocab]`,与主头 chunked LM-head 同构。
  fused 分支 / 未开 chunked / value-head 一律走原实现(no-op)。

开关:复用 QWEN36_CHUNK_LMHEAD=1(与主头 chunked LM-head 同一开关);块大小 QWEN36_MTP_CE_CHUNK(默认 1024)。
"""
import os

import torch
import torch.nn.functional as F

_patched = False
_debug_done = False  # [DEBUG] QWEN36_MTP_CE_DEBUG:三路对照只在首个 microbatch 打一次


def _enabled() -> bool:
    return os.environ.get("QWEN36_CHUNK_LMHEAD", "0") == "1"


def _chunk_size() -> int:
    try:
        return max(1, int(os.environ.get("QWEN36_MTP_CE_CHUNK", "1024")))
    except ValueError:
        return 1024


def chunked_ce_over_seq(hidden, weight, labels, chunk, compute_lm_loss, bias=None):
    """按 seq 分块算逐 token CE(纯机制,可单测)。

    Args:
        hidden: [s, b, h](已 SP-gather);weight: [vocab/tp, h];labels: [b, s]
        chunk: 每块 seq 长度
        compute_lm_loss(labels_chunk[b,c], logits_chunk[c,b,vocab/tp]) -> [b,c]
            (即 LanguageModule.compute_language_model_loss:内部做 vocab-parallel CE)
        bias: 可选 [vocab/tp](output_layer.bias);LM head 通常无 bias。
    Returns: [b, s] 逐 token CE(拼回),数值等于不分块的 full CE。
    """
    s = hidden.size(0)
    outs = []
    for start in range(0, s, chunk):
        end = min(start + chunk, s)
        logits_chunk = F.linear(hidden[start:end], weight, bias)   # [c, b, vocab/tp]
        outs.append(compute_lm_loss(labels[:, start:end], logits_chunk))  # [b, c]
    return torch.cat(outs, dim=1)


def resolve_mtp_ce_weight(weight_arg, col_linear_kwargs, column_parallel_linear):
    """复刻基线非融合分支(language_module.py:188-195)的权重来源(纯逻辑,可单测)。

    非融合 CE 的 logits 用的是 **column_parallel_linear 模块自身的权重**(经 functional_call,detached),
    `weight` 参数只在 fused 分支用。此模型 shared_embedding_or_output_weight()(=`weight` 参数)与
    output_layer 的权重不是同一个,误用 `weight` 参数会让 logits 全错(2026-07-28 诊断:logits_maxdiff=46、
    mtp_loss 14.8 vs 基线 0.45)。取权重优先级:col_linear_kwargs['weight'](已 detach)→ 模块 weight →
    weight_arg 兜底;返回 detach 后的张量(MTP loss 只训 hidden,不反传共享输出投影,与基线一致)。
    """
    w = col_linear_kwargs.get("weight") if isinstance(col_linear_kwargs, dict) else None
    if w is None:
        w = getattr(column_parallel_linear, "weight", None)
    if w is None:
        w = weight_arg
    return w.detach() if w is not None else None


def apply_chunked_mtp_ce_patch():
    """monkey-patch LanguageModule.compute_output_layer_and_language_model_loss(幂等)。

    默认启用;逃生阀 QWEN36_MTP_CE_CHUNK_OFF=1 可整体跳过 → MTP 头走 Megatron 原生全量-logits CE
    (仅在短序列放得下时可用,供对照诊断)。2026-07-28 已修复权重取错的前向 bug(见 resolve_mtp_ce_weight
    及 docs/design/chunked_mtp_lmhead.md §11):分块 logits 与基线逐元素相等,mtp_loss 0.45 == 基线。
    """
    global _patched
    if os.environ.get("QWEN36_MTP_CE_CHUNK_OFF", "0") == "1":
        return
    if _patched:
        return
    from megatron.core.models.common.language_module.language_module import LanguageModule

    _orig = LanguageModule.compute_output_layer_and_language_model_loss

    def _patched_fn(
        self,
        hidden,
        labels,
        weight=None,
        sequence_parallel_enabled=False,
        column_parallel_linear=None,
        col_linear_kwargs={},
        reduction="none",
        ignore_index=-100,
    ):
        # 只接管:开了 chunked + 非融合路径(fused=linear 走原实现) + reduction=none(MTP 用 none)
        fused_linear = (
            getattr(self.config, "cross_entropy_loss_fusion", False)
            and getattr(self.config, "cross_entropy_fusion_impl", None) == "linear"
        )
        if (
            not _enabled()
            or fused_linear
            or reduction != "none"
            or weight is None
            or column_parallel_linear is None
        ):
            return _orig(
                self, hidden, labels, weight, sequence_parallel_enabled,
                column_parallel_linear, col_linear_kwargs, reduction, ignore_index,
            )

        # hidden: [s_local, b, h](SP 时序列维按 tp 切分);labels: [b, s]
        hidden_in = hidden  # 保留 pre-gather 引用,供 debug 对照基线 _orig(它内部自己 gather)
        # 1) SP 先 gather 一次 → [s, b, h](与主头 chunked LM-head bypass 同一处理)
        if sequence_parallel_enabled:
            from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
            tp_group = getattr(column_parallel_linear, "tp_group", None)
            hidden = gather_from_sequence_parallel_region(hidden, group=tp_group)

        s = hidden.size(0)
        # labels [b, s] 与 gather 后的 seq 对齐(compute_language_model_loss 内部再转 [s,b])
        assert labels.size(1) == s, f"labels seq {labels.size(1)} != hidden seq {s}"

        # [BUGFIX 2026-07-28] logits 必须用 **模块自身权重**,不是 `weight` 参数(见 resolve_mtp_ce_weight)。
        mm_weight = resolve_mtp_ce_weight(weight, col_linear_kwargs, column_parallel_linear)
        mm_bias = None if getattr(column_parallel_linear, "skip_bias_add", False) else getattr(column_parallel_linear, "bias", None)
        if mm_bias is not None:
            mm_bias = mm_bias.detach()

        # [DEBUG] QWEN36_MTP_CE_DEBUG=1:首个 microbatch 同一输入下三路对照,定位分块 bug 位置。
        #   L_orig  = 基线 functional_call(ColumnParallelLinear)  ← 正确参照(需 T 放得下,用 CP=1/T≤4096)
        #   L_full  = 我的 gather+F.linear,但 chunk=整段(隔离"前向"是否与基线等价)
        #   L_chunk = 我的 gather+F.linear,chunk=生产值(隔离"分块"是否引入错位)
        # L_full≠L_orig → bug 在 F.linear/gather/weight 前向;L_full==L_orig 但 L_chunk≠ → 分块错位。
        global _debug_done
        if os.environ.get("QWEN36_MTP_CE_DEBUG", "0") == "1" and not _debug_done:
            _debug_done = True
            try:
                with torch.no_grad():
                    # 干净基线:走 ColumnParallelLinear 的 **类方法** forward(绕开 chunked_lm_head 在
                    # *实例* 上打的 bypass —— 实例 forward 被替换成返回 hidden,才是 L_orig 被污染的原因)。
                    # 类方法 forward 内部自己做 SP-gather(sequence_parallel=True)+ matmul,得真 logits。
                    clean_logits, _ = type(column_parallel_linear).forward(
                        column_parallel_linear, hidden_in, **col_linear_kwargs
                    )
                    L_clean = self.compute_language_model_loss(labels, clean_logits)
                    # 我的前向(已修:用 mm_weight/mm_bias)——整段 F.linear,logits 应与干净基线逐元素相等
                    my_logits = F.linear(hidden, mm_weight, mm_bias)
                    L_full = self.compute_language_model_loss(labels, my_logits)
                    L_chunk = chunked_ce_over_seq(hidden, mm_weight, labels, _chunk_size(), self.compute_language_model_loss, mm_bias)
                    print(
                        f"[MTP-CE-DEBUG] hidden_in={tuple(hidden_in.shape)} gathered={tuple(hidden.shape)} "
                        f"labels={tuple(labels.shape)} weight={type(weight).__name__}{tuple(weight.shape)} "
                        f"clean_logits={tuple(clean_logits.shape)} my_logits={tuple(my_logits.shape)} "
                        f"sp={sequence_parallel_enabled}\n"
                        f"[MTP-CE-DEBUG] mean  L_clean={L_clean.mean().item():.4f}  "
                        f"L_full={L_full.mean().item():.4f}  L_chunk={L_chunk.mean().item():.4f}\n"
                        f"[MTP-CE-DEBUG] logits_maxdiff(my-vs-clean)={(my_logits-clean_logits).abs().max().item():.4e}  "
                        f"loss_maxdiff(full-vs-clean)={(L_full-L_clean).abs().max().item():.4e}",
                        flush=True,
                    )
            except Exception as e:  # 对照失败不能拖垮训练
                import traceback
                print(f"[MTP-CE-DEBUG] compare failed: {e!r}\n{traceback.format_exc()}", flush=True)

        # 2)+3) 按 seq 切块 → 每块 matmul(模块权重,detached)+ vocab-parallel CE → 拼回 [b,s]
        return chunked_ce_over_seq(
            hidden, mm_weight, labels, _chunk_size(), self.compute_language_model_loss, mm_bias
        )

    LanguageModule.compute_output_layer_and_language_model_loss = _patched_fn
    LanguageModule._chunked_mtp_ce_patched = True
    _patched = True
