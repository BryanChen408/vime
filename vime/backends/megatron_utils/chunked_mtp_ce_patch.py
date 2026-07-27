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


def _enabled() -> bool:
    return os.environ.get("QWEN36_CHUNK_LMHEAD", "0") == "1"


def _chunk_size() -> int:
    try:
        return max(1, int(os.environ.get("QWEN36_MTP_CE_CHUNK", "1024")))
    except ValueError:
        return 1024


def chunked_ce_over_seq(hidden, weight, labels, chunk, compute_lm_loss):
    """按 seq 分块算逐 token CE(纯机制,可单测)。

    Args:
        hidden: [s, b, h](已 SP-gather);weight: [vocab/tp, h];labels: [b, s]
        chunk: 每块 seq 长度
        compute_lm_loss(labels_chunk[b,c], logits_chunk[c,b,vocab/tp]) -> [b,c]
            (即 LanguageModule.compute_language_model_loss:内部做 vocab-parallel CE)
    Returns: [b, s] 逐 token CE(拼回),数值等于不分块的 full CE。
    """
    s = hidden.size(0)
    outs = []
    for start in range(0, s, chunk):
        end = min(start + chunk, s)
        logits_chunk = F.linear(hidden[start:end], weight)   # [c, b, vocab/tp]
        outs.append(compute_lm_loss(labels[:, start:end], logits_chunk))  # [b, c]
    return torch.cat(outs, dim=1)


def apply_chunked_mtp_ce_patch():
    """monkey-patch LanguageModule.compute_output_layer_and_language_model_loss(幂等)。"""
    global _patched
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
        # 1) SP 先 gather 一次 → [s, b, h](与主头 chunked LM-head bypass 同一处理)
        if sequence_parallel_enabled:
            from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
            tp_group = getattr(column_parallel_linear, "tp_group", None)
            hidden = gather_from_sequence_parallel_region(hidden, group=tp_group)

        s = hidden.size(0)
        # labels [b, s] 与 gather 后的 seq 对齐(compute_language_model_loss 内部再转 [s,b])
        assert labels.size(1) == s, f"labels seq {labels.size(1)} != hidden seq {s}"

        # 2)+3) 按 seq 切块 → 每块 matmul(真 weight 保梯度)+ vocab-parallel CE → 拼回 [b,s]
        return chunked_ce_over_seq(
            hidden, weight, labels, _chunk_size(), self.compute_language_model_loss
        )

    LanguageModule.compute_output_layer_and_language_model_loss = _patched_fn
    LanguageModule._chunked_mtp_ce_patched = True
    _patched = True
