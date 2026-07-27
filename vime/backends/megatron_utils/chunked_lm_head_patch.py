"""[Phase B / dev_12] Chunked LM-head 接入:GPTModel forward monkey-patch。

问题:32k 序列时 GPTModel.forward 在 PP-last rank 一次性算完整 logits [s_local, vocab/tp],
log_probs 阶段 compute_log_probs 材料化它 → OOM(实测 NPU4 53GB 顶爆)。

方案:当 QWEN36_CHUNK_LMHEAD=1 且 labels=None(slime 的 forward 调用方式)时,
patch 后的 forward **跳过 output_layer**,直接返回 decoder hidden [b, s, h],
并把 output_layer 的 weight 引用挂到返回 tensor 的属性上(`._lm_head_weight`)。
下游 loss(get_log_probs_and_entropy)检测到该属性 → 走 chunked_logprob_entropy_from_hidden,
per-chunk 算 logit,峰值 [chunk, vocab/tp] 与序列长无关。

retrocompat:QWEN36_CHUNK_LMHEAD!=1 时 patch 不改变行为(走原 forward)。
仅 PP-last rank(post_process)有 output_layer;非 last rank 原样返回 hidden(本就如此)。
"""
import os

import torch

# [Phase B/dev_12] 模块级保存 PP-last rank 的 lm_head weight 引用。
# 不用 tensor 属性传递(PP 引擎中间可能 .contiguous()/重建 tensor 丢属性),改全局存取最稳。
# 单 PP-last rank 只有一个 output_layer,本进程内唯一,无歧义。
_LM_HEAD_WEIGHT = None


def get_captured_lm_head_weight():
    """供下游 loss 取 lm_head weight(chunked 路径)。None=未开 chunked 或非 PP-last rank。"""
    return _LM_HEAD_WEIGHT


def _chunked_lm_head_enabled() -> bool:
    return os.environ.get("QWEN36_CHUNK_LMHEAD", "0") == "1"


def apply_chunked_lm_head_patch():
    """对 GPTModel.forward 打 monkey-patch(幂等)。在 model 构建后调用。"""
    from megatron.core.models.gpt.gpt_model import GPTModel

    if getattr(GPTModel, "_chunked_lm_head_patched", False):
        return
    _orig_forward = GPTModel.forward

    def _patched_forward(self, *args, **kwargs):
        global _LM_HEAD_WEIGHT
        labels = kwargs.get("labels", None)
        if (
            not _chunked_lm_head_enabled()
            or not getattr(self, "post_process", False)
            or labels is not None
            # [P0] 从"结构含 MTP(mtp_process)"收窄为"本次 forward 真传了 mtp_labels"。
            # logprob forward(forward_only)不传 mtp_labels → 即使模型结构含 MTP 也应分块,
            # 否则一开 --enable-mtp-training 就把主头分块全局关掉、compute_log_prob 先 OOM。
            # 带 mtp_labels 的 train forward 仍跳过(MTP 头分块由 chunked_mtp_ce_patch 处理)。
            or ((kwargs.get("mtp_kwargs") or {}).get("mtp_labels") is not None)
            # [F-PPO-1] critic 的 value head 是 hidden→1 的 output_layer,别旁路它。旁路只为躲 LM-head
            # 的 [T, vocab] logits OOM;value 出 [T, 1],材料化本就 trivial、永不 OOM。若旁路,
            # critic get_values 会收到 hidden [T, h] 而非 values [T, 1] → get_responses 断言
            # size(-1)==1 崩。critic 与 actor 共用此 class-level patch,故按输出维区分:
            # output_layer.weight.shape[0]==1 ⇒ value head(LM head 是 vocab/tp≫1,新条件恒 False)。
            or getattr(getattr(self, "output_layer", None), "weight", None) is None
            or self.output_layer.weight.shape[0] == 1
        ):
            return _orig_forward(self, *args, **kwargs)

        # 旁路 output_layer:patch 其 .forward(nn.Module.__call__ 内部调 self.forward,
        # 实例属性覆盖 forward 生效)。让它返回 (hidden, None),GPTModel 解包后 logits=hidden。
        # ⚠️ SP 关键:output_layer(ColumnParallelLinear, sequence_parallel=True)的原 forward 会先
        #   gather_from_sequence_parallel_region 把序列 [s/tp,b,h] gather 成 [s,b,h] 再 matmul。
        #   bypass 必须补这个 gather,否则返回 SP-sharded hidden(序列维只有 1/tp)→ 下游 slice 错乱。
        #   LM head 是真 vocab-parallel(weight 沿 vocab 切),SP gather 用标准反向(reduce-scatter,
        #   tensor_parallel_output_grad=True 默认),与 GDN(复制计算用 split,dev_11)不同。
        output_layer = self.output_layer
        _LM_HEAD_WEIGHT = output_layer.weight  # [vocab/tp, h];供下游 loss 取
        _sp_enabled = getattr(output_layer, "sequence_parallel", False)
        _tp_group = getattr(output_layer, "tp_group", None)

        def _bypass_forward(hidden, weight=None, runtime_gather_output=None):
            if _sp_enabled:
                from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

                hidden = gather_from_sequence_parallel_region(hidden, group=_tp_group)
            return hidden, None

        _orig_ol_forward = output_layer.forward
        try:
            output_layer.forward = _bypass_forward  # type: ignore
            # 原 forward 在 labels=None 时:logits, _ = output_layer(hidden) → logits=hidden [s,b,h];
            # 末尾 `return logits.transpose(0,1).contiguous()` → [b,s,h]
            out = _orig_forward(self, *args, **kwargs)
        finally:
            output_layer.forward = _orig_ol_forward  # type: ignore

        # out 现在是 [b, s, h] 的 hidden。同时挂属性(双保险:属性活则用属性,否则用全局)。
        try:
            out._lm_head_weight = _LM_HEAD_WEIGHT
        except (AttributeError, RuntimeError):
            pass
        return out

    GPTModel.forward = _patched_forward
    GPTModel._chunked_lm_head_patched = True

