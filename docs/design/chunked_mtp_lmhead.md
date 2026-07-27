# Chunked LM-head 接入 MTP —— 设计文档

> 目标:在 NPU / 长序列(≤32k)下,让 **MTP 训练**(`--enable-mtp-training`)不因 LM-head 全量
> `[seq, vocab=248320]` logits 而 OOM,方式是把 MTP 头的 CE 也纳入 vime 现有的 chunked LM-head 分块路径。
> 背景:Qwen3.6-35B-A3B DAPO-math async RL(单机 A3 aarch64,CANN 9.0.0)。

## 0. TL;DR
- 主头(policy logprob)已有 chunked LM-head(`chunked_lm_head_patch.py`),旁路 `output_layer` 返回
  hidden,下游 `loss.py` 逐块算 logit,峰值 `[chunk, vocab/tp]` 与序列长无关。
- **开 MTP 后两个问题**:
  1. `chunked_lm_head_patch.py:49` 以 `mtp_process`(模型**结构**有 MTP)为跳过条件 → 一开 MTP 就把
     主头分块**全局关掉**,连不碰 MTP 的 logprob forward 都退回全量 logits → 训练开始前先 OOM。
  2. MTP 头的 CE 在 Megatron postprocess 里用**非融合**路径 apply `output_layer` → 全量 logits → OOM;
     NPU 上没有省显存的 fused kernel(见 §2)。
- **关键洞察**:MTP 的 CE = 对 `mtp_labels`(右移 token)的逐 token 负 logprob = **和主头 chunked logprob
  同构** → 复用主头分块,不另写 CE。
- **方案 P0–P3**(§4)+ **6 个 commit**(§6)。

## 1. 调用链(已核实实参)

### A. logprob forward —— `forward_only`(actor 算 old/ref logprob)
`vime/backends/megatron_utils/model.py:413`
```python
forward_kwargs = {"input_ids": tokens, "labels": None, "loss_mask": ..., ...}   # 无 mtp_kwargs
output_tensor = model(**forward_kwargs)
```
→ `GPTModel._postprocess`:`mtp_kwargs.mtp_labels` 为空 → **MTP 分支不触发**(`gpt_model.py:61`)。
→ 本可安全走主头 chunked bypass,但被 §3 的 `:49` 误伤。

### B. train forward —— `train_one_step`
`model.py:605,617`
```python
forward_kwargs = {"input_ids": ..., "labels": None, "loss_mask": ..., ...}
if args.enable_mtp_training:
    forward_kwargs["mtp_kwargs"] = {"mtp_labels": batch["tokens"]}    # labels=None 但带 mtp_labels
output_tensor = model(**forward_kwargs)
```
→ `_postprocess`:主头(labels=None)+ MTP 分支(mtp_labels 有,`gpt_model.py:61/85/93`)**同时**跑。
→ MTP loss 经 `set_*_loss(hidden, scale*mtp_loss)` 挂回,`model.py:867` 用 `MTPLossLoggingHelper.tracker` 读回。

## 2. 为什么 MTP 头省不了显存(NPU）
`megatron/core/models/common/language_module/language_module.py:129
compute_output_layer_and_language_model_loss` 两条路:
- **fused `linear_cross_entropy`**(融合 matmul+CE,不 materialize 全量 logits)——实现在
  `megatron/core/fusions/linear_cross_entropy/**blackwell**/`,是 **NVIDIA Blackwell 的 Triton kernel,
  Ascend NPU 没有**。→ 不能靠 `cross_entropy_fusion_impl=linear` 省显存。
- **非融合**(NPU 实际走):`torch.func.functional_call(output_layer, hidden)` → 全量
  `[seq, vocab/tp]` logits → `compute_language_model_loss` → **OOM**。

## 3. 冲突点(为什么 `:49` 只能整跳)
- 主头 chunked 做法:instance 级覆盖 `output_layer.forward`,让它返回 hidden(`chunked_lm_head_patch.py:72-86`)。
- MTP 非融合分支用 **`functional_call(output_layer, …)`** apply 同一个 `output_layer` → 也会命中被覆盖的
  forward → 拿到 hidden 而非 logits → **MTP 崩**。补丁 `finally` 里在 `_orig_forward` 返回后才恢复
  forward,而 MTP 就在那次 `_orig_forward` 的 postprocess 内跑 → 窗口内必炸。
- 故 `chunked_lm_head_patch.py:49` 直接 `or getattr(self, "mtp_process", False)` 整跳。代价:见 §0 问题 1。

## 4. 设计方案

| 件 | 改动 | 作用 |
|---|---|---|
| **P0 门控** | `chunked_lm_head_patch.py:49` 判据由 `mtp_process`(结构)改为**本次 forward 是否真传 mtp_labels** | logprob forward(无 mtp_labels)恢复分块,消除"开 MTP → compute_log_prob 就 OOM"。**不碰 MTP 头。** |
| **P1 导出 MTP hidden** | patch Megatron `_postprocess` 的 MTP 分支:chunked-mtp 开启时**不调** `compute_output_layer_and_language_model_loss`(全量 logits),改为捕获 `hidden_states_list[1:]`(各 MTP 层 hidden)+ 已 roll 的 `mtp_labels` + loss_mask + scale,交给 vime | MTP 头走"返回 hidden、下游分块"的同构路径,flag 关时 no-op |
| **P2 分块 CE = 复用主头分块** | vime `loss.py`:对 MTP hidden 复用 `chunked_logprob_entropy_from_hidden`,以 mtp_labels 为目标 token → 逐块取 -logprob → 求和得 mtp_loss。复用其 vocab-parallel / CP-offset / SP-gather | 峰值降到 `[chunk, vocab]`,与主头一致;**无新数值路径** |
| **P3 接回总 loss + 日志** | mtp_loss 按 `mtp_loss_scaling_factor/mtp_num_layers` 及 `/num_tokens`(`gpt_model.py:624-631`)缩放,加入 actor 训练 loss;保留/替换 `MTPLossLoggingHelper` 的 `train/mtp_loss` | loss 量级与非分块一致,梯度只进 MTP 参数(`ci_utils.check_mtp_only_grad` 已有校验) |

**关键洞察落地**:MTP CE(`-log p(mtp_label)`)与主头 policy logprob 是同一套逐 token vocab-parallel
logprob 计算,只是目标 token 换成右移的 `mtp_labels`、并做求和/scale。→ P2 直接复用 `loss.py` 主头分块,
不重写 CE。

## 5. 正确性雷区(必须逐条验)
1. **MTP label 的 roll + CP 边界**:Megatron 用 `roll_tensor(shifts=-1, cp_group, packed_seq_params)`
   右移 label(`gpt_model.py:83/95`)。分块前先按原逻辑 roll,再喂 chunked 路径;chunk 边界必须与 CP
   本地序列布局一致 —— 复用 `loss.py:get_logits_and_tokens_offset_with_cp`,勿自算。
2. **vocab-parallel CE**:每块 CE 走 TP 的 vocab-parallel(跨 TP all-reduce),复用主头分块的语义。
3. **SP gather**:matmul 前 `gather_from_sequence_parallel_region`(`chunked_lm_head_patch.py:73-76` 主头已做,MTP 同样要)。
4. **scale/归一**:`mtp_loss_scale = mtp_loss_scaling_factor/mtp_num_layers`,再 `/num_tokens`;与非分块对齐,否则 loss 量级漂。
5. **数值等价**:小样本上先验 `chunked_mtp_ce ≈ 非分块 mtp_ce`(容差内),且 `ci_utils` 的 MTP grad/loss 校验通过,再信。

## 6. 实现计划 —— 共 6 个 commit
| # | commit | 内容 | 可独立验证 |
|---|---|---|---|
| C1 | `docs(mtp): chunked LM-head 接入 MTP 设计文档` | 本文档 | — |
| C2 | `fix(chunked-lmhead): gate skip on mtp_labels, not mtp_process (P0)` | `chunked_lm_head_patch.py:49` 判据改造 | ✅ 开 MTP 后 compute_log_prob 不再 OOM |
| C3 | `feat(mtp): export MTP hidden from postprocess for chunked CE (P1)` | 新 `chunked_mtp_ce_patch.py`;patch `_postprocess` MTP 分支延迟 CE、导出 hidden;flag 关时 no-op | ✅ flag 关无行为变化;flag 开能拿到 mtp hidden |
| C4 | `feat(mtp): chunked MTP-CE reusing main-head chunk (P2)` | `loss.py` 加 MTP CE 分块(复用 `chunked_logprob_entropy_from_hidden`) | ✅ 单测:分块 CE == 非分块 CE(容差) |
| C5 | `feat(mtp): wire mtp_loss into actor loss + scaling + logging (P3)` | actor 总 loss 接入、scale/归一、`train/mtp_loss` 日志、grad-only-MTP 保持 | ✅ 端到端:MTP 训练不 OOM、loss 量级对、grad 校验过 |
| C6 | `test(mtp): numerical-equivalence + CI + script enablement` | 等价性单测、`ci_utils` MTP grad/loss 校验接线、脚本开关文档 | ✅ CI 绿 |

**落地顺序**:C1 先落(持久化设计)→ C2 单独验(logprob 路径复活)→ C3+C4+C5 连做(MTP 头分块正解)
→ C6 收口。C2 零风险,C3-C5 为核心,C4 因复用主头分块而减负,主要工作量在 C3(从 postprocess 干净导出
MTP hidden)与 C5(scale/日志/grad)。

## 7. 关键文件索引
- `vime/backends/megatron_utils/chunked_lm_head_patch.py` —— 主头分块 monkey-patch(`:45-58` 跳过条件,`:49` = P0 改点)
- `vime/backends/megatron_utils/loss.py` —— `get_log_probs_and_entropy` / `chunked_logprob_entropy_from_hidden` / `get_logits_and_tokens_offset_with_cp`(P2 复用)
- `vime/backends/megatron_utils/model.py` —— `forward_only`(logprob)/`train_one_step`(`:617-618` mtp_kwargs)/`:855-891` MTP loss 读回
- `vime/backends/megatron_utils/ci_utils.py` —— `check_mtp_only_grad` / `check_mtp_loss`
- `Megatron-LM/megatron/core/models/gpt/gpt_model.py` —— `:143` mtp_process、`:490-640` `_postprocess`(`:600` MTP loss、`:624-631` scale)
- `Megatron-LM/.../language_module/language_module.py:129` —— `compute_output_layer_and_language_model_loss`(fused vs 非融合)
- `Megatron-LM/megatron/core/fusions/linear_cross_entropy/blackwell/` —— fused CE(**GPU-only,NPU 不可用**)

## 8. 备选与不采纳
- 移植 blackwell triton fused CE 到 Ascend:成本远高于 P1–P3,不采纳。
- 仅调小 `max-tokens-per-gpu` / 靠 recompute:OOM 根因是 `[seq, vocab]` 全量 logits,与之无关,不解决。
