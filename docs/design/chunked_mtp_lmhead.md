# Chunked LM-head 接入 MTP —— 设计文档

> 推理侧在线 draft 权重同步是另一条独立链路，见
> [`mtp_online_draft_weight_sync_plan.md`](mtp_online_draft_weight_sync_plan.md)。

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
logprob 计算,只是目标 token 换成右移的 `mtp_labels`、并做求和/scale。

### 4.1 实现精化(C3 落地时发现的更优拦截点)
读 `gpt_model.py:568-635` 后确认:MTP 的 roll / loss_mask / `MTPLossAutoScaler.apply`(注入反向)/
`MTPLossLoggingHelper`(记录)**全在 Megatron 的 postprocess 里**,唯一材料化全量 logits 的是
`compute_output_layer_and_language_model_loss` 的非融合分支。→ **不必把 hidden 导出到 vime**(原 P1),
只需**拦截 `LanguageModule.compute_output_layer_and_language_model_loss` 的非融合分支做分块**,
返回同样的逐 token CE `[b,s]`,其余机制原样不动。这把 P1+P2+P3(loss 计算部分)收敛成**单一 patch**
`chunked_mtp_ce_patch.py`:SP 先 gather 一次 → 按 seq 切块 → 每块真 weight matmul(保梯度)+ 复用
`self.compute_language_model_loss`(vocab-parallel CE)→ 拼回 `[b,s]`。fused 分支 / 未开 chunked /
value-head 一律走原实现(no-op)。开关复用 `QWEN36_CHUNK_LMHEAD=1`,块大小 `QWEN36_MTP_CE_CHUNK`(默认 1024)。

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
| C3 | `feat(mtp): chunked MTP-CE via compute_output_layer loss patch (P1+P2)` | 新 `chunked_mtp_ce_patch.py`:拦截 `compute_output_layer_and_language_model_loss` 非融合分支做分块(SP-gather→切块→matmul+vocab-parallel CE→拼回);flag 关 no-op。**单一拦截点收敛原 P1+P2** | ✅ flag 关无行为变化(fused/未开/value-head 走原实现) |
| C4 | `feat(mtp): apply patch at model build + enable in MTP script (P3)` | model 构建处调 `apply_chunked_mtp_ce_patch`(挨着 `apply_chunked_lm_head_patch`);MTP async 脚本置 `QWEN36_CHUNK_LMHEAD=1`;loss/scale/日志沿用 Megatron 既有机制(无需改) | ✅ 端到端:MTP 训练不 OOM |
| C5 | `test(mtp): numerical-equivalence of chunked vs full MTP-CE` | 小样本单测:分块 CE == 非分块 CE(容差),含 TP/CP/SP 组合 | ✅ 数值等价 |
| C6 | `test(mtp): CI grad/loss checks + run notes` | `ci_utils` MTP grad/loss 校验接线、脚本开关文档、端到端 run 记录 | ✅ CI 绿 + 跑通 |

**落地顺序**:C1 先落(持久化设计)→ C2 单独验(logprob 路径复活,零风险)→ C3 核心分块 patch → C4 接线+开关
(端到端不 OOM)→ C5 数值等价(建立正确性)→ C6 收口。因 §4.1 收敛,C3 一件即含原 P1+P2,C4 极薄
(loss 机制留在 Megatron),主要正确性风险在 C5 的等价验证。

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

## 9. 启用与验证状态

### 启用
脚本带以下即自动生效(async math 脚本已含 `--chunked-lm-head`):
```
--chunked-lm-head              # 自动置 QWEN36_CHUNK_LMHEAD=1(arguments.py:170 / model.py:289)
--enable-mtp-training          # 触发 MTP + 本设计的分块 CE(model.py:299 apply_chunked_mtp_ce_patch)
--mtp-num-layers 1             # MTP 层数(enable_mtp_training 必须设,arguments.py:1948)
```
可调:`QWEN36_MTP_CE_CHUNK`(MTP CE 每块 seq 长,默认 1024)。

### 实现状态(commit)
- C1 `2c222e73` 设计文档 ✅
- C2 `4352dcd1` P0 门控(logprob 路径在 MTP 下恢复分块)✅ — 消除 compute_log_prob OOM
- C3 `ee3f86c0` 分块 MTP-CE patch(`chunked_mtp_ce_patch.py`,单一拦截点)✅
- C4 `6ad0cc6e` model 构建处接线(`--enable-mtp-training` 时应用)✅
- C5 `2bc3b162` 数值等价单测(前向+反向,TP=1 参照)✅ 已通过
- C6 `1c2e17de` 启用/验证文档 + 确认 CI 钩子接线
- C7 `35492311` P0-fix:跳过旁路时清 stale `_LM_HEAD_WEIGHT`(修 per-forward guard 引入的
  全局泄漏 → train forward 误把 logits 当 hidden → matmul k 轴崩)
- C8 (本次) P0 精化:`mtp_labels` 跳过**仅在 C3 未生效时**触发。C3 生效后 MTP 走
  `F.linear`(不碰 `output_layer.forward`)→ 旁路对 MTP 无害,train forward 照走旁路让**主头也
  分块**,否则主头返回全量 `[T,V]` logits、`logits.float()` 复现 OOM。已核 postprocess:MTP
  autoscaler 梯度挂在主 hidden、旁路返回该 hidden,反向策略+MTP 梯度均流(端到端仍以 NPU run 为准)。

### CI 钩子(已在 model.py 就位,`--ci-test` + MTP 时跑)
- `check_mtp_only_grad`(model.py:666-669):截断时只有 MTP 参数有非零梯度。
- `check_mtp_loss`(model.py:879-882):MTP loss 在合理界内。

### 端到端验证(C6,需 NPU run,由使用者执行)
用 MTP async 脚本跑一步,检查:
1. **不再 OOM** 在 `compute_output_layer_and_language_model_loss`(MTP 头分块生效);
2. 日志有 `train/mtp_loss`(model.py:891),量级与非分块一致;
3. 加 `--ci-test` 时上面两个 CI 校验通过。
若开 static-kernel/ACL-graph 后仍在解码中途静默崩,先回退那两项(与本设计无关,见 [[run 记录]])。
TP2/CP4/SP 组合下的数值等价以此 run 为准(C5 只覆盖 TP=1 机制层)。

## 10. MTP + Context-Parallel:`_roll_tensor_packed_seq` cu_seqlens 约定冲突(独立正确性修复)

> 与 chunked-lm-head(§0-§9,躲 OOM)**无关**的另一条崩溃线。C6 在 CP>1 下开 MTP 会先在这里崩,
> 不修则永远走不到 §9 的验证。故独立成节、独立 patch。

### 现象
`--enable-mtp-training` + `--context-parallel-size>1` 时,train forward 崩在
`megatron/core/transformer/multi_token_prediction.py` 的 `_roll_tensor_packed_seq`:
`tensor_recv_list[1]` IndexError(cp_size>1 分支,line ~287/289)。CP=1 从不崩。

### 根因(cu_seqlens 约定被 ring fix 重指向)
vime 直接驱动 Megatron core `GPTModel.forward`(非 MindSpeed `gpt_forward_wrapper`),ring attention
需要的 `packed_seq_params` 字段由 vime 在 `data.get_batch` 手工填(**commit `e19530af`,作者 ZhihaoSun,
2026-06-30**,"feat(npu/CP): enable context-parallel ring attention")。ring kernel 要求 `cu_seqlens_q`
是 **CP-local、无前导 0**(带前导 0 → 零长段 → `npu_fusion_attention` 161001),于是该 commit 把
`cu_seqlens_q` 重指向成 ring 约定,同时把 **origin(×cp_size、带前导 0)** 保留在:

| 字段 | 约定 | 消费者 |
|---|---|---|
| `cu_seqlens_q`(被重指向) | CP-local,无前导 0 | ring attention(`ring_context_parallel`) |
| `cu_seqlens_q_padded` | origin,带前导 0 | RoPE `_apply_rotary_pos_emb_thd`(内部 `//cp_size`) |
| `cu_seqlens_gdn` | origin,带前导 0 | GDN `undo_attention_load_balancing_thd` |

该 commit 只盘点了当时三个消费者。**MTP 的 `_roll_tensor_packed_seq` 是第四个消费者**(用户后来才开 MTP):
它硬读 `cu_seqlens_q`,却按 origin 约定处理 —— `for i in range(len(cu)-1)` 需前导 0、`cu[i]//cp_size`
需值是 ×cp_size 的。拿到 ring 约定后:迭代错位 + 二次除法 → `tensor_slice` 退化成极短切片 →
`chunk(2)` 只返回 1 块 → `tensor_recv_list[1]` 越界。**与序列长短无关**(`slice_with_cp` 早已把每条
per-seq pad 到 `2*cp_size` 的倍数,本地切片本应 `2*chunk_size≥2`;补 padding 修不了此约定冲突)。

### 修复(C9)
MTP roll 想要的正是 `cu_seqlens_q_padded`。`mtp_cp_roll_patch.py` 在调原 roll 前把
`packed_seq_params.cu_seqlens_q` 临时换成 `cu_seqlens_q_padded`,调用后 finally 还原(ring attention
在后续 layer forward 才读,不受影响)。换后每段本地切片 = `2*chunk_size`(front+镜像 tail),
`chunk(2)` 出对称两块 → 不崩且数学正确(front/tail 布局与 `slice_with_cp` 一致)。

- 作用域:cp==1 时 data.py 不设 `cu_seqlens_q_padded` → `select_roll_cu_seqlens` 回退 `cu_seqlens_q`
  (本就是 origin)→ no-op(这也解释了为何 CP=1 从不崩)。
- 接线:`model.py` 在 `enable_mtp_training` 分支应用,**独立于 `QWEN36_CHUNK_LMHEAD`**(纯正确性,
  不是 OOM 特性)。不动 Megatron 源码;幂等。
- 单测:`tests/test_mtp_cp_roll_patch.py`(5 例,纯逻辑,不依赖 megatron/distributed)——
  验字段选择(优先 `_padded`、缺失回退)+ swap/restore(正常 & 异常路径)+ 无 `_padded` 时透传。
- CP>1 的 isend/irecv 端到端等价仍由 C6 NPU run 覆盖。

### 文件
- `vime/backends/megatron_utils/mtp_cp_roll_patch.py`(新)
- `tests/test_mtp_cp_roll_patch.py`(新)
- `vime/backends/megatron_utils/model.py`(接线,`enable_mtp_training` 分支)

## 11. `chunked_mtp_ce` 权重取错(前向数值 bug,2026-07-28 修复)

### 现象
开 MTP 训练时 `train/mtp_loss ≈ 14.8`(> 随机基线 `ln(248320)=12.4`),`grad_norm ≈ 10`(不开 MTP 时 ~0.25)。
而推理侧 MTP 投机解码采信率 **73%**(per-position 0.86/0.73/0.60)——头是好的、官方训练过的。训练侧 loss
与推理侧质量严重矛盾 → 训练侧 CE 算错。

### 定位(集群 debug,megatron 无法本地起 TP=2 复现)
CP=1/TP=2/SP,同一 microbatch 三路对照(`QWEN36_MTP_CE_DEBUG=1`):
- **L_clean**(基线 `type(output_layer).forward` 类方法,绕开 chunked_lm_head 的实例级 bypass)= **0.45** ✅
- **L_full/L_chunk**(我的 `gather+F.linear`)= **14.77**,`logits_maxdiff(my-vs-clean)=46`
- `L_full==L_chunk` → **分块机制无罪**;差异在**前向 logits**。
- CP roll 用仿真证死(§10 相邻);SP-gather(`gather_from_sequence_parallel_region`)与基线内部
  `dist_all_gather_func` 是同一种沿 dim0 的 all-gather → gather 无罪。

### 根因
Megatron 基线**非融合分支**(`language_module.py:188-195`)经 `functional_call` 用
**`column_parallel_linear` 模块自身的权重**(detached);`weight` 参数**只在 fused 分支**用。此模型
`shared_embedding_or_output_weight()`(即传进来的 `weight` 参数)与 `output_layer` 的权重**不是同一个**,
我的分块版误用 `weight` 参数做 `F.linear` → 用错矩阵 → logits 全错 → loss 14.8。

### 修复(C10)
`resolve_mtp_ce_weight(weight_arg, col_linear_kwargs, column_parallel_linear)`:复刻基线取权重优先级
`col_linear_kwargs['weight']` → 模块 `weight` → `weight_arg` 兜底,并 `detach`(MTP loss 只训 hidden,
不反传共享输出投影)。bias 一并透传(LM head 通常 None)。修后:`logits_maxdiff=0`、
`L_chunk==L_clean=0.45`、`mtp_loss≈0.3-0.4`、`grad_norm≈0.12`。

### 状态与开关
- 默认**启用**;逃生阀 `QWEN36_MTP_CE_CHUNK_OFF=1` 整体跳过走基线全 logits(仅短序列)。
- `QWEN36_MTP_CE_DEBUG=1`:首个 microbatch 打一次三路对照(需 T 放得下,用 CP=1/T≤4096)。
- 回归单测 `tests/test_chunked_mtp_ce.py::test_resolve_weight_uses_module_not_arg`——TP=1 数值等价测法
  抓不到"权重取错",故专门断言选模块权重而非 `weight` 参数 + detach。

### 文件
- `vime/backends/megatron_utils/chunked_mtp_ce_patch.py`(`resolve_mtp_ce_weight` + `chunked_ce_over_seq` 加 bias + debug 探针)
- `tests/test_chunked_mtp_ce.py`(加权重来源回归)
