# Qwen3.6 MTP 在线 draft 权重同步修复方案

> 状态：代码已实现，待真机验证
> 更新时间：2026-08-31
> 基线：`a18cac5e` (`dev/mtp`)
> 证据日志：`/workspace/vime/logs/sync_factor1p5.log`

## 1. 结论

当前 MTP 接受率为 0 的直接原因不是模型、TP、同步 rollout 函数或 Megatron 的
chunked MTP-CE，而是 **RL 在线权重同步只完整更新了 vLLM target model，没有完整更新
speculative draft model**。

当前单机配置中的六个 rollout 引擎全部 `share: actor`，都走
`UpdateWeightFromTensor` 的 NPU IPC 路径。该路径漏同步 draft，所以所有引擎从第一条统计
开始都是：

```text
Mean acceptance length: 1.00
Accepted: 0 tokens
Avg Draft acceptance rate: 0.0%
```

“共卡”不是 MTP 必然为 0 的条件；当前共卡实现缺少 draft 更新阶段才是原因。

## 2. 证据闭环

### 2.1 同一历史 run 隔离了故障变量

`/mnt/pipeline-data/train_log/train_qwen36_polar_20260826-130952.log` 使用异步 Polar rollout
函数和混合布局：

- `.56` 共卡引擎走 NPU IPC，从最初统计开始接受率为 0；
- `.64` 专用推理引擎走 HCCL，启动初期平均接受长度为 `3.17-3.54`，接受率为
  `72%-85%`；
- `.64` 在后续只更新 target、不更新 draft 后，接受率下降到约 `2.7%-7%`。

这说明 HF MTP 权重和 vLLM MTP 推理能力正常，async/sync rollout 不是决定因素；IPC
路径首次同步就漏了 draft，而 HCCL 路径在在线训练后也必须更新 draft。

### 2.2 当前 run 的参数指纹

`sync_factor1p5.log` 的 `[MTP_SYNC_DEBUG] phase=after` 显示：

```text
received=31333
received_mtp=0
target parameters changed=true
draft MTP parameters changed=false
embedding same_storage=true
lm_head same_storage=false
draft lm_head.weight abs_sum=0.0
```

同一日志还显示 `enable_mtp_training=False`、`mtp_num_layers=None`：该次 trainer 权重流本来
就不含 `mtp.*`，这与 `received_mtp=0` 一致。随后用于重启的配置已经改成
`--mtp-num-layers 1 --enable-mtp-training` 并换用含 MTP 的 torch-dist。本文两阶段方案针对
这个新配置；若训练侧不构建 MTP block，不能靠重放 target-only 权重流恢复 draft block。

Qwen3.5/3.6 MTP proposer 共享 target embedding，但不共享 `lm_head`。target 更新只能透传
共享 embedding；draft 独立的 `lm_head` 和 MTP block 仍是未加载/旧值。

### 2.3 不能只按 `mtp.*` 分流

- target `Qwen3_5ForConditionalGeneration.load_weights()` 跳过 `mtp.*`；
- draft `Qwen3_5MTP.load_weights()` 把 `mtp.*` 映射到内部 `model.*`，同时接收全局
  `embed_tokens` 和 `lm_head`，忽略其余 target 权重。

所以 draft 需要的是**完整 checkpoint-format 权重流**，而不只是 `mtp.*` 子集。单次接收
中按前缀分流仍会漏掉独立 `lm_head`，不构成正确修复。

## 3. 修复语义

采用历史提交 `99a3f2c9 feat: sync MTP draft weights online (#351)` 的两阶段语义，并适配
当前 IPC/HCCL 混合分发：

```text
pause generation + flush cache

phase 1: target
  start_weight_update(is_checkpoint_format=True)
  replay complete HF weight stream
  target.load_weights(stream)       # loader 自动跳过 mtp.*
  finish_weight_update()

phase 2: draft
  start_draft_weight_update()
  replay the same complete HF weight stream
  draft.load_weights(stream)        # loader 取 mtp.* + embedding + lm_head
  finish_weight_update()

publish policy version + resume generation
```

两阶段必须处于同一个 pause 窗口。phase 2 失败时 fail closed：不发布 policy version，不
恢复推理。首次 rollout 也必须等首次 target+draft 回注全部完成。

## 4. 最小实现

### 4.1 复用现有 vLLM collective RPC

当前 VIME 已通过 `/collective_rpc` 调用 worker extension 的 `update_weights_chunk`。新增
`start_draft_weight_update` 也复用该入口，不增加 vLLM HTTP protocol/router patch：

```python
def start_draft_weight_update(self) -> dict:
    return self._make_request(
        "collective_rpc",
        {"method": "start_draft_weight_update", "kwargs": {}},
    )
```

这样改动完整落在 VIME `dev/mtp`，不依赖本机 `/workspace/vllm-023` 和
`/workspace/vllm-ascend-023` 的工作树状态。

### 4.2 worker extension 的 selected-model 状态机

扩展现有 `_VLLMHijack`：

1. target start 选择 `model_runner.model` 和 target config；
2. draft start 选择 `model_runner.drafter.model` 和 `draft_model_config`；
3. checkpoint-format start/finish 对当前选中模型执行 layerwise reload；
4. IPC `update_weights_chunk` 加载当前选中模型；
5. HCCL 原生 `update_weights` 通过短生命周期 model/config swap 复用 vllm-ascend 原实现；
6. target/draft 分别恢复 `weight_loader` 参数属性；
7. 无 drafter/config、嵌套 start、非法 update/finish 立即报错；finish 后清空 selected state。

MTP 训练的 rollout 引擎，无论共卡还是专用，都加载同一个 worker extension。非 MTP 配置
保持当前加载范围和单阶段行为。

### 4.3 trainer 两阶段重放

统一判据：

```text
sync_mtp_draft =
    args.enable_mtp_training
    and args.vllm_speculative_config.method == "mtp"
```

真机启动参数必须同时看到 `enable_mtp_training=True` 和 `mtp_num_layers=1`；否则第二阶段
不会启动，并应把该 run 判为配置错误而不是 MTP 修复验证。

覆盖两类更新器：

- `UpdateWeightFromTensor`：纯 IPC 和 IPC+HCCL 混合布局；
- `UpdateWeightFromDistributed`：纯 HCCL/NCCL 布局。

混合布局每个 phase 都同时覆盖两半：

```text
target: IPC target session + HCCL target sessions -> send once
draft:  IPC draft session  + HCCL draft sessions  -> send once
```

每个 phase 重新调用 `get_hf_weight_chunks(megatron_local_weights)`，不能复用已消费的
generator 或上一阶段 IPC handle。一个 target+draft 事务只增加一次 `weight_version`。

## 5. 测试与验收

CPU 测试覆盖：

1. MTP 开启时严格执行 target start/send/finish、draft start/send/finish；
2. 两阶段收到相同的完整名称集合；
3. 纯 IPC、纯 distributed、混合拓扑均更新 draft；
4. 非 MTP 只发送一次；
5. draft 失败时不 resume、不发布版本；
6. worker selected-model 状态机及非法顺序；
7. draft loader 能看到 `mtp.*`、`embed_tokens` 和独立 `lm_head`。

真机首次同步后必须证明：

- target 参数已更新；
- draft MTP block 和独立 `lm_head` 为有限非零值且与 trainer 固定切片数值一致；
- 接受率不再持续为 0，稳态平均接受长度恢复到历史/单跑约 3 的量级；
- 连续至少三个 policy version，target/draft 同版本更新；
- 混合拓扑的 IPC/HCCL 引擎均正常。

接受率恢复只证明 MTP 权重一致；stop token/tool parser 的文本正确性另行验收。

## 6. 风险

- 完整权重流发送两次，约 65 GiB/255 桶的同步窗口理论上接近翻倍；第一版先保证正确性。
- 两阶段串行并逐 chunk 释放临时 tensor/IPC handle，峰值不应变成两份全模型。
- draft 必须经过 checkpoint-format reload，不能退回手工 `param.copy_` 绕过 MoE/packed
  权重处理。

## 7. 预期提交

| 顺序 | commit | 内容 |
|---|---|---|
| C1 | `docs(mtp): design online draft weight synchronization` | 根因、方案、验收 |
| C2 | `feat(mtp): add draft weight update session` | actor collective RPC + worker selected-model 状态机 |
| C3 | `feat(weight-sync): replay online weights to MTP draft` | Tensor/Distributed/混合两阶段重放 |
| C4 | `test(mtp): verify online target and draft weight sync` | 生命周期、异常和 loader 数值测试 |
| C5 | `fix(mtp): validate colocated and hybrid acceptance after sync` | 真机记录及必要小修 |

C1-C4 在本分支开发；C5 只能在真机 run 完成后提交。

## 8. 不在本修复范围

- Polar `stop_token_ids` 和工具格式；
- `enforce_eager` 参数层级；
- Megatron chunked MTP-CE/CP roll；
- 超订、session cancel 和 sample 回灌；
- 接受率恢复后的吞吐 A/B。
