# Qwen3.6 YaRN 开发接续上下文

> 状态：历史接续快照；YaRN 功能与本轮最长 300K 验收已于 2026-09-15 完成
> 原始快照时间：2026-09-13
> 当前入口：[YaRN 开启与配置](../README_qwen36_yarn.md)、
> [设计与最终验证记录](qwen36_yarn_rl_training_enablement.md)、
> [外部依赖 patch bundle](patches/qwen36_yarn_20260915/README.md)

本文第 1--13 节保留 2026-09-13 开工前的分支、环境、风险和计划，用于追溯开发顺序，
不再代表当前状态。当前结论以以上三个入口为准；原计划的 524K/1.01M 阶段已在范围收敛后
取消，最终验收上限为 300K。

## 1. 下一 session 应从哪里开始

实际开发工作区是 `/workspace/vime-a3pd-integrated`，不是仍有用户改动的
`/workspace/vime`。当前集成分支为 `dev/a3pd-integrated`；YaRN 开发前已经完成真机验收的功能基线
HEAD 是 `39b02ac9a2d4b20577f6bb8cffe62066268abb1a`，本文档提交只在它上面增加接续资料。

开始写代码前按下面顺序恢复上下文：

1. 阅读本文，确认当前分支、已完成范围和不能触碰的本地配置。
2. 阅读 [YaRN 设计主文档](qwen36_yarn_rl_training_enablement.md)。它包含训练侧阻塞、上游参考、
   改动面、commit 结构和分阶段验证方案，是下一阶段的实现依据。
3. 阅读 [a3-pd topic 集成决策](a3pd_remote_topic_integration_plan.md)，尤其是 §3.8、§6 和 §8，
   理解 YaRN 为什么必须作为独立 topic，不与 MTP、sync rollout 或共卡拓扑重新混合。
4. 阅读 [真机分阶段验证记录](a3pd_remote_bringup_test_20260911.md)，了解 YaRN 开发前已经通过的
   基线、可复用命令参数、日志位置和已知告警。
5. 用只读命令重新确认三个仓库的 HEAD 和 dirty 状态。本文记录的是 2026-09-13 的本地状态，
   不能替代下一 session 开始时的重新检查。

本次同时把原来仅存在于 `/workspace/vime/docs/design/`、且未被 Git 跟踪的 YaRN 设计稿内容迁入
集成分支的 `docs/design/qwen36_yarn_rl_training_enablement.md`，只规范了 Markdown 行尾空格。原文件 SHA256 为
`a98300ced416d0c400d72694c0053ea34bce8242041a1eb9f35d2dc8e0d0dd1f`；后续以集成分支内的副本为
评审和开发入口。

## 2. 项目背景与目标

- 训推框架：VIME。
- agent/rollout 控制面：Polar，路径 `/home/c00937190/polar`。
- 实际模型：Qwen3.6-35B-A3B。
- 上游 SFT 已使用静态 YaRN。RL 必须让 training 与 rollout 使用同一套位置编码参数，避免 actor
  重算 logprob 与 rollout 生成语义不一致。
- 目标参数的当前候选值来自 Qwen 官方示例：`factor=4.0`、
  `original_max_position_embeddings=262144`、`rope_theta=10000000`、
  `partial_rotary_factor=0.25`，vLLM 目标 `max_model_len=1010000`。
- 官方示例只是候选基线。最终唯一事实源必须是实际承接的 SFT checkpoint 配置和 SFT 启动日志。

这次目标不是自行发明 YaRN 算法，而是尽量回移 NVIDIA Megatron-LM 和 Transformers 的成熟实现，
只补 Qwen3.6 partial rotary 所需的最小缺口，并在 VIME 中可靠接通 training/rollout 两侧配置。

## 3. 当前 Git 与工作区状态

### 3.1 VIME 集成分支

```text
path:     /workspace/vime-a3pd-integrated
branch:   dev/a3pd-integrated
verified functional baseline: 39b02ac9a2d4b20577f6bb8cffe62066268abb1a
upstream: bryan/a3-pd
base:     b2503de272cd1addcc5e56e2368af7e34ab047a6
status:   功能基线 ahead 24；本文档作为其后的 docs-only commit
```

这里的结果不是把旧本地 `a3-pd` 整体机械 rebase，而是以远端
`bryan/a3-pd@b2503de2` 为基点，按 topic 重新加回经确认的能力。共同 merge-base 就是
`b2503de2`，所以 24 个本地提交形成线性的远端顶部增量。

`/workspace/vime` 仍是旧 `a3-pd` 工作区，包含用户的启动脚本修改和多份未跟踪文档。不要在那里
继续 YaRN 功能开发，也不要清理、reset 或覆盖其中内容。

### 3.2 Polar 分支

```text
path:     /home/c00937190/polar
branch:   feat/ascendc-rl-t2a
HEAD:     d7e0524200c8cbea7d9864fc5c21d373f20358a0
upstream: mine/feat/ascendc-rl-t2a
base:     9432dd2bec3f270111e59c38a249f36c50951921
status:   ahead 2，另有需要保留的本地配置和运行产物
```

Polar 顶部两个 topic commit：

```text
cd336f4c feat(rollout): cancel surplus tasks for sync oversubscription
d7e05242 feat(rollout): add gateway startup self-check
```

以下 Polar tracked 修改是用户本机部署配置，已经从 stash 恢复并被最后一次真机验证实际使用，不能
删除、覆盖或纳入 YaRN commit：

```text
deploy/ascend_operator/profile.t2a.yaml
deploy/ascend_operator/profile.vime.yaml
operator_runtime_t2a/tools/ascendc_eval_pipeline.sh
```

Polar 下的 `.codegraph/`、profile backup、`_archive/`、input/judge/runtime output 等未跟踪内容同样
不是 YaRN 工作范围。YaRN 当前设计不需要修改 Polar。

### 3.3 推送状态

截至本文落盘前，VIME 相对本地 remote-tracking ref 显示 ahead 24，Polar 显示 ahead 2。因此不能把
“rebase/回迁已完成”误写成“已经推到远端”。下一步是否 push、推到哪个远端分支由用户决定；执行前
先 fetch 并重新核对远端顶部，不能根据本文的旧 remote-tracking ref 盲推。

## 4. a3-pd 回迁已经完成了什么

最终约定和实施结果如下：

| Topic | 已落地结果 |
| --- | --- |
| 远端 Polar/session pool/PD/负载调度 | 保持远端实现和默认行为 |
| 在线 MTP draft 权重同步 | 从 a3-pd 剥离；完整超集留在 `dev/mtp`，不在当前分支保留半套实现 |
| 共卡拓扑 | 用本地 `engine_roles` 单一真源代替远端散落判断，覆盖物理卡和真实 actor rank |
| sync rollout | 作为显式 opt-in 并行路径加回；默认 async/session-pool 不变 |
| sync oversubscription | 支持 factor > 1 的取消、确认和回队，失败时 fail closed |
| durable policy transition | sync zero-inflight 已适配远端事务；async 原有 durable 语义保持 |
| Polar admin ACK | 同时兼容聚合 ACK 和裸 fan-out ACK |
| 显存探针 | `VIME_MEM_PROBE` opt-in，默认关闭 |
| launcher/layout | 在远端顶部重建并用契约测试固定 |
| runtime dump | 从最终工作树删除并补充 ignore |
| YaRN | 刻意未混入，留给下一独立 topic |

VIME 的 24 个增量提交可用下面的命令完整查看：

```bash
git -C /workspace/vime-a3pd-integrated log --reverse --oneline bryan/a3-pd..HEAD
```

其中最后四个文档提交记录了真机 bring-up：

```text
d31ea3b7 test(docs): record staged a3-pd bringup results
bcd624ca test(docs): record durable policy transition retest
c57b3285 test(docs): record restored Polar topology
39b02ac9 test(docs): record durable topology retest
```

## 5. 已完成验证与准确边界

结论：按双方约定的回迁验收范围，除跨节点外的单机验证已经完成。不能把这个结论扩张成“所有可能
组合都测过”。准确结果如下：

| 场景 | 结果 | 关键覆盖 |
| --- | --- | --- |
| async 分卡 | 通过 | actor `4-11`、rollout `12-15`、session-pool、权重同步、1 train step |
| sync factor=1 同构共卡 | 通过 | 16 actor 卡、12 shared rollout 卡、offload/wake、2 train steps |
| 单机异构共卡/专用 engine | 通过 | IPC 与 HCCL 双通路、真实 role/rank、两轮同步 |
| sync factor=1.5 | 通过 | oversubscription、surplus cancel、ACK、requeue、zero in-flight |
| durable 三版本 | 通过 | 三轮 rollout/train，policy epoch `0→1→2→3`，runner exit 0 |
| 连续显存 probe | 通过 | 两个完整 step，无不可解释的 HBM 单调下降 |
| 部分共卡跨节点 | 未测 | 外部机器上的 Polar/session 和 SSH 权限阻塞；用户已明确暂不处理跨机 |

静态/topic 回归已有一次 `183 passed` 记录；最后阶段另跑了相关静态集 `69 passed`。全仓 pytest
受仓库既有 Ray/Transformers stub 和 Typer/Click 环境收集问题影响，不能声称全仓测试 clean。

有一个不属于验收失败的边界需要保留：hybrid async 在 shared 卡上让 Megatron actor 与常驻 vLLM
同时初始化时发生过 OOM。async 基线改用 disaggregated 布局验证；共卡路径由 sync + offload 场景
覆盖。这个结果不影响当前约定，但未来若要支持“async + shared engine 常驻且无 offload”，需要单独
做显存设计，不能引用本次验收作为支持证明。

## 6. durable 日志中的 502 是什么

每次权重更新后有两条版本通知：

1. durable 新主链路携带完整 transition/epoch 信息，负责真正暂停、排空、切权重版本并恢复服务；
2. 旧 `version_span.py` 还会额外发一个只有 `version=N` 的 best-effort 通知。

Polar 当前按新 durable 协议工作，旧通知缺少新协议要求的信息，所以旧请求返回
`502 Bad Gateway`。调用方只记 warning，随后 durable 主链路仍成功完成 `0→1→2→3`，三轮 rollout
和训练均成功。因此这是旧兼容旁路的日志噪声，不是权重同步或训练失败。

YaRN 开发不需要顺手修这个 502。若后续要消除，应作为独立 compatibility/cleanup topic：让旧旁路
在 durable 模式下不再发送，或让 Polar 提供兼容响应。不要把它混入位置编码改动。

最后一次 durable 证据：

```text
log:       /mnt/pipeline-data/train_log/train_a3pd_stage6_durable_retest_20260913b.log
ray temp:  /tmp/ray_s6d13
topology:  /home/c00937190/polar/output/ascend_operator/runs/
           polar_20260913_024611/run_artifacts/effective_topology.yaml
result:    runner_exit=0
```

## 7. YaRN 当前状态和已知结论

YaRN 目前只有设计和前期源码核对，尚无功能代码进入 `dev/a3pd-integrated`，也没有任何真机 YaRN
验收。此前 a3-pd 分阶段测试均明确在 YaRN 关闭状态下完成，只能作为改动前基线。

已经确定的设计结论：

- rollout 侧不需要修改 vllm-ascend 的 YaRN 数学实现；使用单个合并后的 HF overrides 即可。
- rollout 侧仍需 VIME 显式把 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` 传到 Ray actor 创建的 vLLM
  子进程，不能只依赖外层 shell export。
- training 侧本地 Megatron 已有 `YarnRotaryEmbedding` 主体路径，但 CLI/parser、args 到
  `TransformerConfig` 的映射尚未完整接通，应从 NVIDIA Megatron-LM 上游回移。
- Qwen3.6 的 head dimension 为 256，`partial_rotary_factor=0.25`，所以 rotary dimension 必须为
  64、inverse-frequency 数量为 32。当前通用 YaRN 类没有应用 `rotary_percent`，这是最高精度风险。
- 不自行实现 YaRN inverse-frequency/correction/attention-scaling 公式；使用上游实现和
  Transformers oracle。
- MindSpeed 现有通用 wrapper 按 128 维计算，不适合本模型的 64 维目标；当前方案不改 MindSpeed。
- Polar 不感知 RoPE/YaRN，当前方案不改 Polar。
- `CP=8` 不是开启 YaRN 的必要条件，但如果生产 launcher 使用 CP=8，它就是上线前的必测组合。
  CP=1 应作为数值基准，CP=8 用于验证 packed THD/position 切分没有偏移。
- 当前 a3-pd 已剥离在线 MTP draft sync。除非用户重新声明生产任务启用 MTP，否则不要为 YaRN
  顺手扩大 MTP allowlist，也不要把 `dev/mtp` 内容带回。

## 8. YaRN 开工前仍需确认的三个输入

这些不是要求先回答才能阅读代码，但在启用生产 launcher 或判断数值正确前必须有明确答案：

1. 实际 SFT checkpoint 的完整 YaRN fingerprint：`rope_type`、`rope_theta`、
   `partial_rotary_factor`、`factor`、`original_max_position_embeddings`、`beta_fast`、`beta_slow`、
   `mscale`、`mscale_all_dim` 和 correction rounding。
2. `1010000` 的语义：单样本总 token 上限、prompt 上限，还是 prompt + generation 总上限。
3. 生产 RL 是否启用 MTP。按当前分支约定答案应为“不启用”；只有用户改变约定才增加 MTP commit。

若 SFT 的实际 fingerprint 暂时拿不到，可以先实现默认关闭的通用能力、parser/config 测试和 oracle
测试，但不能把 Qwen36 生产开关判定为完成，更不能直接启动 1.01M 长跑。

## 9. 建议实现边界与 commit 结构

开发涉及两个独立 Git 仓库，不能伪装成一个原子 commit。每个 commit 都应保持 feature 关闭时无
行为变化。

### Megatron-LM commit 1

```text
backport(yarn): wire non-MLA YaRN CLI and TransformerConfig
```

- 从 NVIDIA 上游回移参数选项、默认值和 config mapper。
- 加 parser/config 单测。
- 不包含 Qwen-specific dimension fix，不改 VIME launcher。

### Megatron-LM commit 2

```text
fix(yarn): honor rotary_percent and add Transformers parity tests
```

- 让 `YarnRotaryEmbedding` 复用标准 RoPE 的 partial rotary 维度规则。
- 用 Transformers oracle 验证 inverse frequencies、attention scaling、Q/K rotary 前缀和未旋转尾部。
- 验证 checkpoint state key 不变。

### VIME commit 1

```text
feat(qwen36): enable matched YaRN for training and vLLM rollout
```

- 增加默认关闭的 `FEAT_YARN`/等价开关。
- training 侧追加 Megatron YaRN 参数。
- 将 architecture 与 `text_config.rope_parameters` 合并为同一个 HF overrides JSON。
- 显式传递 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` 到真实 vLLM 子进程。
- 对 training/rollout resolved fingerprint 做打印和 fail-fast 一致性检查。
- 加 launcher、environment forwarding 和默认路径不变测试。

### 可选 MTP commit

只有用户确认生产启用 MTP 时才创建；当前约定下省略。

YaRN 主设计文档中预计为 3 个功能 commit（MTP 开启时 4 个），外加 docs/validation 记录。不要把
Megatron 上游回移、Qwen 数值修复、VIME 启用和真机调参压成一个大提交。

## 10. 分阶段验证顺序

不要以“1M 能启动”代替精度验证。下一阶段按以下顺序推进，并持续追加一份新的 YaRN 验证日志文档：

1. 配置单测：parser、TransformerConfig mapper、feature off 无回归、真实子进程 env forwarding。
2. CPU float32 oracle：rotary dim=64、inverse frequencies=32，与目标 Transformers 版本对齐。
3. Q/K apply 测试：前 64 维按 YaRN 旋转，后 192 维保持不变；factor=1 与标准 RoPE 对齐。
4. checkpoint 结构与加载：无新增 trainable key，无 missing/unexpected parameter。
5. NPU `CP=1` 最小 forward/backward，先关闭 fused RoPE。
6. 当前生产 CP 组合（设计基线为 CP=8）最小 microbatch，与 CP=1 建立的容差基线比较。
7. `262144` homo 完整 rollout + train step：只改变位置编码，不跨原始长度，并保持已验证 launcher 的原生容量配置。
8. `~270K`：第一次越过 262144，确认 vLLM 环境变量、无截断、无 position/index 错误。
9. `~524K`：验证 KV cache、训练显存和通信容量。
10. `1010000`：最后做容量和恢复压力验证。
11. 同一 checkpoint、同一 token 的 rollout logprob 与 actor 重算对齐；固定样本跑 5-10 个 RL step，
    观察 TIS、KL、entropy、loss 和 grad norm。

每阶段同时记录 training `seq_length/max_position_embeddings`、rollout token cap、vLLM
`max_model_len` 和 KV cache 配置。`factor=4` 只解决位置频率外推，不会自动解决 1M 序列的显存、KV
容量和吞吐问题。

## 11. 运行环境和操作边界

- 当前机器为 16 卡。已验证的 durable disaggregated 配置使用 actor `4-11`、rollout `12-15`、
  Polar lease pool `0-3`；同构共卡验证曾使用 16 actor + 12 rollout。
- Polar 在宿主机启动，容器内不能负责重启它。可以清理本轮 VIME/Ray/vLLM 进程，但不要误杀 Polar。
- 每次长测试前重新读 Polar 最新 `effective_topology.yaml`，不要假设 endpoint 仍与本文一致。
- 上一轮有效 endpoint 是 rollout `.52:8180`、gateway `.52:8200`、inference router `.52:8001`，
  served-model alias 为 `/home/docker/Qwen3.6-35B-A3B`。
- 用户允许测试期间执行 `ray stop` 或定点 `pkill` VIME/vLLM；这不授权修改或重启宿主 Polar。
- 大日志写入 `/mnt/pipeline-data/train_log/`，Git 文档只记录路径、关键参数、结果和失败归因。
- 不跨机是当前已完成基线；YaRN 初期也先做单机。跨机验证仍是独立后续范围。

## 12. 下一 session 的完成定义

最小开发完成不是“launcher 加了参数”，而是同时满足：

- SFT、Megatron training 和 vLLM rollout 的 resolved YaRN fingerprint 完全一致；
- Qwen3.6 rotary dimension 明确为 64，并通过独立 Transformers oracle；
- feature 关闭时当前已验证的 async/sync/durable 行为不变；
- vLLM 真实 engine 环境收到 long max model length 开关；
- CP=1 和生产 CP 组合通过数值/执行验证；
- 262K homo 和首次跨界约 270K 两阶段通过后，才讨论更长容量测试；
- rollout logprob 与 actor 重算没有超出改动前基线的异常偏差；
- 每个独立 topic 有对应 commit 和测试记录，Polar/MindSpeed 没有无关改动。

## 13. 快速证据索引

- YaRN 完整方案：[qwen36_yarn_rl_training_enablement.md](qwen36_yarn_rl_training_enablement.md)
- a3-pd topic 决策：[a3pd_remote_topic_integration_plan.md](a3pd_remote_topic_integration_plan.md)
- 真机验证明细：[a3pd_remote_bringup_test_20260911.md](a3pd_remote_bringup_test_20260911.md)
- Stage 6 durable 日志：
  `/mnt/pipeline-data/train_log/train_a3pd_stage6_durable_retest_20260913b.log`
- Stage 7 显存日志：
  `/mnt/pipeline-data/train_log/train_a3pd_stage7_mem_probe_20260912.log`
- MTP 完整超集：`/workspace/vime-dev-mtp-run` 的 `dev/mtp@b68eca13`
- Megatron-LM：`/workspace/Megatron-LM`
- MindSpeed：`/workspace/MindSpeed`（当前方案预计零改动）
- vllm-ascend 参考工作区：`/workspace/vllm-ascend-023`（当前方案预计无需核心 YaRN 改动）
- Polar：`/home/c00937190/polar`（当前方案预计零改动）

下一 session 应先以本文和 YaRN 主设计文档校验当前 Git/环境事实，再开始实现 Megatron commit 1；
不要重新讨论或重做已经验收的 a3-pd 回迁 topic。
