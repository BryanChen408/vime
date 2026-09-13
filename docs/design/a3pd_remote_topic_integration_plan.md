# a3-pd 远端基线 Topic 集成决策

> 状态：实施中，回合 0 基线已完成
> 日期：2026-09-11
> 目标基线：`bryan/a3-pd` (`b2503de272cd`)
> 本地来源：`a3-pd` (`688eff810d61`)
> MTP 独立分支：`dev/mtp` (`b68eca133cbf`)
> 共同分叉点：`bd54fbc37101`

## 0. 回合 0 基线

- 隔离分支：`dev/a3pd-integrated`
- 隔离工作区：`/workspace/vime-a3pd-integrated`
- 基点及 upstream：`bryan/a3-pd@b2503de272cd`
- Polar 调度和 policy boundary：`38 passed`
- PD proxy 和容量路由：`29 passed`
- weight-sync 组合：`102 passed, 19 failed`
- 19 个失败中，18 个来自生产 MTP draft 同步已删除但对应测试仍残留；另 1 个来自
  `b2503de2` 误回退独立的 Tensor transpose alias 修复。
- 已跟踪 shell 脚本全部通过 `bash -n`。
- 远端 runtime dump：22 个已跟踪文件，共 268,593,138 bytes。
- 回合 0 结束时隔离工作区 clean，原 `/workspace/vime` 的脏工作区状态未变化。

实际 `scripts/start_sync_hybrid.sh` 启动 fingerprint：

| 项目 | 基线值 |
| --- | --- |
| actor / rollout | `1 x 16 NPU` / `24 NPU` |
| rollout engine | `4 NPU/engine` |
| 并行度 | `TP=1, PP=2, CP=8, EP=8` |
| rollout/sample/global batch | `16/8/128` |
| rollout 数 | `100` |
| sequence/context/model length | `262144/262144/262144` |
| checkpoint interval | `5` |
| rollout timeout | `21600` |
| learning rate | `3e-6` |
| rollout function | `generate_rollout_polar_async` |
| scheduler | `session_pool` |
| durable policy transition | enabled |
| dataset | `op_assets_drkernel_2k_difficulty_filtered_v3` |
| main launcher SHA256 | `999ed4fb93235de3ca9c92ce1d5fa48a05e9a4fb8502a35c292ac720005bd7b8` |
| hybrid wrapper SHA256 | `85360267af159d694ae71b2012dbbfcc3032c4f92b98f4886ebe325bdde20df1` |

## 1. 目标和原则

本次不把本地 `a3-pd` 整体 rebase 后逐个解决历史冲突，而是以远端
`bryan/a3-pd` 顶部为功能基线，按 topic 判断哪些本地能力需要重新加回。

执行原则：

1. 远端新增能力默认保留，不用本地旧文件整文件覆盖。
2. 本地能力按功能边界定点移植，不按原提交时间线机械 replay。
3. MTP、共卡拓扑、rollout 调度分别处理，不能因共享文件而混成一个提交。
4. 启动配置最后重建；学习率、batch、数据集等实验语义不在代码合并时顺手改变。
5. 每个 topic 独立提交、独立测试、可单独回滚。
6. 当前工作区已有的未提交和未跟踪文件属于用户，不用 `reset`、整文件 checkout 或
   `git add -A` 处理。

## 2. 决策总表

| Topic | 最终决策 | 默认行为 |
| --- | --- | --- |
| 远端 Polar/session pool/负载调度 | 以远端为准 | 保持远端行为 |
| 在线 MTP draft 权重同步 | 从 `a3-pd` 剥离，完整实现只留 `dev/mtp` | 不启用、不残留半套测试或入口 |
| 共卡拓扑 | 本地 `engine_roles` 替换远端散落判据 | 简单拓扑行为不变，支持部分共卡和单机异构 |
| Polar sync rollout | 本地完整能力作为 opt-in 并行路径加回 | 默认仍走远端 async/session pool |
| Polar durable policy transition | 保留远端，实现 sync zero-inflight 兼容 | async 语义不变，sync 显式选择 |
| Polar admin ACK 解析 | 加回本地修复 | 兼容聚合响应和裸 fan-out 响应 |
| 共卡显存探针 | 加回本地实现 | `VIME_MEM_PROBE` 默认关闭 |
| 启动脚本和资源布局 | 在最终代码上重建 | 远端参数值先冻结，不借合并调参 |
| 远端运行产物 | 从最终树删除 | 不保留 dump、`.o` 和 exception blob |
| YaRN | 后续独立 topic | 不与本次分叉集成混合 |

远端已经存在且不需要“加回”的能力包括：

- `4f85e394`：Polar 持久化权重边界和运行观测；
- `0df78aaa`、`25971518`：KV 负载均衡和容量加权 session 配额；
- `3f9238f6`、`e2a98f25`：任务轮询限制和 HTTP client 事件循环隔离；
- `4f2f68aa`：PD engine 指标聚合；
- `9885ea3e`：独立 HF rollout eval launcher；
- `0b421d24`、`54270565`：失败轨迹 credit 和 t3a 主链过滤；
- `3ebce883`：policy boundary 与 scheduler cleanup 回归测试。

这些提交作为远端基线能力保留。本地 topic 移植后的测试必须证明没有把它们覆盖或降级。

## 3. Topic 详细约定

### 3.1 在线 MTP draft 权重同步

#### 范围

这里的“MTP 剥离”特指分叉后新增的在线 draft 权重同步 topic：

- target/draft 两阶段在线权重重放；
- `start_draft_weight_update` 会话；
- MTP 在线同步专用测试、设计文档和启动开关；
- 当前工作区中按 `mtp.*` 名称临时路由到 drafter 的未提交原型和 debug 代码。

不删除共同基线中既有、与本次在线同步 topic 无关的 MTP 训练基础，例如
`chunked_mtp_ce_patch.py`、`mtp_cp_roll_patch.py` 及其原有测试。是否继续保留这些共同基线
能力不由本次分支整理决定。

#### 依据

`dev/mtp` 上 8 个提交是完整实现。与远端曾经存在的 8 个 MTP 提交比较，7 个补丁完全
等价，启动器提交在 `dev/mtp` 上还额外覆盖 single52。当前未提交的单阶段名称路由会漏掉
draft 独立 `lm_head` 和所需 embedding，不能作为完整实现保留。

远端 `b2503de2` 已回滚主要生产逻辑，但仍残留：

- `docs/design/mtp_online_draft_weight_sync_plan.md`；
- weight-sync/vLLM 单测中的 `start_draft_weight_update` 和 `_sync_mtp_draft_enabled`；
- 共享启动器中的 `FEAT_MTP`、`FEAT_MTP_TRAIN` 和 `VLLM_SPEC_CONFIG`。

#### 实施约定

1. 从最终 `a3-pd` 删除上述在线同步 topic 的残留。
2. 丢弃当前工作区 Python/test 中未提交的 MTP 临时路由和 `VIME_MTP_SYNC_DEBUG`。
3. 启动脚本按 hunk 删除 MTP 参数，保留同文件中与 profile、batch、显存探针等相关修改。
4. 将来需要在线 MTP 时，从 `dev/mtp` 整体移植，不从本次清理结果中恢复零散代码。

#### 验收

- 生产代码、单测和 launcher 不再引用 `start_draft_weight_update`、
  `_sync_mtp_draft_enabled` 或 `MTP-WEIGHT-SYNC`。
- 非 MTP 权重同步测试正常。
- 共同基线已有的离线/训练侧 MTP 工具没有被误删。

### 3.2 共卡拓扑

#### 决策

以本地 `vime/ray/engine_roles.py` 为共卡判断的单一真源，替换远端以下近似判据：

1. vLLM 启动侧按“是否位于 actor 节点”选择 `npu_ipc`/`nccl`；
2. weight-sync 按 rollout slot 和 actor GPU 总数切 IPC/HCCL；
3. offload 按 group offset 和 share GPU 数量判断 sleep 范围；
4. IPC gather 直接把 rollout slot 当 actor rank。

本地定义以实际物理放置 `(node, device)` 为准，并为每个 engine 给出：

- engine index；
- rollout GPU slot；
- 物理卡集合；
- 是否与 actor 完全共卡；
- 对应的真实 actor ranks。

#### 来源提交

- `2d02adb0`：拓扑问题与设计记录；
- `55556fae`：`engine_roles` 单一真源；
- `b79cf153`：vLLM 启动和 rollout 消费方；
- `33a97d55`：weight-sync 共卡计数；
- `d5e33a37`：IPC gather 使用真实 actor ranks。

#### 移植规则

1. 保留远端 `rollout.py` 中 policy boundary、PD、调度、指标等非拓扑逻辑。
2. 只替换共卡分类、backend 选择、offload 集合和 gather rank 计算。
3. 不整文件覆盖 `vime/ray/rollout.py`、`vllm_engine.py` 或
   `update_weight_from_tensor.py`。
4. `--colocate`、位置式分离部署和 `--resource-layout` 都必须由同一解析器覆盖。
5. 若保留旧参数兼容 fallback，必须有简单拓扑对拍测试，且不能在合法 layout 上静默回退。

#### 验收矩阵

| 拓扑 | 预期 |
| --- | --- |
| 全共卡 | 所有 engine 走 IPC，并参与 sleep |
| 全分离 | 所有 engine 走 HCCL/NCCL，不参与共卡 sleep |
| 跨机混合 | share 前缀走 IPC，专用段走 HCCL |
| 单机异构 | 同一节点上的专用卡不能因“同节点”被误判为 IPC |
| 部分共卡 | share 少于 actor 总卡数时，专用 engine 不能被 actor GPU 数量误判 |
| slot != actor rank | IPC gather 使用实际 actor rank，GPU UUID 路由正确 |

### 3.3 Polar sync rollout function

#### 决策

保留远端 `generate_rollout_polar_async`、`AsyncPolarRolloutWorker`、session pool、KV 负载
均衡和容量配额，不修改其默认选择。将本地完整同步实现作为并行入口加回：

```text
默认：generate_rollout_polar_async -> 远端 async/session_pool
显式：generate_rollout_polar_sync  -> 本地 one-shot sync
```

启动器使用 `FEAT_SYNC_ROLLOUT` 选择 `--rollout-function-path`，默认值为 `0`。

#### 已有本地能力

- 同步入口不创建或复用全局 async worker；
- 返回时所有已提交任务已终态，或取消已获 Polar 确认；
- rejected group 有界 top-up，避免不足 batch 或无限重试；
- `factor=1.0` 严格同步；
- `factor>1.0` 超订、最快 group 选择、未选 group 取消和回队；
- 超订要求 data source 真正支持 `add_samples()`；
- 超订上限 `1.5`，取消/回队失败时 fail closed；
- 同步延迟和 tail ratio 指标；
- TIS 独立控制，不因 staleness 为 0 自动关闭；
- eval 继续复用现有 one-shot eval 路径。

当前本地 CPU 测试基线为 `29 passed`。

#### 与远端 durable transition 的新增适配

这不是重写 sync rollout，而是补齐分叉后远端新增的协议：

1. async 路径继续要求 `scheduler_mode=session_pool`，行为不变。
2. sync 路径允许以 one-shot zero-inflight 契约替代本地 session-pool cutoff。
3. sync task metadata 必须携带远端要求的 policy namespace 和 epoch。
4. sync 返回失败、取消未确认或回队失败时，不得进入 train/weight update。
5. `prepare_policy_update`、engine abort、weight update、KV restore、
   `finish_policy_update` 的远端事务顺序不变。
6. durable transition 失败时保持 admission closed；sync 不增加降级放行路径。

sync 使用同一模块的 policy hook 是既有设计，不应为了隔离而绕过这些 hook。隔离的是 rollout
调度状态，不是权重版本事务。

#### 启用矩阵

| Rollout | Durable transition | 预期 |
| --- | --- | --- |
| async/session_pool | off | 保持远端 legacy 行为 |
| async/session_pool | on | 保持远端 durable 行为 |
| sync factor=1.0 | off | 普通边界，返回时零 in-flight |
| sync factor=1.0 | on | zero-inflight 证明 + durable gateway/engine 事务 |
| sync factor>1.0 | off/on | 取消全部获确认后才允许返回训练 |

### 3.4 Polar admin ACK 解析

#### 决策

加回本地 `a18cac5e` 的行为和测试，但在远端最新控制面上重新落补丁，不覆盖远端文件。

该修复让 admin 请求同时接受：

- 聚合后的标准 acknowledgement；
- 网关返回的裸 fan-out acknowledgement。

远端 durable transition 没有让 legacy 路径消失，并且 durable 默认仍可关闭，所以该修复仍是
有效的兼容性修复，而不是被远端替代的旧代码。

#### 验收

- 所有网关成功时返回成功；
- 任一网关拒绝、缺失或版本不一致时失败；
- 不能把 HTTP 200 直接等价为业务成功；
- async legacy、async durable 和 sync 三种调用方都使用同一解析规则。

### 3.5 共卡显存探针

#### 决策

加回本地 4 个显存探针提交的能力，并整理为默认关闭的 observability topic：

- `3482f877`：探针从 Megatron model 模块解耦；
- `40a85876`：train driver 通过 Ray 读取 actor 显存；
- `3e745c56`：在共卡交接点埋点；
- `43ed1dce`：门控和交接点测试。

#### 约束

1. `VIME_MEM_PROBE` 未开启时不增加 Ray RPC，不改变训练控制流。
2. 只记录显存，不依据观测结果自动修改 util、batch 或 offload 策略。
3. 探针插入远端最新 `train.py` 的 durable transaction `try/fail-closed` 结构中，不改变异常处理。
4. 至少覆盖 rollout onload、rollout 完成、engine offload、train 完成、train offload、
   weight shell onload、KV onload 等交接点。

#### 验收

- 关闭探针时调用次数为 0；
- 开启时每个预期交接点都有稳定 tag；
- 两个以上 rollout step 中 HBM 不应出现无法解释的单调增长；
- 探针异常不得吞掉原训练异常，也不得改变 policy transition 状态。

### 3.6 启动脚本和资源布局

#### 决策

不 cherry-pick 本地脚本演进历史，也不把远端 `b2503de2` 的全部 hunk 解释为 MTP 删除。
核心代码合并后，以远端脚本为底重建最终 launcher。

最终脚本必须明确表达：

- async 是默认 rollout function；sync 由 `FEAT_SYNC_ROLLOUT=1` 显式启用；
- sync oversubscribe 默认 `1.0`，大于 1 显式设置且最大 1.5；
- TIS 独立开关，默认策略不随 sync/async 自动改变；
- MTP 在线同步参数全部移除；
- topology 使用最终 `engine_roles` 可解释的资源布局；
- `VIME_MEM_PROBE` 保持 opt-in；
- durable transition 与 rollout 模式的组合在启动前校验；
- YaRN 暂不混入本次脚本提交。

#### 远端当前实验参数处置

以下值来自远端顶部，但不是“MTP 删除”的自然结果：

| 参数 | 远端当前值 | 本次处置 |
| --- | --- | --- |
| `num_gpus_per_engine` | `4` | 先保留，按最终物理拓扑校验 |
| `save_interval` | `5` | 先保留，不在集成提交中调参 |
| rollout timeout | `21600` | 先保留，真机验证超时行为 |
| learning rate | `3e-6` | 先保留，需实验 owner 明确认领 |
| rollout/sample/global batch | `16/8/128` | 先保留，按目标机器容量做启动前 gate |
| 数据集 | `op_assets_drkernel_2k_difficulty_filtered_v3` | 先保留，需实验 owner 明确认领 |

“先保留”表示以远端为基线不擅自回退，并不表示这些值已由代码合并证明正确。最终真机任务启动前
应输出配置 fingerprint，由实验 owner 确认。

#### 已知遗留：稀疏物理卡布局 padding

`49bf6cc8` 修改 `_build_layout_bundles`，按每个节点 YAML 中最大的物理卡号补齐
`0..max_device` bundle。其出发点是假定 Ray 会把 placement group 内 12 个稀疏设备压缩编号为
`0..11`；但 `dev/sync-rollout` 的 2026-08-28 T4 实机日志与该假定冲突：launcher 以
`NPUS_PER_NODE=12`、`ASCEND_RT_VISIBLE_DEVICES=4..15` 启动 Ray，12-bundle placement group
实际探测并成功映射了物理卡 `4..15`，随后越过 placement、拉起全部 rollout engine 和训练 actor，
直到权重同步阶段才暴露另一个协议问题。

当前先保留 `49bf6cc8`，不在本轮完整性修复中改共享 placement 代码，也不通过把 single52
launcher 改成 16 卡或修改 Polar 卡池掩盖矛盾。因此异构 single52 真机回归暂时受此遗留阻塞：
现实现会为 YAML 最大卡号 15 申请 16 个 bundle，而恢复后的已验证 launcher 只向 Ray 注册 12 卡。
后续应单独复现实机的 Ray 设备编号语义，再定点撤销 padding 或实现兼容映射；验收必须保持
sync launcher 的 `12 + 4..15` 外部契约不变。

### 3.7 远端运行产物和配置清理

#### 运行产物

`131ea73c` 带入了 `extra-info/data-dump/{0,1}` 下的 exception dump、`.o` 和 JSON 产物，
约 268 MB。最终工作树删除这些文件，并通过 ignore 规则防止再次误提交。

如果只在远端顶部追加删除 commit，历史 blob 仍存在，clone 体积不会下降。是否重写远端历史以
彻底移除 blob 是独立仓库治理动作，本方案不默认执行历史改写。

#### 配置清理

`131ea73c` 和 `b2503de2` 的脚本/config hunk 按上一节逐项保留或重建，不与 dump 删除、MTP
剥离混在同一个提交中。配置 comment 必须与实际 engine TP、数量和布局一致。

### 3.8 YaRN

YaRN 保持独立 topic，依据 `docs/design/qwen36_yarn_rl_training_enablement.md` 实施。它不阻塞
本次分支 topic 整理，也不能借本次 launcher 重建提前混入。

## 4. 推荐实施顺序

```text
远端 bryan/a3-pd
  |
  +-- 清理运行产物
  +-- 清理在线 MTP topic 残留
  |
  +-- 共卡 topology 单一真源
  +-- Polar ACK 兼容
  +-- sync rollout + durable transition 适配
  +-- 显存探针
  |
  +-- 最终重建 launcher/resource layout
  +-- 全量 CPU/契约测试
  +-- 分阶段真机验证
```

MTP 清理必须先于 topology，因为两者都会触及 weight-sync 文件。启动脚本必须最后处理，避免
每个功能 topic 都重复解决同一组脚本冲突。

## 5. 预期 Commit 结构

推荐 17 个 commit，包含本文档；评审时可合并相邻测试提交，但不能跨 topic 混合：

| # | 预期 commit | 内容 |
| ---: | --- | --- |
| 1 | `docs(integration): record a3-pd topic integration decisions` | 本文档 |
| 2 | `chore(repo): remove accidental Ascend runtime dumps` | 删除 dump/`.o`，补 ignore |
| 3 | `chore(mtp): remove online draft sync remnants from a3-pd` | 清生产残留、测试、文档和 launcher |
| 4 | `feat(ray): add engine role topology source of truth` | `engine_roles` |
| 5 | `fix(colocate): route engine startup offload and weight sync by role` | 替换远端散落判据和 actor rank 映射 |
| 6 | `test(colocate): cover homogeneous heterogeneous and partial sharing` | topology 回归矩阵 |
| 7 | `fix(polar): accept raw fan-out admin acknowledgements` | ACK 解析和测试 |
| 8 | `refactor(bridge): expose shared rollout acceptance checks` | sync/async 共用纯函数，async 行为不变 |
| 9 | `feat(rollout): add opt-in strict synchronous Polar rollout` | one-shot sync，factor=1 |
| 10 | `test(rollout): cover sync isolation and async non-regression` | 严格同步和默认路径测试 |
| 11 | `feat(rollout): add opt-in sync cancellation and requeue` | factor>1 超订能力 |
| 12 | `fix(policy): support durable transition for zero-inflight sync rollout` | namespace、epoch、fail-closed 适配 |
| 13 | `refactor(mem): expose NPU memory probes outside model setup` | 探针解耦和 Ray plumbing |
| 14 | `feat(mem): add opt-in colocate handoff probes` | `train.py` 交接点 |
| 15 | `test(mem): cover probe gating and handoff points` | CPU 测试 |
| 16 | `feat(scripts): rebuild a3-pd launchers and resource layouts` | 最终配置和模式开关 |
| 17 | `test(scripts): pin launcher mode and configuration contracts` | shell/AST 契约测试 |

## 6. 验证方案

### 6.1 每个 topic 的 CPU/静态验证

建议在依赖可用的 VIME 环境中单文件执行，避免仓库既有的 pytest 全目录收集问题：

```bash
export PYTHONPATH="/usr/local/lib/python3.11/site-packages:/workspace/Megatron-LM:$PWD:${PYTHONPATH:-}"

python -m pytest tests/test_engine_roles.py -q -o addopts=""
python -m pytest tests/test_weight_sync_hybrid.py -q -o addopts=""
python -m pytest tests/test_version_span_ack.py -q -o addopts=""
python -m pytest tests/test_colocate_memory_probe.py -q -o addopts=""
python -m pytest tests/test_npu_memory_handoff.py -q -o addopts=""
python -m pytest vime_bridge/tests/test_vime_polar_sync_rollout_cpu.py -q -o addopts=""
python -m pytest vime_bridge/tests/test_vime_polar_scheduler_cpu.py -q -o addopts=""
python -m pytest vime_bridge/tests/test_policy_update_pause_contract_cpu.py -q -o addopts=""
python -m pytest vime_bridge/tests/test_version_span_contract_cpu.py -q -o addopts=""
python -m pytest tests/unit/test_polar_train_boundary_order.py -q -o addopts=""

bash -n scripts/run-qwen36-35b-polar-multi-pd.sh
bash -n scripts/start_sync_hybrid.sh
```

MTP 和运行产物清理检查：

```bash
git grep -n -E 'start_draft_weight_update|sync_mtp_draft|MTP-WEIGHT-SYNC'
git ls-files 'extra-info/data-dump/**'
```

两条命令在最终分支中都应无输出。共同基线原有 MTP 文件不在这个禁止列表中。

### 6.2 模式契约验证

必须覆盖：

1. 不设置 `FEAT_SYNC_ROLLOUT` 时仍加载远端 `generate_rollout_polar_async`。
2. async 路径仍创建 persistent worker，session-pool 测试结果不变。
3. sync 路径不访问 `_global_async_worker`、ready queue 或 deferred queue。
4. sync factor=1 返回时无 pending task。
5. sync factor>1 只在 Polar 返回取消成功且 group 回队成功后返回。
6. sync/durable task metadata 带正确 policy namespace 和 epoch。
7. 任一 engine version 不一致、ACK 缺失或 transition commit 失败时保持 fail closed。

### 6.3 真机分阶段验证

按风险递增执行，不直接上长跑：

1. async 分卡回归：证明远端默认路径未受影响。
2. 单机全共卡、sync factor=1、`NUM_ROLLOUT=2`：证明 sleep/weight/KV 交接闭环。
3. 单机异构：证明同节点专用 engine 走 HCCL 而非 IPC。
4. 部分共卡跨机：证明 share 少于 actor 时仍正确分流。
5. sync factor=1.25 或 1.5 canary：证明 cancel acknowledgement、回队和零 in-flight。
6. durable transition canary：连续至少 3 个 policy version，无混合版本、无旧任务复活。
7. 开启 `VIME_MEM_PROBE=1` 做至少 2 个完整 step，确认 HBM 没有不可解释的单调增长。

关键日志/指标：

- 每台 engine 的 `placement`、`colocated`、backend 和 actor ranks；
- `polar/sync/accepted_groups`、`aborted_groups`、`aborted_sessions`、`requeued_groups`；
- policy namespace、from/to epoch 和所有 engine weight version；
- rollout 返回、engine sleep、weight onload、KV onload 各交接点显存；
- async 回归的 session-pool active/open/ready/owned group 指标。

## 7. 完成定义

只有同时满足以下条件，topic 集成才算完成：

- 最终分支以远端功能为基线，不丢远端 Polar/PD/调度能力；
- 在线 MTP draft sync 在 `a3-pd` 中无半套残留，完整实现仍可在 `dev/mtp` 追溯；
- 所有共卡消费者只依据同一个 engine-role 结果；
- async 是默认路径且非回归，sync 是显式 opt-in；
- sync 严格模式、超订模式和 durable transition 组合均 fail closed；
- ACK、显存探针、launcher 和资源布局均有独立测试；
- 运行 dump 不在最终工作树；
- 实验参数已有 fingerprint 和 owner 确认记录；
- YaRN 仍作为后续独立 topic，没有混入本次提交。

## 8. 参考源

### VIME 本地

- `docs/design/a3pd_colocate_migration_plan.md`
- `docs/design/colocate_topology_robustness_plan.md`
- `dev/mtp:docs/design/mtp_online_draft_weight_sync_plan.md`
- `docs/design/colocate_bringup_handoff_20260829.md`
- `docs/design/qwen36_yarn_rl_training_enablement.md`
- `vime/ray/engine_roles.py`
- `vime_bridge/rollout.py`
- `vime_bridge/version_span.py`
- `vime/backends/megatron_utils/update_weight/update_weight_from_tensor.py`

### Polar 本地依赖

- `/home/c00937190/polar/src/polar/rollout/manager.py`
- `/home/c00937190/polar/src/polar/rollout/server.py`
- `/home/c00937190/polar/src/polar/rollout/pipeline.py`
- `/home/c00937190/polar/tests/rollout/test_task_cancellation.py`
- `/home/c00937190/polar/tests/rollout/test_manager_cancel.py`

### Git 参考

- 远端功能基线：`bryan/a3-pd` at `b2503de272cd`
- 本地 sync/topology 来源：`a3-pd` at `688eff810d61`
- MTP 完整实现：`dev/mtp` at `b68eca133cbf`
- 在线 MTP 参考提交范围：`ccf4dfb4^..b68eca13`
- 本地 sync rollout 核心：`58314ff3`、`847c70c6`、`3a4b6861`、`e2d3bba3`、
  `688eff81`
- 本地 topology 核心：`55556fae`、`b79cf153`、`33a97d55`、`d5e33a37`
- 本地 ACK 修复：`a18cac5e`
- 本地显存探针：`3482f877`、`40a85876`、`3e745c56`、`43ed1dce`
