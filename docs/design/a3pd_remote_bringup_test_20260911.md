# a3-pd 回迁特性真机分阶段验证记录

状态：阶段性完成（Stage 6 durable bootstrap 已通过；完整 E2E 受 Polar gateway 后端配置阻塞）
日期：2026-09-11--2026-09-12（容器时间；Polar run 目录使用宿主机时间）  
代码：`dev/a3pd-integrated`，基线 `bryan/a3-pd@b2503de2`，当前 HEAD `d31ea3b7`

## 1. 验证范围

按 `a3pd_remote_topic_integration_plan.md` §6.3，从低风险到高风险验证远端 a3-pd 基线上的回迁 topic：

1. async 分卡回归；
2. 单机共卡 sync，factor=1；
3. 单机异构共卡/专用 engine；
4. 部分共卡跨节点；
5. sync factor=1.25/1.5 canary；
6. durable policy transition；
7. 显存 probe 连续 step。

本次机器按用户给定目标使用 **16 卡 actor + 12 卡 rollout + 4 卡 Polar lease pool**。对应仓内单机布局是
`scripts/resource_layout.single52_homo_colocate.yaml`：actor 声明 `0-15`，rollout 声明 `4-15`，
共 6 个 TP2 engine；Polar 的 `0-3` 是宿主服务租约池，不进入 Ray placement。由于 actor 覆盖全部物理卡，
该布局不能同时声明 `polar_reserved`，这是布局文件中已记录的约束，不是漏配。

## 2. 阶段记录

| 阶段 | 目标 | 状态 | 证据/结论 |
| --- | --- | --- | --- |
| 0 | 环境、版本、端点与拓扑 fingerprint | 通过（外部依赖正常） | 见 §3 |
| 1 | async 分卡回归 | 通过（disaggregated） | actor `4-11`、rollout `12-15`，2 个 TP2 engine；Polar session-pool、权重同步、1 step 和 checkpoint 均完成，见 §4.4；hybrid 共卡路径仍按 §4.3 受显存约束 |
| 2 | sync factor=1、`NUM_ROLLOUT=2` | **通过** | 16 卡 actor、12 卡 rollout、6 个 TP2 engine 完成 2 次 rollout、2 个训练 step、3 次权重同步及 checkpoint，见 §5.3 |
| 3 | 单机异构 engine-role 分流 | **通过** | actor/shared rollout `4-11`，dedicated rollout `12-15`；IPC/HCCL 双通路、role-selective sleep、两轮同步和 step 0 均完成，见 §6 |
| 4 | 部分共卡跨节点 | 外部资源阻塞 | `.64` Polar gateway 当前有 12 个实例/64 个运行中 session，SSH 无凭据，不能安全复用为第二训练节点；见 §7 |
| 5 | sync factor=1.25/1.5 | **通过** | factor=1.5 disaggregated canary 完成 cancel/requeue、训练 step 和最终同步，见 §8 |
| 6 | durable transition 三个 policy version | 部分通过/外部配置阻塞 | 重启后 durable bootstrap API 已上线并成功提交；完整 rollout 被 Polar gateway 后端 `.56:8011` 不可达阻塞，见 §9.1 |
| 7 | `VIME_MEM_PROBE=1` 至少两个完整 step | **通过** | 两个 sync rollout、两个 train step、三次 policy version 推进和显存 handoff probe 完成，见 §10 |

## 3. 阶段 0：初始环境快照

### 3.1 代码与工作区

- 隔离工作区：`/workspace/vime-a3pd-integrated`。
- 分支：`dev/a3pd-integrated`，相对 `bryan/a3-pd` ahead 17 个 topic commit。
- 初始工作区 clean；原 `/workspace/vime` 的用户脏改动未触碰。
- CPU/静态组合回归在 Step13 已通过 `183 passed`。

### 3.2 Ray 与 NPU

- 容器内发现旧 Ray head：`.52:6461`，session 目录为
  `/tmp/ray_qwen36_vime_polar/session_2026-09-11_10-33-09_802828_2060280`。
- `ray status` 显示单节点、16 NPU、当前 Ray 使用量为 0；这是残留运行时，不作为本轮验证基线。
- `npu-smi` 显示 16 张 910B2C；当前卡 4-15 有 vLLM 进程/约 49-50 GiB HBM 占用，卡 0-3 在当前容器视角无进程。
- 后续只清理 VIME/Ray/vLLM 残留，不调用 Polar 宿主启动/停止脚本。

### 3.3 Polar 宿主服务

最新宿主 run：`/home/c00937190/polar/output/ascend_operator/runs/polar_20260912_014715`。

- rollout：`http://80.48.5.52:8180`，`GET /health` 返回 `{"status":"ok","nodes":1}`。
- gateway：`http://80.48.5.52:8200`，`GET /health` 返回 200，节点和 session 管理正常。
- effective topology：Polar NPU lease pool 为 `0,1,2,3`，gateway inference base URL 为
  `http://80.48.5.52:8001`。
- gateway health 当前报告 inference `ConnectError: All connection attempts failed`，原因是 VIME/vLLM
  router `:8001` 尚未在本轮启动；这不是 Polar rollout/gateway 自身故障。
- 容器内未发现 hostctl token，因此不通过 hostctl 操作 Polar；Polar 出问题时按约定跳过依赖 Polar 的阶段。

### 3.4 清理动作

- 执行 `ray stop --force`，停止旧 `.52:6461` Ray session，共 118 个 Ray 进程；未操作 Polar 服务。
- 仅针对已核实的 `VLLM::EngineCore`/`VLLM::Worker` PID 发送 `SIGTERM` 后 `SIGKILL`，释放旧 6 个 TP2 engine。
- 清理后 `npu-smi` 16 张卡均无运行进程；Polar `:8180/health` 仍返回 200，说明宿主服务未被清理动作影响。

## 4. 阶段 1：async 分卡回归

本阶段使用 `scripts/resource_layout.single52_hybrid_colocate.yaml`：Polar 租约池 `0-3`，actor
`4-11`（8 卡），rollout `4-15`（8 张 shared + 4 张 dedicated，共 6 个 TP2 engine）；Ray 可见
全 16 卡但 placement 只申请 actor/rollout 的 12 张。该阶段先验证 async/session-pool 和默认
weight-sync；sync 专属的 zero-inflight、sleep/onload 闭环留到阶段 2。

启动参数（唯一变化是使用集成分支 launcher；不启用 sync/durable/MTP/YaRN）：

```text
CURRENT_IP=MASTER_ADDR=80.48.5.52, NNODES=1, NPUS_PER_NODE=16
ASCEND_RT_VISIBLE_DEVICES=0-15
RESOURCE_LAYOUT=scripts/resource_layout.single52_hybrid_colocate.yaml
ACTOR_NUM_GPUS_PER_NODE=8, ROLLOUT_NUM_GPUS=12, ROLLOUT_NUM_GPUS_PER_ENGINE=2
TRAIN_ENTRY=train_async.py, FEAT_SYNC_ROLLOUT=0, FEAT_OFFLOAD=0, FEAT_COLOCATE=0
POLAR_ROLLOUT_URL=http://80.48.5.52:8180, VLLM_ROUTER_PORT=8001, FEAT_LB_PROXY=1
ROLLOUT_BATCH_SIZE=1, N_SAMPLES_PER_PROMPT=1, GLOBAL_BATCH_SIZE=1, NUM_ROLLOUT=1
```

判定：Ray placement 日志必须显示 actor `4-11`、rollout `4-15` 的 role fingerprint 与 layout 一致；
`generate_rollout_polar_async` 必须建立 persistent/session-pool worker；gateway `/health` 的 inference
错误必须在 VIME LB proxy `:8001` ready 后消失；至少一个 rollout group 返回 usable trace，且训练进程
不出现 engine version/ACK/weight-sync 错误。最终 checkpoint 仅作为运行副产物保存，不纳入 Git。

### 4.1 首次执行结果（2026-09-11 18:07 UTC）

- 启动日志：`/home/docker/logs/a3pd_stage1_async_20260911.log`；Ray 临时目录：
  `/tmp/ray_a3pd_stage1_async`。
- launcher 闸门通过：Ray 单节点注册 NPU=16，layout 专用资源检查通过（actor `4-11`，dedicated
  rollout `12-15`）；async 入口、session-pool、LB proxy、TIS 均按预期展开，未启用 sync/offload/
  durable/MTP/YaRN。
- 训练尚未创建任何 actor/vLLM engine，在 `create_placement_groups()` 阶段退出，退出码非零。核心错误：
  `ValueError: Ray cluster is missing requested rollout devices: 80.48.5.52:12, ...:15. Available
  bundles: 80.48.5.52:0, ...:11`。
- 根因：`_build_layout_bundles()` 为 hybrid layout 只创建 actor `4-11` 和专用 rollout `12-15`
  共 12 个 NPU bundle；Ray 给 placement group 内的 12 个 bundle 分配连续 accelerator id `0-11`，
  并不会因为 bundle 的 YAML 顺序自动保留物理卡号空洞 `0-3`。随后
  `_select_rollout_bundles_with_share()` 按 YAML 物理卡号查找 `12-15`，因而失败。这个问题发生在
  VIME placement 映射层，和 Polar rollout/gateway 无关。
- 影响：当前 hybrid/稀疏物理卡 layout 不能进入真正训练；同构 `actor=0-15` 的 layout 不触发该
  空洞映射问题，但不能替代对 hybrid 角色分流的验证。
- 运行时清理：本次脚本退出后 Ray head 未保留可用 GCS；未操作 Polar。修复前不继续启动大模型，避免
  重复占卡。

### 4.2 修复方向（待单独 commit）

保持 YAML 的物理卡语义不变，在 layout placement group 中为每个节点从 `0` 到该节点 layout
引用的最大 device id 创建占位 bundle，使 Ray accelerator id 与物理 device id 对齐；actor/rollout
仍只选择各自声明的 bundle，`polar_reserved` 只用于对齐/预留，不创建 engine。随后用同一条 Stage 1
命令重跑，并先验证 role fingerprint，再等待 vLLM/Polar 业务状态。

### 4.3 修复后 hybrid async 执行结果（2026-09-11 18:18--18:23 UTC）

- placement 修复已生效：同一条 hybrid 命令再次运行时，16 个 padding bundle 使 Ray 探测到的
  物理卡号为 `0-15`；actor 日志出现 `LOCAL_RANK=4..11`，role fingerprint 与 YAML 一致。
- 6 个 TP2 engine 均完成 vLLM 进程启动、权重加载、KV cache/profile 和 graph capture；共卡段使用
  `NPUIPCWeightTransferEngine`，专用段使用 `HCCLWeightTransferEngine`。LB proxy 在 `:8001` ready，
  `:8001/health` 返回 `{"status":"ok","dp_instances":6}`，Polar `:8180/health` 与
  `:8200/health` 均恢复 200，gateway 报 `inference.status=ok`。
- 训练 actor 初始化随后失败：卡 6/9 报 `torch.OutOfMemoryError`，单次申请 7.50 GiB 时仅剩
  2.94 GiB；日志显示 vLLM shared engine 已占约 32.92 GiB 权重，Megatron actor 已占 14.40 GiB。
  进程在第一个 rollout 前退出，没有产生训练 step/trajectory。
- 结论：placement 修复通过；但 `train_async.py` 没有 rollout offload/sleep-wake 边界，不能在
  shared actor 卡上与常驻 vLLM 同时初始化。该组合不作为 async 通过条件；改用新增的
  `resource_layout.single52_disagg_polar.yaml`（actor `4-11`、专用 rollout `12-15`）重做 async
  最小回归。hybrid 共卡仍由阶段 2/3 的同步+offload 测试覆盖。

### 4.4 disaggregated async 最小回归（2026-09-11 18:26--18:42 UTC）

- 启动日志：`/home/docker/logs/a3pd_stage1_async_disagg_20260911.log`；Ray 临时目录：
  `/tmp/ray_a3pd_stage1_async_disagg`；输出目录：`/home/docker/a3pd_stage1_async_disagg_output`。
- 使用新增布局 `scripts/resource_layout.single52_disagg_polar.yaml`：actor 物理卡 `4-11`（8 卡），
  专用 rollout `12-15`（4 卡、2 个 TP2 engine），Polar 保留 `0-3`。role fingerprint 和 Ray
  `LOCAL_RANK=4..11` 与 YAML 一致，确认 padding bundle 修复覆盖了稀疏物理卡号。
- 两个 vLLM engine 均完成 checkpoint 加载、graph capture、`/health=200`；LB proxy `:8001`
  启动后 gateway `:8200/health` 的 inference 从 `ConnectError` 恢复为 `status=ok`。
- Megatron 8 卡完成 HCCL 建组，加载 `/home/docker/Qwen3.6-35B-A3B_mtp_torch_dist` 成功。
  首次权重同步建立 world size 5（8 卡 actor 中源 rank + 2 个 TP2 engine），128 个权重分片更新
  全部收到 HTTP 200，随后 gateway policy version 推进到 1。
- Polar async session-pool 成功创建并执行 rollout：至少 1 组完整 trace 返回
  `status=COMPLETED`；本次 benchmark 汇总为输入 24,939 token、生成 878 token、rollout success
  rate `1.0`。训练随后完成一个 actor train microbatch（约 80.6 s），日志输出 `step 0`，包含
  `train/tis=0.99777`、`train/grad_norm=0.01296`，并保存 debug train/rollout data。
- 进程收尾时又触发 policy version 2 的权重同步；同步期间 session-pool 取消了一批尚未完成的
  请求，日志出现 `aborted generation (weight-update cutoff)` 与 `zero trainable tokens`，随后等待
  16 个 in-flight task、停止 async worker 并正常退出。该现象不影响本次 `NUM_ROLLOUT=1` 的首个
  step 判定，但说明 async 收尾仍会先清空预取队列，后续长跑需要观察版本切换与 in-flight drain
  的吞吐/数据丢弃策略。
- 结束后 VIME/Ray/vLLM 进程均已退出；Polar `:8180/health` 仍为 200，`8200` 仍为 200（仅因
  router 已退出而报告 inference `ConnectError`）。未重启或停止 Polar。
- 判定：**Stage 1 disaggregated 路径通过**；**Stage 1 hybrid 共卡 async 路径不通过**（已知显存
  边界，保留为阶段 3 sync+offload 的验证对象）。

## 5. 阶段 2：sync factor=1、同构共卡与 offload

### 5.1 首次启动失败（2026-09-11 18:48--18:49 UTC）

- 启动日志：`/mnt/pipeline-data/train_log/train_a3pd_stage2_sync_homo_20260911.log`；Ray 临时目录：
  `/tmp/ray_a3pd_stage2_sync_homo`。布局为 actor `0-15`、shared rollout `4-15`，共 6 个 TP2
  engine；入口 `train.py`，`FEAT_SYNC_ROLLOUT=1`，`FEAT_OFFLOAD=1`，factor=1，`NUM_ROLLOUT=2`。
- placement 已通过：Ray 创建 16 个对齐 bundle，actor/rollout 物理卡映射与 YAML 一致；RolloutManager
  也识别出 shared engine 需要 offload。说明本次 padding bundle 修复在同构布局下没有破坏资源选择。
- 6 个 vLLM engine 在多进程 CPU Gloo 建组阶段退出，训练尚未完成 engine health、offload 或 actor
  初始化。根因是 launcher 默认 `SOCKET_IFNAME=data0.172`，而本容器只有 `ens1f3=80.48.5.52`
  （另有 docker/bridge 网卡）；错误为：`RuntimeError: ... gloo/transport/tcp/device.cc:84 ...
  Unable to find address for: data0.172`。
- 该失败属于当前容器网卡配置与脚本默认值不匹配，和 YaRN rope 参数、Polar endpoint 或同步协议无关。
  重跑将显式设置 `SOCKET_IFNAME=ens1f3`，并保留相同代码、布局及训练参数，以隔离环境变量影响。

### 5.2 网卡修正后的启动与 Polar model alias 阻塞（2026-09-11 18:51--19:00 UTC）

- 启动日志：`/mnt/pipeline-data/train_log/train_a3pd_stage2_sync_homo_retry_20260911.log`；使用同一
  16-card/6-engine layout 和 `SOCKET_IFNAME=ens1f3`。placement、Gloo/HCCL 建组、6 个 vLLM
  `/health=200`、LB proxy `:8001` 和 gateway `:8200` 均正常，说明网卡修正有效。
- `train.py` 的 actor 16 卡初始化成功；训练权重 4 个 flat buffer、约 30,105.6 MiB 已 offload，
  每卡剩余约 58.4 GiB。startup memory handoff 完成，vLLM engines 从 sleep level 2 唤醒 weights，
  255 个 weight fragment 的首轮 `update_weights` 全部返回 HTTP 200；随后 engine 再次进入可推理状态。
- sync rollout 未获得可训练样本：Polar adapter 每次请求的 `model` 字段为
  `/home/docker/Qwen3.6-35B-A3B`，而本次 vLLM 仅以实际 checkpoint 路径
  `/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16` 提供服务，故 vLLM 返回
  `404 NotFoundError: The model '/home/docker/Qwen3.6-35B-A3B' does not exist.`；9 个补发 group
  全部被拒，`PolarRolloutSchedulerError` 在 rollout 0 退出。没有进入训练 step。
- 根因是 vLLM served-model alias 未在此次手工命令中设置，不是同步 zero-inflight、offload/wake 或
  权重同步实现错误。仓内 `vllm_engine.py` 已支持 `VLLM_SERVED_MODEL_NAME`（会同时保留真实路径和
  alias）；下一次复现显式设置 `VLLM_SERVED_MODEL_NAME=/home/docker/Qwen3.6-35B-A3B`。
- 结束后 VIME/Ray/vLLM 均退出；Polar `:8180/health`、`:8200/health` 保持 200，未操作 Polar。

### 5.3 model alias 修正后的完整闭环（2026-09-11 19:02--19:18 UTC）

- 启动日志：
  `/mnt/pipeline-data/train_log/train_a3pd_stage2_sync_homo_alias_20260911.log`；Ray 临时目录：
  `/tmp/ray_a3pd_stage2_sync_homo_alias`。除显式设置 `SOCKET_IFNAME=ens1f3` 和
  `VLLM_SERVED_MODEL_NAME=/home/docker/Qwen3.6-35B-A3B` 外，保持 actor 16 卡、shared rollout
  `4-15`、6 个 TP2 engine、`CP=8`、factor=1 和 `NUM_ROLLOUT=2`。vLLM 的真实 checkpoint 为
  `/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16`，served model 同时包含真实路径和
  Polar 使用的 alias。
- Ray placement role fingerprint 与布局一致：actor 使用物理卡 `0-15`，rollout 使用物理卡
  `4-15`，Polar 宿主 lease pool 保持 `0-3`。6 个 vLLM engine 全部通过 health；shared engine
  使用 NPU IPC 权重通道，LB proxy `80.48.5.52:8001` ready，Polar gateway inference 恢复正常。
- 16 个 actor rank 完成 checkpoint 加载；每 rank 的 4 个 flat buffer 约 30,105.6 MiB 在 rollout
  前卸载，rank 0 卸载后空闲显存约 59.5 GiB。vLLM 使用
  `VLLM_GPU_MEM_UTIL=0.70`，sleep level 2 每个 worker 释放约 41.27 GiB；actor 和 rollout 的
  sleep/wake/onload handoff 连续两轮完成，没有 NPU OOM。
- 权重同步完整执行三次（初始、step 0 后、结束前）：每次均为 255 个 fragment、6 个 TP2 engine，
  耗时约 72--74 s，HTTP collective/start/finish 请求均返回 200。gateway 在切换前两次报告
  `paused=True, drained=True, inflight=0`，policy version 依次推进到 1 和 2，切换后所有节点
  `all_resumed=True`。
- 两次同步 rollout 均成功：rollout 0 在 42.3 s 收集 `1/1` group，rollout 1 在 50.4 s 收集
  `1/1` group；两次都是 `submitted=1`、`accepted=1`、`rejected=0`、`topups=0`、success rate
  `1.0`。生成 token 分别为 3,205 和 4,024，未出现 model 404、scheduler error 或 policy-version
  拒绝。
- 训练完成 `step 0` 和 `step 1`。step 0：`train/tis=0.9986751`、
  `train/grad_norm=0.004497`；step 1：`train/tis=0.9987841`、
  `train/grad_norm=0.004883`。loss、logprob 差异和梯度指标均为有限值，进程最终退出码为 0。
  Megatron distributed checkpoint 成功保存至 `/workspace/Qwen3.6-35B-A3B_vime_polar/`。
- 发现一个不影响本阶段训练/同步判定的 launcher 缺陷：即使调用方显式设置 `SAVE_HF`，脚本的
  shell 默认值展开仍会在值末尾多出一个 `}`，最终报
  `Failed to save HuggingFace format: Single '}' encountered in format string`。Megatron checkpoint
  已正常保存；HF 导出问题需单独修复并回归，不归因于 topology/rollout topic。
- 判定：**Stage 2 通过**。这是目标单机共卡拓扑（16 卡训练、其中 12 卡同步推理、Polar 使用宿主
  0--3 卡）的完整 2-step 真机闭环证据。

## 6. 阶段 3：单机异构 engine-role 分流

### 6.1 运行配置与中间结果（2026-09-11 19:20 UTC 起）

- 启动日志：`/mnt/pipeline-data/train_log/train_a3pd_stage3_sync_hybrid_20260911.log`；Ray 临时目录：
  `/tmp/ray_a3pd_stage3_sync_hybrid`。使用
  `scripts/resource_layout.single52_hybrid_colocate.yaml`：actor `4-11`（8 卡）、shared rollout
  `4-11`（4 个 TP2 engine）、dedicated rollout `12-15`（2 个 TP2 engine），Polar lease pool
  `0-3`。训练并行度为 `TP=2, PP=1, CP=4, EP=8`，sync factor=1，`NUM_ROLLOUT=1`。
- launcher 默认数据集路径 `operator_tasks.16.jsonl` 在当前环境不存在，本次显式使用同目录已有的
  `operator_tasks.jsonl`。这属于部署数据路径差异，未修改代码或数据内容。
- Ray 创建 16 个对齐 bundle；actor 的 `LOCAL_RANK=4..11`、rollout 的物理卡 `4-15` 与 YAML
  一致。4 个 shared engine 使用 `NPUIPCWeightTransferEngine`、
  `vLLMColocateWorkerExtension` 和 `gpu_memory_utilization=0.70`；2 个 dedicated engine 使用
  `HCCLWeightTransferEngine`、不加载 colocate extension，并使用 `gpu_memory_utilization=0.85`。
  专用通路实际建立了 world size 5 的 HCCL 更新组（1 个 actor source + 4 个 dedicated rollout
  ranks），并非只完成配置解析。
- 6 个 engine 均完成 checkpoint、KV cache 和 graph capture。startup 与 rollout-to-train 交接时，
  只有 4 个 shared engine 执行 sleep level 2，每 worker 释放约 41.27 GiB；dedicated engine 保持
  常驻。8 个 actor rank 各卸载约 30,105.6 MiB，rank 0（物理卡 4）卸载后约 56.9 GiB 空闲。
- 首轮混合权重同步完成 `255/255` 个 fragment，耗时 262.8 s；IPC 与 HCCL consumer 均返回成功。
  同步后 Polar sync rollout 在 26.2 s 收集 `1/1` group，`submitted=1`、`accepted=1`、
  `rejected=0`、`topups=0`，输入 24,949 token、生成 1,955 token，success rate 1.0。
- rollout 后 shared engine 再次 sleep，rank 0 的设备空闲从 15.60 GiB 回升到 56.87 GiB；训练模型
  onload 后完成 step 0，输出 `train/tis=0.99745196`、`train/grad_norm=0.00629596`，actor train
  耗时 118.9 s。`VIME_MEM_PROBE=1` 的 startup、rollout、offload、pre/post fwd-bwd 和 train
  handoff tag 均已出现。
- Megatron checkpoint 已写入。为避免本 canary 再写约 66 GB 的重复 HF 副本，本次故意设置
  `SAVE_HF=/proc/a3pd_stage3_hf_{rollout_id}`；因此日志中的 HF 目录创建失败是预期测试配置，主流程
  继续执行，不作为功能失败。`{rollout_id}` 未再出现额外 `}`，说明 `3a239e16` 的 launcher 修复
  已在真实调用中生效。
- step 0 后的第二次 `255` fragment 权重同步于 19:41:40 完成，耗时 273.8 s；gateway 返回
  `paused=True, drained=True, inflight=0`，清理 sticky session 后 `all_resumed=True`，最终日志为
  `Finished Polar bridge policy_version=1 weight update`。整个 launcher 退出码为 0，VIME/Ray/vLLM
  进程均已退出；Polar `:8180/health` 仍为 200，未操作宿主服务。
- 判定：**Stage 3 通过**。本阶段证明同一节点上 shared/dedicated engine 会依据物理 role 分别选择
  IPC/HCCL，并且只卸载 shared engine；sync rollout、训练、checkpoint 和最终 zero-inflight
  policy update 均闭环。日志中的 layerwise `Failed to load weights` 是本版本分片重载时已知的非致命
  warning，未导致 HTTP/训练失败。

## 7. 阶段 4：部分共卡跨节点

- 只读连通性检查（2026-09-12）：`.64:8180/health` 和 `.64:8200/health` 均返回 HTTP 200，但
  `.64:8200/health` 报告 `12` 个 rollout 实例和 `64` 个 `RUNNING` session，说明该节点正被 Polar
  业务占用；`.64:22` 可建立 TCP 连接但 `root` SSH 返回 `Permission denied (publickey,password)`。
- 当前容器只有 `.52` 的 16 张 NPU 可调度；没有第二节点的 Ray worker、空闲 NPU 或授权清理入口。为
  避免影响现有 Polar 业务，本阶段不启动双机 Ray/HCCL 训练，不向 `.64` 发送 stop/kill/重启操作。
- 判定：**外部资源阻塞，非 VIME 代码失败**。需要提供空闲第二节点和可用 Ray/HCCL 启动权限后，才可
  按 `resource_layout.hybrid56cola64infer.yaml` 重做部分共卡跨机验证。

## 8. 阶段 5：sync factor=1.5 canary

- 启动日志：`/mnt/pipeline-data/train_log/train_a3pd_stage5_sync_factor15_20260912.log`；Ray 临时目录：
  `/tmp/ray_a3pd_stage5_sync_factor15`。使用 `resource_layout.single52_disagg_polar.yaml`：actor
  `4-11`（8 卡），dedicated rollout `12-15`（2 个 TP2 engine），Polar lease pool `0-3`；
  `FEAT_SYNC_ROLLOUT=1`、`FEAT_OFFLOAD=0`、`POLAR_SYNC_OVERSUBSCRIBE_FACTOR=1.5`、
  `POLAR_POLICY_TRANSITION_ENABLED=0`。
- 初始权重同步完成 `128/128` fragments，耗时 225.0 s。sync rollout 在 18.4 s 返回 `1/1` usable
  group；内部提交 `2` 个候选，`accepted=1`、`rejected=0`、`aborted=1`、`aborted_sessions=1`、
  `requeued=1`、`topups=0`。这证明超订窗口中未选 group 会先收到取消确认并回队，函数返回时没有把
  未确认任务遗留在后台；rollout success rate 为 `1.0`，输入/输出 token 为 `24,919/1,255`。
- 训练完成 step 0，指标为 `train/tis=0.99881893`、`train/grad_norm=0.00572126`；distributed
  checkpoint 成功写入 `/workspace/Qwen3.6-35B-A3B_vime_polar_stage5/`。末轮 `128/128` fragments
  同步完成，耗时 221.2 s，进程退出码为 0；结束后 VIME/Ray/vLLM 均退出，Polar `:8180/health`
  仍为 200。
- `VIME_MEM_PROBE=1` 的 handoff 日志存在，但本阶段只有一个完整 step；连续 step 的显存趋势留给
  阶段 7。`SAVE_HF=/proc/a3pd_stage5_hf_{rollout_id}` 导致的 HF 保存 `FileNotFoundError` 是有意的
  canary 配置，不影响 distributed checkpoint 或本阶段判定；layerwise load warning 也未转化为
  HTTP、同步或训练错误。
- 判定：**Stage 5 通过**。factor>1 的 cancel acknowledgement、group requeue、zero-inflight
  返回和后续 train/checkpoint/weight-sync 闭环均有真实日志证据。

## 9. 阶段 6：durable policy transition

- 启动日志：`/mnt/pipeline-data/train_log/train_a3pd_stage6_durable_probe_20260912.log`；Ray 临时目录：
  `/tmp/ray_a3pd_stage6_durable_probe`。使用 disaggregated actor `4-11`、dedicated rollout `12-15`，
  `FEAT_SYNC_ROLLOUT=1`、`FEAT_OFFLOAD=1`、`POLAR_POLICY_TRANSITION_ENABLED=1`、
  `NUM_ROLLOUT=3`、`VIME_MEM_PROBE=1`。
- placement、actor checkpoint 加载和 rollout engine 初始化均成功；actor 先完成模型 offload，物理卡
  空闲从约 30.1GiB 增至约 49.7GiB，说明 durable 组合没有先触发显存冲突。
- 首个 durable bootstrap 在调用 `POST http://80.48.5.52:8180/rollout/admin/policy/bootstrap/begin`
  时收到 `404 Not Found`。随后 fail-closed 清理调用的
  `/rollout/admin/policy-transitions/<transition>/fail` 也收到 404；训练在任何 rollout/step 前退出，
  没有继续走非 durable 降级路径。`.52:8180/health` 始终为 200，故这是当前宿主 Polar 版本缺少
  policy-control 路由，不是 Polar 服务整体不可用，也不是 VIME 显存或 placement 失败。
- 判定：**外部接口阻塞，代码路径未完成通过判定**。需要在宿主 Polar 部署包含 bootstrap/transition
  policy-control API 的版本后，用相同参数重跑；当前实现的 fail-closed 行为本身已被触发并保留在日志中。

### 9.1 重启后的 durable 回归（2026-09-12 15:46--16:10 UTC）

- Polar 重启后，`.52:8180/openapi.json` 已出现完整 durable 路由：
  `policy/initialize`、`policy/bootstrap/begin`、`policy-transitions/*`、`policy/quiesce` 和
  `policy_version`。对 `POST /rollout/admin/policy/bootstrap/begin` 的方法探测返回 `405 Allow: POST`，
  不再是旧的 `404`，确认宿主服务已加载新版本。
- 前两次启动只暴露了测试命令的资源注册错误：一次以 `NPUS_PER_NODE=12` 注册却要求 16 个逻辑
  bundle，另一次以 `NPUS_PER_NODE=16` 配合可见卡 `4-15` 被 Ray 拒绝。两次均在模型业务初始化前
  清理，没有修改代码。第三次改为注册逻辑卡 `0-15`，再用 layout 选择 actor `4-11`、rollout
  `12-15`，placement 成功。
- 第三次运行参数保持 `FEAT_SYNC_ROLLOUT=1`、`FEAT_OFFLOAD=1`、
  `POLAR_POLICY_TRANSITION_ENABLED=1`、`NUM_ROLLOUT=3`、`CP=4`、`EP=8`，日志：
  `/mnt/pipeline-data/train_log/train_a3pd_stage6_durable_retest_20260912c.log`，Ray 日志目录：
  `/tmp/ray_a3pd_stage6_durable_retest_c`。两个 TP2 vLLM engine、8 卡 actor、LB proxy 均完成启动；
  actor rank 0 的 offload 从约 `30.14GiB` 空闲提升到 `49.73GiB`，释放 2 个 flat buffer、
  `20070.4MiB`。
- 首次同步的 `254/254` 个权重 fragment 全部返回 HTTP 200，pause/resume 也均为 200。随后 Polar
  成功提交 durable bootstrap transition：
  `bootstrap-32d33e780ead2aba-0-to-0`，`phase=serving`、`verified_policy_epoch=0`、
  `engine_versions={engine-000:1,engine-001:1}`、`engine_abort_confirmed=true`、`last_error=null`。
  这部分证明 VIME 的 durable bootstrap 调用链与新 Polar API 已实际闭环。
- 权重同步完成后，VIME 尝试向 `POST /rollout/admin/policy_version?version=1` 发布下一版本时收到
  `502 Bad Gateway`。最新 Polar run
  `/home/c00937190/polar/output/ascend_operator/runs/polar_20260912_233241` 的
  `run_artifacts/effective_topology.yaml` 明确记录 `rollout_server_url=.64:8180`、
  `gateway public_url=.64:8200`、`inference base_url=.56:8011`；同一时刻 Polar 返回的 gateway node 配置也是
  `base_url=http://80.48.5.56:8011`，而本次 VIME 启动的 router/LB 是 `80.48.5.52:8001`；从容器对
  `.56:8011` 的 HTTP 探测也只能得到代理层 `504`，无法建立有效 inference 请求。因此首个
  rollout 一直没有得到可训练结果，未进入 `step 0`。
- 为避免继续无限等待，16:09 UTC 主动中止该测试。随后 Polar 状态恢复为
  `all_reachable=true, all_drained=true, inflight=0, active_generations=0`，但 manager 中本轮 4 个
  `a3pd_stage6_durable_retest_20260912c-polar-op-0-*` task 仍为 `running`。随后逐个调用精确 task cancel
  清理；4 个 task 均返回 `status=cancelled, all_cancelled=true, cancelled_sessions=4,
  failed_sessions=0`，没有操作其他任务。确认 `/tasks?status=running` 为空后，再以本轮 namespace/epoch
  调用 durable `policy/quiesce`；transition 最终为 `phase=quiesced`、`admission_closed=true`、
  `serving=false`，gateway `paused=true, drained=true, inflight=0`。该中止不是 VIME 代码崩溃，也不是
  durable bootstrap 事务失败，且测试退出后的 policy 状态已按协议关闭。
- 判定：**durable bootstrap/首轮权重同步通过；完整“三版本 + rollout + train step”仍被外部
  gateway profile/backend 地址阻塞**。需要让重启后的 Polar 使用与本轮 VIME 相同的 inference
  backend（单机应为 `.52:8001`，或提供可达的 `.56:8011` 服务）后，再以相同命令复测 policy
  version 1/2/3 的 begin--drain--commit--resume 全链路。本轮没有修改 Polar 配置。

## 10. 阶段 7：连续 step 显存 probe

- 启动日志：`/mnt/pipeline-data/train_log/train_a3pd_stage7_mem_probe_20260912.log`；Ray 临时目录：
  `/tmp/ray_a3pd_stage7_mem_probe`。使用 disaggregated actor `4-11`、dedicated rollout `12-15`，
  `FEAT_SYNC_ROLLOUT=1`、`FEAT_OFFLOAD=1`、`POLAR_POLICY_TRANSITION_ENABLED=0`、factor=1、
  `NUM_ROLLOUT=2`、`VIME_MEM_PROBE=1`。
- startup 的 `onload_kv` probe 为 rank 0 `dev_free=48.69GiB`；第一个 rollout 后训练/同步/恢复闭环
  正常，rollout 0 收集 `1/1` group（62.5s），step 0 指标为 `tis=1.00038695`、
  `grad_norm=0.00421216`。step 0 后同步耗时 227.6s，gateway 推进到 policy version 2。
- 第二个 rollout 收集 `1/1` group（16.2s），step 1 指标为 `tis=0.99445134`、
  `grad_norm=0.00540986`；step 1 后 rank 0 的关键 probe 为：训练前 `dev_free=20.74GiB`、
  offload 后 `46.46GiB`、wake 后 `26.87GiB`，与 step 0 的约 `22.82GiB` post-forward 和
  `27.70GiB` wake 后余量同一量级，没有随 step 单调下降的显存趋势。
- 最终 `254/254` fragments 同步完成（226.9s），gateway 推进到 policy version 3，distributed
  checkpoint 成功写入 `/workspace/Qwen3.6-35B-A3B_vime_polar_stage7/`，launcher 退出码为 0；
  VIME/Ray/vLLM 均退出，Polar `:8180/health` 仍为 200。
- `SAVE_HF=/proc/a3pd_stage7_hf_{rollout_id}` 导致的 HF 保存 `FileNotFoundError` 是为避免复制大模型
  副本而设置的预期结果；标准 HCCL gather、layerwise load 和 compiler cache warning 未影响 step、
  同步或 checkpoint。
- 判定：**Stage 7 通过**。该阶段证明 opt-in memory probe 可覆盖至少两个完整 sync train step，
  且 offload/onload handoff 没有出现不可解释的单调 HBM 增长。

## 11. 判定规则

- 每个阶段都记录启动参数、进程/Ray 资源、关键日志和退出状态。
- Polar endpoint 返回非 2xx、router 无法建立连接或宿主 lease pool 异常时，标记为外部依赖阻塞，
  不把该失败归因于 VIME topic，并继续执行不依赖 Polar 的阶段。
- 真机验证不修改训练代码；若发现代码缺陷，先保存完整日志和复现参数，再单独创建修复 commit。
- 最终结论区分“通过”“代码路径已运行但外部依赖阻塞”“失败待修复”，不把 launcher dry-run 当作真机通过。

## 12. 静态回归与证据索引

- 与本次 topic 直接相关的静态回归在补充 `PYTHONPATH=/workspace/Megatron-LM` 后通过：
  `pytest -q tests/test_launcher_contracts.py tests/test_resource_layout_placement.py tests/test_engine_roles.py
  tests/test_npu_weight_offloader.py tests/test_npu_training_state_offloader.py tests/test_colocate_memory_probe.py`
  结果为 **69 passed**。不设置该路径时，engine-role 的 2 个测试仅因 `megatron` 未进入 Python path 而
  collection failure。
- 直接运行全量 `pytest -q` 收集到 35 个既有环境错误、1 skipped，未进入完整断言阶段：`tests/unit/conftest.py`
  的全局 Ray/Transformers stub 会污染后续模块，且当前 Typer/Click 版本不兼容；典型错误为
  `ModuleNotFoundError: ray.util`、`ImportError: ray.ObjectRef` 和 `TypeError: Choice is not subscriptable`。
  这不是本轮真机路径失败；Step13 已有同一代码基线的 `183 passed` topic 回归记录。
- 大日志和运行副产物均保留在 `/mnt/pipeline-data/train_log/` 及对应 `/workspace/Qwen3.6-35B-A3B_vime_polar_stage*/`
  目录，不纳入 Git。

后续阶段的命令输出、日志目录和关键指标追加到本文件；大日志只记录路径、时间范围和摘要，避免把完整运行产物提交进 Git。
