# Async Split NPU Training — HCCL Weight Transfer Failure Analysis

## Background

目标是让 `/workspace/run_qwen3_8b_npu0_3.sh` 以 **异步、分卡** 的方式运行：

- NPU 0-1：actor 训练
- NPU 2-3：rollout / vLLM 推理
- 使用 `train_async.py`
- 非 `--colocate`

在这个模式下，训练环路会在 actor 初始化后立即执行第一次权重同步：

1. actor 侧加载 Megatron checkpoint
2. rollout vLLM engine 启动并通过 `/health`
3. `train_async.py` 在第一个 train loop 前调用 `actor_model.update_weights()`
4. actor 将权重同步给 rollout engine

前面已经先后修复了：

- W&B 登录检测误判（netrc 登录）
- W&B 依赖缺失（`sentry_sdk`）
- Megatron-LM 在 NPU 上错误调用 `torch.cuda.get_device_properties`
- MindSpeed `repatch()` 在 runtime args 合并时缺少默认字段（`optimization_level`、`optimizer_selection` 等）
- TP=1 导致 8B actor 在 2 卡 actor 拓扑下整模型复制并 OOM
- `vllm-ascend` 中完全缺失 NPU distributed weight transfer 运行时支持

---

## 本次对话中已解决问题与原理（完整记录）

这一节记录本次对话里已经完成的修复、每个问题的现象、根因、修改点与原理。目的不是只记录“改了什么”，而是保留一条完整的排障链，以便后续继续调异步分卡路径时能快速定位哪些问题已经被排除。

### 1. W&B 登录检测误判：`wandb login` 成功，但 `wandb status` 仍显示 `api_key: null`

#### 现象

用户执行：

```bash
wandb login --relogin --host=http://127.0.0.1:8088
wandb status
```

之后 `wandb status` 仍显示：

```json
"api_key": null
```

但 `wandb login` 提示已写入凭据。

#### 根因

W&B 的当前登录信息实际落在：

- `/root/.netrc`

而不是：

- `/root/.config/wandb/settings`

`wandb status` 展示的是 settings/env 视角下的 `api_key` 字段，不会把 netrc 中的凭据回显到这个字段。因此“status 显示 null”不代表未登录。

#### 验证证据

- `/root/.netrc` 中存在：
  - `machine 127.0.0.1:8088`
  - `password local-wandb_v1_...`
- `wandb status` 仍然 `api_key: null`

#### 修改点

修改：

- `/workspace/run_qwen3_8b_npu0_3.sh`

加入了：

- W&B host 可达性检测
- netrc 凭据检测（Python `netrc` 读取）
- 若无有效登录则直接 fail-fast，不再静默关闭 W&B

#### 原理

从“依赖 `wandb status` 某个字段是否为 null”切换到“检测实际认证来源是否存在”，即：

1. `WANDB_API_KEY`
2. `~/.netrc` 中目标 host 的登录信息
3. W&B host 可访问性

这样与本地/自建 W&B 的真实认证方式一致。

### 2. W&B 运行时缺少 `sentry_sdk`

#### 现象

训练在 `wandb.init()` 时崩溃：

```text
ModuleNotFoundError: No module named 'sentry_sdk'
```

位置在：

- `/usr/local/lib/python3.11/site-packages/wandb/sdk/wandb_settings.py:1571`

#### 根因

W&B 在构造 settings 的 `_aws_lambda` 相关逻辑时 import 了 `sentry_sdk.integrations.aws_lambda`，但当前环境没有安装 `sentry-sdk`。

#### 修改点

执行安装：

```bash
python -m pip install sentry-sdk
```

#### 原理

这是运行依赖不完整，不是训练脚本或 W&B host 配置问题。补齐 Python runtime 依赖即可继续推进。

### 3. Megatron-LM 在 NPU 环境下错误调用 CUDA API

#### 为什么 8B 训练方案之前没走到这条 CUDA-only 检查？

关键不是“异步”或“分卡”本身，而是**模型参数差异**。

8B 方案使用的是：

- `/workspace/vime/scripts/models/qwen3-8B.sh`

它是一个 dense 模型参数集合，不包含：

- `--moe-grouped-gemm`
- `--num-experts ...`
- `--moe-layer-freq ...`
- `--moe-router-topk ...`
- `--moe-token-dispatcher-type ...`

所以 8B 方案不会触发 Megatron 里与 MoE grouped GEMM 相关的 CUDA capability 检查。

而 35B A3B 方案使用的是：

- `/workspace/vime/scripts/models/qwen3.5-35B-A3B.sh`

它明确包含：

- `--moe-grouped-gemm`
- `--num-experts 256`
- `--moe-layer-freq ...`
- `--moe-router-topk 8`
- `--moe-token-dispatcher-type alltoall`
- `--expert-model-parallel-size 8`

Megatron 中对应的校验逻辑是：

```python
if args.moe_grouped_gemm:
    dc = torch.cuda.get_device_capability()
    assert dc[0] >= 8
```

所以：

- **8B 方案没进这条分支**，因为它不是 MoE grouped-gemm 模型参数组合；
- **35B A3B 方案会进这条分支**，因为模型脚本显式打开了 `moe_grouped_gemm`。

也就是说，这不是“改成 12 卡导致的新问题”，而是：

> 切换到 35B A3B 这套 MoE 参数后，Megatron 首次走到了之前 8B 根本不会经过的 CUDA-only 校验路径。

#### 现象

切到 async 分卡 + TP>1 后，Megatron 参数校验阶段直接报：

```text
AssertionError: Torch not compiled with CUDA enabled
```

调用链：

- `/workspace/Megatron-LM/megatron/training/arguments.py`
- `/workspace/Megatron-LM/megatron/training/utils.py:get_device_arch_version()`

#### 根因

Megatron 的这段逻辑默认假设设备是 CUDA：

```python
torch.cuda.get_device_properties(torch.device("cuda:0")).major
```

在 NPU-only PyTorch 构建下，这会直接炸。

#### 临时绕过与副作用

最开始通过把：

- `--tensor-model-parallel-size 2`

临时降成：

- `--tensor-model-parallel-size 1`

绕过了这段 CUDA-only 检查，但随之带来 actor 两卡整模型复制，最终导致 OOM。

#### 正式修改点

修改：

- `/workspace/Megatron-LM/megatron/training/utils.py`

让 `get_device_arch_version()` 在 `torch.cuda.is_available() == False` 时返回 `0`，而不是继续碰 CUDA API。

#### 原理

对 NPU-only 环境提供安全降级，使 Megatron 的“GPU 架构判断”不会再成为阻塞点。这样就可以把 TP 恢复为 2，避免 actor 全模型复制。

### 4. MindSpeed `repatch()` 缺少默认字段：`optimization_level`

#### 现象

actor 初始化期间崩溃：

```text
AttributeError: 'Namespace' object has no attribute 'optimization_level'
```

调用链：

- `/workspace/MindSpeed/mindspeed/megatron_adaptor.py:repatch()`
- `/workspace/MindSpeed/mindspeed/features_manager/feature.py:19`

#### 根因

`repatch(args)` 会把 runtime args 合并到 `get_full_args()` 返回的对象，但这个对象在某些 Ray actor 初始化路径里并不包含 MindSpeed feature manager 期望的默认字段。

#### 第一轮修改

先在：

- `/workspace/MindSpeed/mindspeed/megatron_adaptor.py`

里对 `optimization_level` 做单字段兜底。

#### 发现的新问题

随后又报：

```text
AttributeError: 'Namespace' object has no attribute 'optimizer_selection'
```

说明问题不是缺一个字段，而是缺一批默认 MindSpeed 参数。

### 5. MindSpeed `repatch()` 缺少整套默认参数：`optimizer_selection` 等

#### 根因

`repatch()` 不能只补单个字段，而必须先拥有完整的默认 MindSpeed 参数集，否则 feature manager 的 `pre_register_patches()` 会继续在别的字段上崩。

#### 正式修改点

修改：

- `/workspace/MindSpeed/mindspeed/megatron_adaptor.py`

逻辑改成：

1. `get_full_args()`
2. `get_mindspeed_args(get_defaults=True)`
3. 先把默认 MindSpeed args 全量 merge 到 `full_args`
4. 再覆盖 runtime args

#### 原理

这等价于把“MindSpeed 的默认 CLI 参数命名空间”显式补到 actor 进程里的 runtime args 上，让 patch manager 在 Ray / actor / NPU 初始化这种特殊路径下仍能拿到与正常 CLI 启动一致的参数全集。

### 6. TP=1 导致 8B actor 在 2 卡 actor 拓扑下 OOM

#### 现象

在 TP=1 的临时绕过方案下，actor 初始化时 OOM：

```text
torch.OutOfMemoryError: Tried to allocate 30.51 GiB
```

日志显示每个 actor rank 持有大约 8.19B 参数。

#### 根因

TP=1 使 2 张 actor 卡不再切分模型，而是每张卡各自持有完整 actor。随后 DDP grad buffer 还要再申请大块显存，导致 60G NPU 上爆掉。

#### 修改点

在修好 Megatron 的 NPU-safe arch check 后，恢复：

- `--tensor-model-parallel-size 2`

并保持：

- actor 2 卡
- rollout 2 卡
- 非 colocate

#### 原理

TP=2 才能让 8B actor 在 2 张训练卡上分片持有，避免“每卡一整份模型”的显存浪费。

### 7. `vllm-ascend` 缺失 NPU distributed weight transfer 模块

#### 现象

第一次真正进入 actor->rollout 权重同步后，trainer 侧报：

```text
ModuleNotFoundError: No module named 'vllm_ascend.distributed.weight_transfer'
```

#### 根因

当前安装的：

- `/workspace/vllm-ascend/vllm_ascend/distributed/`

下只有：

- `device_communicators`
- `kv_transfer`
- `parallel_state`

但没有：

- `weight_transfer/`

而 vime 的非共卡 NPU 路径显式期待：

- `vllm_ascend.distributed.weight_transfer.hccl_engine`

#### 修改点

新增：

- `/workspace/vllm-ascend/vllm_ascend/distributed/weight_transfer/__init__.py`
- `/workspace/vllm-ascend/vllm_ascend/distributed/weight_transfer/hccl_engine.py`

补上：

- `HCCLWeightTransferEngine`
- `HCCLTrainerSendWeightsArgs`

#### 原理

对齐 upstream vLLM 的 `NCCLWeightTransferEngine` 抽象，但将底层通信从 CUDA NCCL 切换为 Ascend HCCL / `PyHcclCommunicator`。

### 8. NPUWorker 缺少 weight-transfer 生命周期 RPC

#### 现象

即便 trainer 侧补了 HCCL engine，engine 侧 HTTP 仍然返回：

```text
Method 'init_weight_transfer_engine' is not implemented.
```

调用落在：

- `/workspace/vllm/vllm/v1/serial_utils.py:505`

#### 根因

upstream GPU worker 有：

- `init_weight_transfer_engine`
- `start_weight_update`
- `update_weights`
- `finish_weight_update`

但 `vllm-ascend` 的 `NPUWorker` 没实现这组接口，因此 `collective_rpc` 找不到方法。

#### 修改点

修改：

- `/workspace/vllm-ascend/vllm_ascend/worker/worker.py`

新增：

- `_check_weight_transfer_engine()`
- `init_weight_transfer_engine()`
- `start_weight_update()`
- `update_weights()`
- `finish_weight_update()`
- `shutdown()`

#### 原理

把 upstream GPU worker 的 weight-transfer 生命周期语义迁移到 NPU worker，使 rollout engine 能真正响应 RLHF / async 分卡所需的 HTTP control-plane 调用。

### 9. NPUWorker 不能走 upstream 的 CUDA-only weight-transfer factory 逻辑

#### 现象

虽然已经新增了 HCCL engine 文件，但如果 NPU worker 仍通过 upstream factory 走默认 `backend == "nccl"` 的逻辑，它会继续倾向于实例化 CUDA 的 `NCCLWeightTransferEngine`，从而与 NPU runtime 不匹配。

#### 修改点

修改：

- `/workspace/vllm-ascend/vllm_ascend/worker/worker.py`

逻辑改为：

- 若 `weight_transfer_config.backend == "nccl"`
  - 在 NPU worker 上**直接实例化**新增的 `HCCLWeightTransferEngine`
- 其他 backend 仍可走 upstream factory

#### 原理

这里的 `backend == "nccl"` 在 vime/vllm 的高层协议里语义更接近“distributed weight transfer path”，但在 NPU worker 上其真实底层实现必须是 HCCL。通过 worker 侧重定向，可以保持控制面协议不变，同时把数据面适配到 Ascend runtime。

---

修完这些以后，训练已经能推进到：

- actor 2 卡 TP=2 初始化成功
- checkpoint 从 `/home/l00830933/images/Qwen3-8B_torch_dist` 成功加载
- 两个 rollout engine 启动成功
- `/init_weight_transfer_engine` 已返回 `200 OK`

现在的主要阻塞已经收缩为：**第一次 NPU distributed weight transfer 的 HCCL communicator 初始化失败**。

---

## 当前失败现象

最近一次验证中，训练失败于第一次 `actor_model.update_weights()`：

- 入口：`/workspace/vime/train_async.py:29`
- 训练 actor 侧：`/workspace/vime/vime/backends/megatron_utils/actor.py:671`
- 分布式权重同步入口：`/workspace/vime/vime/backends/megatron_utils/update_weight/update_weight_from_distributed.py:123`

最终异常：

```text
RuntimeError: HCCL error: parameter error
```

调用栈：

```text
connect_rollout_engines_from_distributed()
  -> HCCLWeightTransferEngine.trainer_init(...)
    -> HCCLWeightTransferEngine._stateless_init_process_group(...)
      -> PyHcclCommunicator(pg, device=device)
        -> hcclCommInitRank(...)
          -> RuntimeError: HCCL error: parameter error
```

对应文件：

- `/workspace/vime/vime/backends/megatron_utils/update_weight/update_weight_from_distributed.py:351-435`
- `/workspace/vllm-ascend/vllm_ascend/distributed/weight_transfer/hccl_engine.py`
- `/workspace/vllm-ascend/vllm_ascend/distributed/device_communicators/pyhccl.py:98-127`
- `/workspace/vllm-ascend/vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py:223-226`

---

## 参数传递链（详细）

### 1. 训练脚本层

`/workspace/run_qwen3_8b_npu0_3.sh` 以异步非共卡方式启动：

- `python /workspace/vime/train_async.py`
- `--actor-num-gpus-per-node 2`
- `--rollout-num-gpus 2`
- `--rollout-num-gpus-per-engine 1`
- `--tensor-model-parallel-size 2`
- `--rollout-backend vllm`
- `--vllm-weight-sync-mode native`

### 2. async train loop 入口

`/workspace/vime/train_async.py:26-29`

- `create_training_models(args, pgs, rollout_manager)`
- `actor_model.update_weights()`

这里第一次把 actor 权重推送给 rollout engine。

### 3. actor 侧权重同步入口

`/workspace/vime/vime/backends/megatron_utils/actor.py:650-684`

关键逻辑：

- `rollout_manager.get_updatable_engines_and_lock()`
- `self.weight_updater.connect_rollout_engines(...)`
- `self.weight_updater.update_weights()`

对于非 `colocate`，权重同步器选型为：

`/workspace/vime/vime/backends/megatron_utils/actor.py:192-196`

```python
if self.args.colocate:
    update_weight_cls = UpdateWeightFromTensor
else:
    update_weight_cls = UpdateWeightFromDistributed
```

因此异步分卡路径必然走：

- `UpdateWeightFromDistributed`

### 4. distributed weight sync 建链

`/workspace/vime/vime/backends/megatron_utils/update_weight/update_weight_from_distributed.py:94-129`

`UpdateWeightFromDistributed.connect_rollout_engines()` 会调用：

- `connect_rollout_engines_from_distributed(...)`

### 5. distributed trainer / engine group 参数构造

`/workspace/vime/vime/backends/megatron_utils/update_weight/update_weight_from_distributed.py:351-435`

核心逻辑：

```python
world_size = sum(engine_gpu_counts) + 1  # +1 for training rank 0
```

当前 2 个 rollout engine，每个 `rollout_num_gpus_per_engine = 1`，因此：

- `engine_gpu_counts = [1, 1]`
- `world_size = 3`

然后对每个 rollout engine 下发：

```python
engine.init_weights_update_group.remote(
    master_address=master_address,
    master_port=master_port,
    rank_offset=cumulative[i] + 1,
    world_size=world_size,
    group_name=group_name,
    backend=backend,
)
```

其中：

- engine 0: `rank_offset = 1`
- engine 1: `rank_offset = 2`
- backend = `hccl`

同时 trainer 侧调用：

```python
HCCLWeightTransferEngine.trainer_init(
    {
        "master_address": master_address,
        "master_port": master_port,
        "world_size": world_size,
    }
)
```

注意：trainer 侧 rank 固定为 `0`。

### 6. engine HTTP 控制面

`/workspace/vime/vime/backends/vllm_utils/vllm_engine.py:955-979`

每个 rollout engine 的 `init_weights_update_group()` 最终会通过 HTTP 调用：

- `POST /init_weight_transfer_engine`

payload:

```json
{
  "init_info": {
    "master_address": "<host>",
    "master_port": <port>,
    "rank_offset": <1 or 2>,
    "world_size": 3
  }
}
```

在最近一次运行中，这一步已经成功返回 `200 OK`，说明：

- engine 端 `init_weight_transfer_engine`
- engine 端 `NPUWorker.init_weight_transfer_engine`
- engine 端 `HCCLWeightTransferEngine.init_transfer_engine`

已经不再是缺实现状态。

### 7. engine 端 worker 内部 rank 计算

新增的 NPU 实现位于：

- `/workspace/vllm-ascend/vllm_ascend/distributed/weight_transfer/hccl_engine.py`
- `/workspace/vllm-ascend/vllm_ascend/worker/worker.py`

其中 engine 侧 rank 计算逻辑为：

```python
dp_rank = self.parallel_config.data_parallel_index
world_size_per_dp = self.parallel_config.world_size
rank_within_dp = self.parallel_config.rank
worker_rank = dp_rank * world_size_per_dp + rank_within_dp
rank = worker_rank + init_info.rank_offset
```

对于单卡 rollout engine，理论上 `worker_rank = 0`，因此：

- engine 0 rank = `1`
- engine 1 rank = `2`

这与 trainer rank `0` 对应的三方 world 一致。

### 8. trainer 端 communicator 初始化

trainer 侧使用新增的：

- `/workspace/vllm-ascend/vllm_ascend/distributed/weight_transfer/hccl_engine.py`

逻辑：

```python
pg = StatelessProcessGroup.create(
    host=master_address,
    port=master_port,
    rank=0,
    world_size=world_size,
)
return PyHcclCommunicator(pg, device=device)
```

而 `PyHcclCommunicator` 内部会：

1. 从 `StatelessProcessGroup` 读出：
   - `rank`
   - `world_size`
2. 通过 `broadcast_obj()` 广播 `hcclUniqueId`
3. 调用：

```python
self.hccl.hcclCommInitRootInfo(world_size, unique_id, rank, ...)
```

在这一步报出：

```text
HCCL error: parameter error
```

---

## 这次修改逻辑（按阶段）

### 阶段 1：脚本与运行时前置问题修复

已修复：

- netrc 登录识别
- W&B 运行依赖缺失
- MindSpeed runtime args 缺省字段
- Megatron NPU 下 CUDA-only 架构探测
- TP=1 导致 actor OOM

### 阶段 2：补齐 vllm-ascend 的 NPU weight transfer 运行时

新增：

- `/workspace/vllm-ascend/vllm_ascend/distributed/weight_transfer/__init__.py`
- `/workspace/vllm-ascend/vllm_ascend/distributed/weight_transfer/hccl_engine.py`

目标：

- 对齐 vLLM upstream 的 `NCCLWeightTransferEngine` 结构
- 提供 NPU 版：
  - `HCCLWeightTransferEngine`
  - `HCCLTrainerSendWeightsArgs`

### 阶段 3：让 NPUWorker 真正支持 weight-transfer 生命周期

修改：

- `/workspace/vllm-ascend/vllm_ascend/worker/worker.py`

新增：

- `init_weight_transfer_engine()`
- `start_weight_update()`
- `update_weights()`
- `finish_weight_update()`
- `shutdown()`

并在初始化时：

- 对 `backend == "nccl"` 直接实例化 `HCCLWeightTransferEngine`
- 不再依赖 upstream factory 的 CUDA-only 路径

### 阶段 4：恢复异步非共卡拓扑

训练脚本恢复为：

- actor 2 卡
- rollout 2 卡
- TP=2
- 非 `colocate`

这是为了继续验证真正的异步分卡链路，而不是退回 colocate IPC 路径。

---

## 当前判断：为什么还会报 HCCL parameter error

当前最可能的原因已经不再是“接口不存在”，而是：

1. **trainer / engine 三方 HCCL rank/world_size 仍有不一致**
   - trainer 固定 rank 0
   - 两个 engine 分别被认为是 rank 1 / 2
   - 需要进一步核实 `parallel_config.rank` / `world_size` 在 NPU worker 里的实际值

2. **StatelessProcessGroup + PyHcclCommunicator 这套组合在当前 Ascend 版本下可能对参数更敏感**
   - world size = 3 是否被当前 HCCL runtime 正确支持
   - rank_offset 传递到 engine 侧后是否与实际 worker rank 冲突

3. **当前 PyHcclCommunicator 假设的 group 初始化方式，可能与 vLLM-ascend 其余分布式路径的 group 约定不同**
   - 当前 vLLM-ascend 主要已有能力集中在：
     - `device_communicators`
     - `parallel_state`
     - `kv_transfer`
   - `weight_transfer` 是这次新补的能力，尚未与现有 HCCL PG 复用逻辑充分对齐

---

## 接下来建议的调试方向

1. 在 `HCCLWeightTransferEngine.init_transfer_engine()` 和 `trainer_init()` 中增加更细日志：
   - `rank`
   - `world_size`
   - `rank_offset`
   - `parallel_config.rank`
   - `parallel_config.world_size`
   - 当前 NPU device

2. 在 `PyHcclCommunicator.__init__()` 中打印：
   - 进入前 rank / world_size
   - unique id 广播是否成功
   - `hcclCommInitRootInfo()` 的参数值

3. 重点验证 engine 侧是否真的满足：
   - trainer rank = 0
   - engine0 rank = 1
   - engine1 rank = 2
   - world_size = 3

4. 若 HCCL 对这种 trainer + 2 单卡 rollout 的 `StatelessProcessGroup` 组合不兼容，则需要改成：
   - 使用 vllm-ascend 现有 distributed primitive 的 group 构建方式
   - 或调整为与 engine 端现有 `collective_rpc` / worker group 完全一致的 rank 体系

---

## 当前结论

当前异步分卡训练已经推进到：

- actor 初始化通过
- checkpoint 加载通过
- vLLM rollout server 启动通过
- `/init_weight_transfer_engine` HTTP 已返回 200

**最新唯一核心阻塞：trainer 侧使用新增 HCCL weight-transfer engine 建立 communicator 时，`hcclCommInitRootInfo` 报 `parameter error`。**

这说明问题已经收敛到 **HCCL communicator 初始化参数链**，不再是高层脚本或功能缺失问题。

---

## 后续新增修复与当前运行状态（本轮对话后半段）

在上面的分析之后，又继续解决了两批关键问题，使异步分卡训练从“卡在第一次权重同步初始化”推进到了“已经持续执行多轮 rollout + train step”。

### 10. Trainer / engine HCCL communicator 初始化已打通

#### 现象

在补齐 `HCCLWeightTransferEngine` 和 NPU worker RPC 生命周期之后，第一次 `/init_weight_transfer_engine` 已经不再报 `Method not implemented`，但 trainer 侧最开始仍然会在：

```text
hcclCommInitRootInfo(...)
```

报：

```text
RuntimeError: HCCL error: parameter error
```

#### 修复方向

为了让 engine 侧与 trainer 侧的权重同步实现真正对齐，进行了两步修复：

1. `vllm-ascend` 新增 HCCL weight transfer engine：
   - `/workspace/vllm-ascend/vllm_ascend/distributed/weight_transfer/hccl_engine.py`
2. `NPUWorker` 不再把 `backend == "nccl"` 交给 upstream CUDA worker factory，而是直接实例化新增的 `HCCLWeightTransferEngine`：
   - `/workspace/vllm-ascend/vllm_ascend/worker/worker.py`

#### 原理

这里的高层配置字段仍然叫 `backend = "nccl"`，但在 NPU worker 上它必须被解释为“分布式权重同步路径”，其底层真实实现必须替换为 HCCL。否则即使 control plane 正常，data plane 仍会落入 CUDA-only 路径。

#### 结果

后续日志里已经可以看到：

- trainer 侧：
  - `vLLM in-process weight transfer: addr=... world_size=3 device=0`
  - `Found hccl from library libhccl.so`
  - `vLLM is using pyhccl`
- engine 侧：
  - `POST /init_weight_transfer_engine HTTP/1.1" 200 OK`

说明初始化能力已经比之前更进一步，engine 和 trainer 已经都进入了 HCCL weight-transfer 路径。

### 11. Packed HCCL 路径未实现，切换到 non-packed 路径

#### 现象

在 HCCL 初始化进一步成功后，第一次真正进入权重广播阶段时，日志报：

```text
NotImplementedError: Packed HCCL weight transfer is not implemented yet.
```

原因是：

- vime 的分布式权重同步器在 `UpdateWeightFromDistributed._use_vllm_packed()` 判断通过后，会优先走 packed bucket 广播
- 但新增的 `HCCLWeightTransferEngine` 只实现了简单逐 tensor 广播，没有实现 packed producer / consumer

#### 修改点

修改训练脚本：

- `/workspace/run_qwen3_8b_npu0_3.sh`

新增：

```bash
--no-vllm-weight-sync-packed
```

#### 原理

先强制禁用 packed weight sync，让训练走已经实现的简单 HCCL broadcast 路径。这样可以优先验证“异步分卡训练主链路是否可跑通”，而不把 bring-up 依赖在 packed 优化路径上。

#### 结果

这一步之后，日志里已经能看到完整的第一轮权重同步 HTTP 生命周期：

- `POST /init_weight_transfer_engine` -> `200 OK`
- `POST /start_weight_update` -> `200 OK`
- `POST /update_weights` -> `200 OK`
- `POST /finish_weight_update` -> `200 OK`
- `POST /resume` -> `200 OK`

这意味着：

**第一次 actor -> rollout 的 non-packed HCCL 权重同步已经成功打通。**

### 12. 异步分卡训练已经真正跑起来了

#### 现象

在成功走完第一轮 non-packed HCCL 权重同步后，训练不再停在 bring-up，而是进入了真正的循环：

- rollout router 注册了两个 worker：
  - `http://80.48.5.56:15002`
  - `http://80.48.5.56:15006`
- `RolloutManager` 已开始连续执行 `generate`
- `MegatronTrainRayActor.train` 已经持续执行多个 train step

#### 已确认的运行证据

日志中已经反复出现：

- `rollout.py:1257 - perf N: {...}`
- `data.py:237 - rollout N: {...}`
- `timer.py:32 - Timer train end (elapsed: ...)`
- `train_metric_utils.py:44 - perf N: {...}`

而且 `pgrep` 结果显示：

- `train_async.py` 进程仍在
- `RolloutManager.generate` 在跑
- 两个 `MegatronTrainRayActor.train` 在跑
- 两个 `VLLM::EngineCore` 在跑

#### 原理

这表明异步分卡训练的“主链路”已经打通：

1. actor init
2. checkpoint load
3. rollout engine startup
4. HCCL weight sync
5. resume generation
6. rollout generation
7. actor train step
8. 下一轮 update_weights / rollout

也就是说，当前系统已经不再处于“修 bring-up”阶段，而是进入了“训练质量 / 效果优化”阶段。

### 13. 当前已确认的训练指标与质量问题

#### 已观测到的训练指标

从当前运行日志中已经可以看到至少 `rollout 0` 到 `rollout 11`：

- `rollout/raw_reward` 已多次为非零，例如：
  - `0.4375`
  - `0.3125`
  - `0.25`
  - `0.125`
  - `0.0625`
- `perf/update_weights_time` 大约：
  - `1.6s ~ 3.2s`
- `perf/actor_train_time` 大约：
  - `54s ~ 79s`
- `perf/actor_train_tok_per_s` 大约：
  - `850 ~ 1254 tok/s`

这说明：

- actor 训练在持续进行
- 权重同步在持续成功发生
- rollout 结果在被训练侧消费

#### 当前主要质量问题

虽然训练链路已经跑通，但样本质量仍不理想：

- `rollout/response_lengths` 频繁接近或等于 `4096`
- `rollout/truncated_ratio` 经常在：
  - `0.75`
  - `0.8125`
  - `0.875`
  - `1.0`
- `rollout/rewards` 仍然经常接近 `0`
- `rollout/raw_reward` 虽然已不再长期全 0，但仍偏稀疏

#### 原理

这说明当前阶段的主要问题已经从：

- 环境依赖
- 运行时缺实现
- actor 初始化
- distributed weight sync

转移为：

- **response 仍然经常撞 `--rollout-max-response-len 4096` 上限**
- **reward 信号虽然已经不再“全死”，但仍然偏弱**

即：

> 现在训练已经跑通，但训练质量仍受长推理被截断影响，后续优化重点应放在降低截断率、提升 reward 信号质量。

### 14. 当前运行状态总结

截至当前文档更新时间，最新 run：

- `/workspace/vime/runs/qwen3_8b_npu0_3_20260706_092740`

已经确认：

- 异步分卡训练主进程在运行
- rollout engine 在运行
- HCCL 非 packed 权重同步成功
- rollout generation 已持续执行多轮
- actor training 已持续执行多轮

因此当前状态可总结为：

**异步分卡训练已经真正跑起来了。当前主要剩余问题不是 bring-up，而是较高的 truncation ratio 与偏弱的 reward signal。**


---

## 本次对话完整问题清单、修复过程与当前训练方案说明

本节的目标不是只给出“最后改了哪些参数”，而是完整记录本次对话中实际遇到过的所有关键问题、每个问题的根因、修复方式、为什么这样修，以及当前已经跑通的异步分卡训练方案是如何启动的、依赖什么关键参数。

这样做的目的有两个：

1. 后续继续优化 reward / truncation 时，不需要重新排查已经解决过的 bring-up 问题；
2. 即使更换执行人，也能从文档中直接理解当前训练方案的设计意图与运行约束。

### A. 本次对话里遇到的核心问题总览

按实际出现顺序，本次对话中遇到的问题可以分成 4 个层级：

1. **监控 / 运行环境问题**
   - W&B 登录检测误判
   - W&B Python 依赖缺失

2. **训练框架兼容问题**
   - Megatron 在 NPU 环境误调用 CUDA API
   - MindSpeed `repatch()` 缺少默认参数命名空间

3. **拓扑 / 显存问题**
   - TP=1 临时绕过导致 8B actor 在 2 卡 actor 拓扑下整模型复制并 OOM

4. **异步分卡特有的 distributed weight transfer 问题**
   - `vllm-ascend` 缺失 distributed weight_transfer 模块
   - NPUWorker 没有 upstream vLLM 的 weight-transfer RPC 生命周期方法
   - NPU worker 仍会走 CUDA-only factory 路径
   - packed HCCL weight sync 未实现
   - HCCL communicator 初始化参数错误（历史阻塞，后已进一步推进）

这些问题的特点是：

- 前三类问题属于“运行前/初始化前/训练框架兼容性”问题；
- 最后一类才是“异步分卡训练真正独有”的问题；
- 只有把前三类问题清理干净，才有机会走到异步分卡链路的真正核心：
  **actor -> rollout 的 distributed weight sync。**

---

### B. 详细问题与解决过程

#### B1. W&B 登录检测误判

##### 现象

用户已经执行：

```bash
wandb login --relogin --host=http://127.0.0.1:8088
```

但：

```bash
wandb status
```

依然显示：

```json
"api_key": null
```

如果训练脚本继续使用：

```bash
wandb status | grep 'api_key'
```

这类逻辑做判断，就会把“已登录”误判成“未登录”。

##### 根因

W&B 当前实际凭据写在：

- `~/.netrc`

而不是：

- `~/.config/wandb/settings`

`wandb status` 展示的是 settings/env 视角，并不会把 netrc 里的凭据填回 `api_key` 字段。

##### 解决方式

修改训练脚本：

- `/workspace/run_qwen3_8b_npu0_3.sh`

加入：

- host 可达性检测
- netrc 凭据检测
- 无有效登录则 fail-fast
- 设置 `WANDB_BASE_URL`

##### 原理

不要再用 `wandb status` 的 `api_key` 字段来代表“是否已登录”，而是直接检查：

1. `WANDB_API_KEY`
2. `~/.netrc` 中是否存在目标 host 的凭据
3. host 是否可访问

##### 价值

这一步修完后，W&B 不再是“偶尔能连上”的不稳定状态，而是：

- 登录来源明确
- host 明确
- 不会在无监控的情况下静默裸跑训练

---

#### B2. W&B 缺少运行依赖 `sentry_sdk`

##### 现象

W&B 在 `wandb.init()` 时崩溃：

```text
ModuleNotFoundError: No module named 'sentry_sdk'
```

##### 根因

当前环境中安装了 `wandb`，但未安装其运行过程中实际触发到的 `sentry-sdk` 依赖。

##### 解决方式

安装：

```bash
python -m pip install sentry-sdk
```

##### 原理

这属于 Python runtime 依赖缺失，不是训练逻辑错误，也不是 W&B host 配置错误。

##### 价值

修复后：

- `wandb.init()` 能完成
- W&B run 能实际建立
- 后续 rollout / train 指标才有地方记录

---

#### B3. Megatron 在 NPU-only 环境误调用 CUDA API

##### 现象

切到 async 分卡 + TP>1 后，Megatron 参数校验阶段崩溃：

```text
AssertionError: Torch not compiled with CUDA enabled
```

##### 根因

Megatron 的 `get_device_arch_version()` 直接写死：

```python
torch.cuda.get_device_properties(torch.device("cuda:0"))
```

但当前是 NPU-only 构建，没有 CUDA backend。

##### 解决方式

修改：

- `/workspace/Megatron-LM/megatron/training/utils.py`

让 `get_device_arch_version()`：

- 若 `torch.cuda.is_available()` 为真，则走原逻辑
- 否则返回 `0`

##### 原理

该函数的目的只是为了让 Megatron 选择一组与“旧架构/新架构”相关的策略，不应该在 NPU-only 场景里直接把训练杀死。返回 `0` 等价于走保守路径。

##### 价值

这一步修复后，训练终于可以在 **TP=2** 条件下继续跑，而不是被迫退回 TP=1。

---

#### B4. MindSpeed `repatch()` 缺少默认参数，先后暴露 `optimization_level` / `optimizer_selection`

##### 现象

actor 初始化时先报：

```text
AttributeError: 'Namespace' object has no attribute 'optimization_level'
```

随后修这个字段后，又报：

```text
AttributeError: 'Namespace' object has no attribute 'optimizer_selection'
```

##### 根因

`MindSpeed.repatch(args)` 的逻辑假设：

- `get_full_args()` 返回的对象已经拥有完整的 MindSpeed CLI 默认参数

但在 Ray actor 初始化路径中，这个假设不成立。它拿到的 `Namespace` 并没有完整默认字段，只含部分运行时参数。

##### 第一轮修复（不充分）

在：

- `/workspace/MindSpeed/mindspeed/megatron_adaptor.py`

里只给 `optimization_level` 加了兜底默认值。

##### 第二轮正式修复

进一步发现问题是“默认字段全集缺失”，于是改为：

1. `get_full_args()`
2. `get_mindspeed_args(get_defaults=True)`
3. 先把默认 MindSpeed args 全量 merge 到 `full_args`
4. 再覆盖 runtime args

##### 原理

不要修一个字段补一个字段，而是把 **完整默认参数命名空间**补进去。这样 feature manager 在任何路径下都能拿到完整语义。

##### 价值

这一步修完后：

- actor init 不再被 MindSpeed patch manager 卡死
- Ray actor 路径与常规 CLI 路径行为更一致

---

#### B5. TP=1 绕过虽然能过初始化，但导致 actor OOM

##### 现象

最初为了绕过 CUDA-only 检查，把 TP 从 2 降到 1，结果 actor 初始化时 OOM：

```text
torch.OutOfMemoryError: Tried to allocate 30.51 GiB
```

##### 根因

TP=1 让 2 张 actor 卡不再分片持有模型，而是每张卡各自持有完整 8B actor，再叠加 DDP grad buffer 分配，显存被打爆。

##### 解决方式

在修好 Megatron 的 NPU-safe arch check 后，把训练脚本恢复为：

- `--tensor-model-parallel-size 2`

##### 原理

对于当前 2 actor NPU 拓扑，TP=2 是必要条件，否则无法真正把模型切开。

##### 价值

这一步直接决定了当前异步分卡方案在 2 actor 卡上是否具备显存可行性。

---

#### B6. `vllm-ascend` 缺失 NPU distributed weight transfer 模块

##### 现象

第一次真正进入 actor->rollout distributed weight sync 时，trainer 侧 import 失败：

```text
ModuleNotFoundError: No module named 'vllm_ascend.distributed.weight_transfer'
```

##### 根因

当前安装的 `vllm-ascend` 只有：

- `device_communicators`
- `kv_transfer`
- `parallel_state`

没有：

- `distributed/weight_transfer/`

而 vime 非共卡 NPU 路径明确依赖：

- `vllm_ascend.distributed.weight_transfer.hccl_engine`

##### 解决方式

新增：

- `/workspace/vllm-ascend/vllm_ascend/distributed/weight_transfer/__init__.py`
- `/workspace/vllm-ascend/vllm_ascend/distributed/weight_transfer/hccl_engine.py`

##### 原理

用 HCCL 实现一套与 upstream `NCCLWeightTransferEngine` 语义等价的 NPU 分布式权重同步引擎。

##### 价值

这是让非共卡异步分卡链路具备“数据面传权能力”的基础补丁。

---

#### B7. `NPUWorker` 缺少 weight-transfer RPC 生命周期方法

##### 现象

即使 trainer 侧补了 HCCL engine，engine 侧 HTTP 路由仍然报：

```text
Method 'init_weight_transfer_engine' is not implemented.
```

##### 根因

upstream GPU worker 有一套完整的：

- `init_weight_transfer_engine`
- `start_weight_update`
- `update_weights`
- `finish_weight_update`

但 `vllm-ascend` 的 `NPUWorker` 没有这几组方法。

##### 解决方式

修改：

- `/workspace/vllm-ascend/vllm_ascend/worker/worker.py`

补上：

- `_check_weight_transfer_engine()`
- `init_weight_transfer_engine()`
- `start_weight_update()`
- `update_weights()`
- `finish_weight_update()`
- `shutdown()`

##### 原理

engine 侧必须能响应 trainer 的 weight-transfer 控制面请求，否则即便 HCCL / trainer / metadata 都准备好了，也根本没法开始收权重。

##### 价值

这一步修完后，`/init_weight_transfer_engine` 已经可以返回 `200`，说明 engine 侧接口打通。

---

#### B8. NPU worker 不能走 upstream 的 CUDA-only weight transfer factory

##### 现象

虽然已经给 `vllm-ascend` 新增了 HCCL engine 文件，但如果 NPU worker 仍通过 upstream factory 创建 `backend == "nccl"`，它倾向于实例化 CUDA 的 `NCCLWeightTransferEngine`。

##### 根因

上层协议字段仍然叫 `backend = "nccl"`，但在 NPU 设备上，这个字段语义应该是“分布式权重同步”，而不是“使用 CUDA NCCL 实现”。

##### 解决方式

修改：

- `/workspace/vllm-ascend/vllm_ascend/worker/worker.py`

逻辑改为：

- 若 `weight_transfer_config.backend == "nccl"`
  - 直接实例化 `HCCLWeightTransferEngine`
- 其他 backend 再走 upstream factory

##### 原理

保持控制面配置兼容 vLLM/vime 上层协议，但在 NPU worker 内部把底层数据面重定向到 HCCL。

##### 价值

这是让 `NPUWorker` 真正使用新增 HCCL engine 的关键一步，否则新增文件只是“存在”，但不会被实际执行到。

---

#### B9. HCCL communicator 初始化参数错误

##### 现象

在补完 HCCL engine 和 NPU worker 方法后，训练一度前进到首次 communicator 建链阶段，但 trainer 侧报：

```text
RuntimeError: HCCL error: parameter error
```

##### 根因

此时 engine 侧和 trainer 侧都已经进入 HCCL path，但 `StatelessProcessGroup + PyHcclCommunicator` 这套 rank/world_size 组合还未完全和当前 2 actor + 2 rollout 单卡 engine 的运行方式对齐。

##### 解决方向

继续沿 trainer 侧 rank=0、engine rank_offset=1/2、world_size=3 的参数链打通，让 engine 侧与 trainer 侧建 group 参数真正对齐。

##### 结果

后续日志已经能看到：

- `POST /init_weight_transfer_engine` -> `200 OK`
- trainer 和 engine 都打印 `vLLM is using pyhccl`

说明已经从“模块缺失”推进到“真正建 HCCL 通信器”的层面。

---

#### B10. Packed HCCL weight sync 未实现

##### 现象

在 HCCL communicator 进一步打通后，第一次真正广播权重时又报：

```text
NotImplementedError: Packed HCCL weight transfer is not implemented yet.
```

##### 根因

vime 默认在某些条件下会启用 packed weight sync，但新增的 HCCL engine 只实现了非 packed 逐 tensor 广播路径。

##### 解决方式

在训练脚本中加入：

```bash
--no-vllm-weight-sync-packed
```

##### 原理

先禁用 packed 优化路径，优先验证主链路（non-packed HCCL broadcast）是否能跑通。

##### 价值

这一步之后，第一次完整权重同步生命周期已经全部成功：

- `/init_weight_transfer_engine` 200
- `/start_weight_update` 200
- `/update_weights` 200
- `/finish_weight_update` 200
- `/resume` 200

这标志着：

**异步分卡训练的第一轮 actor->rollout 权重同步主链路已经成功打通。**

---

### C. 当前训练方案的启动方式

当前已经实际跑起来的训练方案是：

- **训练入口**：`/workspace/run_qwen3_8b_npu0_3.sh`
- **训练主程序**：`/workspace/vime/train_async.py`
- **训练模式**：异步、非共卡、分卡训练
- **硬件分配**：
  - actor: NPU 0-1
  - rollout: NPU 2-3
- **actor TP**：2
- **rollout engine 数量**：2 个
- **每个 rollout engine 使用 1 张卡**

从结构上看，这套方案是：

1. actor 侧先完成 checkpoint 加载
2. actor 通过 HCCL 向 rollout engine 推送新权重
3. rollout engine 恢复生成
4. RolloutManager 采样样本并给出 reward
5. actor 用这批 rollout 数据训练
6. 下一轮再同步权重

这正是用户要求的“异步、分卡”的核心能力。

---

### D. 当前关键启动参数解释

下面只解释当前方案里真正关键、且与训练是否能跑通 / 训练质量是否合理直接相关的参数。

#### D1. 拓扑与模式相关参数

##### `python /workspace/vime/train_async.py`

表示使用异步训练主循环，而不是同步版 `train.py`。

作用：
- 允许当前 rollout 的训练与下一轮 rollout 数据准备在流程上重叠
- 是实现“异步分卡”能力的入口

##### `--actor-num-gpus-per-node 2`

actor 训练使用 2 张 NPU。

作用：
- 让 8B actor 能以 TP=2 分片加载
- 避免 TP=1 时整模型复制导致 OOM

##### `--rollout-num-gpus 2`

rollout 总共使用 2 张 NPU。

作用：
- 提供两个单卡 rollout engine
- 与 actor 2 卡一起组成 2+2 分卡拓扑

##### `--rollout-num-gpus-per-engine 1`

每个 rollout engine 使用 1 张卡。

作用：
- 当前总 rollout 卡数为 2，因此形成 2 个 engine
- 便于 consistent-hash router 把不同 session 路由到两个 worker

##### `--tensor-model-parallel-size 2`

actor 训练 TP=2。

作用：
- 把 8B actor 切在两张 actor 卡上
- 是当前 2 actor NPU 拓扑下避免 OOM 的必要条件

#### D2. reward / rollout 质量相关参数

##### `--rm-type math`

使用 math reward，而不是 `deepscaler`。

作用：
- 去掉 `deepscaler` 对 `</think>` / `###Response` 的额外门控
- 直接按数学答案抽取与判分逻辑工作
- 避免“模型答了但 reward 被静默打成 0”

##### `--rollout-max-response-len 4096`

单条 response 最长允许 4096 token。

作用：
- 比最初的 1024 大很多，已经显著改善“全 1024 截断”问题
- 但当前日志表明，4096 仍然经常被打满，说明后续很可能还需要进一步调大

##### `--vllm-max-model-len 12288`

vLLM 侧总 context budget。

作用：
- 为 prompt + response 提供更大的总长度空间
- 与 `rollout-max-response-len` 配合，避免 prompt 较长时过早打爆序列长度

##### `--n-samples-per-prompt 4`

每个 prompt 采样 4 个 response。

作用：
- 提高 group 内 reward 差异的机会
- 对 GRPO 类训练尤其重要

##### `--rollout-batch-size 4`

每次 rollout 取 4 组 prompt。

作用：
- 与 `n-samples-per-prompt=4` 配合，形成当前的采样/训练批结构

#### D3. 异步分卡权重同步相关参数

##### `--vllm-weight-sync-mode native`

告诉 vime/vLLM 使用原生 weight-transfer API 路径。

作用：
- 触发当前这条 distributed weight sync control/data plane 链路
- 是异步分卡模式下 actor -> rollout 同步的关键开关之一

##### `--no-vllm-weight-sync-packed`

显式关闭 packed weight sync。

作用：
- 避免进入尚未在当前 NPU HCCL engine 中实现的 packed 路径
- 强制走 non-packed 逐 tensor HCCL broadcast 路径

#### D4. 显存与推理稳定性相关参数

##### `--vllm-gpu-memory-utilization 0.40`

每个 rollout engine 给 vLLM 使用约 40% 的 NPU 显存预算。

作用：
- 为 rollout engine 留足够空间
- 同时避免挤压 actor/graph/缓存

当前日志里 vLLM 还给出了提示：
- 若后续要更充分利用 rollout NPU，可考虑略微上调到接近 `0.4096`

##### `--vllm-enable-sleep-mode`

启用 vLLM sleep/wake 机制。

作用：
- 配合权重更新周期在需要时进入 weight update 状态
- 是当前 vLLM weight update 生命周期的一部分

##### `--no-offload-train`

训练侧不做额外 offload。

作用：
- 降低某些 NPU 下训练 actor 侧 offload / reload 的不稳定因素
- 当前 bring-up 阶段优先保证主链路稳定

##### `--no-offload-rollout`

rollout 侧不做额外 offload。

作用：
- 避免 rollout engine 在当前阶段再叠加不必要的 offload 状态复杂度
- 先让异步分卡主链路跑通

---

### E. 当前已经跑通到什么程度

截至当前文档更新时，当前 run：

- `/workspace/vime/runs/qwen3_8b_npu0_3_20260706_092740`

已经确认：

1. 异步分卡主进程在运行
2. actor 2 卡 TP=2 正常工作
3. rollout 2 卡 / 2 engine 正常工作
4. non-packed HCCL 权重同步成功
5. rollout router 正常调度到两个 worker
6. rollout generation 已持续执行多轮
7. actor train 已持续执行多轮
8. W&B 正常记录

也就是说：

> **异步分卡训练主链路已经真正跑起来了。**

这次对话解决的重点已经从“bring-up 能否通过”转成“训练质量如何提升”。

---

### F. 当前剩余问题

当前最主要的剩余问题，不再是训练起不来，而是：

1. **截断率仍偏高**
   - `rollout/truncated_ratio` 经常在 `0.75 ~ 1.0`
   - `response_lengths` 仍大量撞 `4096`

2. **reward 信号虽然已不再全死，但仍偏弱/偏稀疏**
   - `rollout/raw_reward` 已多次非零
   - 但很多轮 `rollout/rewards` 仍接近 0

3. **后续优化重点**
   - 适度提高 `rollout-max-response-len`
   - 联动提高 `vllm-max-model-len`
   - 结合显存与吞吐重新平衡 batch / seq 配置

---

## 本文档用途

这份文档现在应该被视为当前异步分卡 NPU 训练 bring-up 的主记录。后续再调训练质量时，应先把这里记录的已解决问题视为“已排除项”，避免重复回到：

- W&B 登录 / 依赖
- Megatron CUDA-only 检查
- MindSpeed runtime args 缺省
- TP=1 OOM
- 缺失 HCCL engine
- 缺失 NPUWorker weight-transfer RPC
- packed HCCL 未实现

这些已经在本次对话中被逐步解决。当前阶段的关注点已经是：

**如何在训练已经跑通的前提下，进一步降低 truncation、增强 reward 信号，并观察最终 critic/reward 曲线表现。**


---

## 新的 12 卡 Qwen3.6-35B-A3B 异步分卡训练方案

用户后续要求切换到：

- `--hf-checkpoint /home/weight/Qwen3.6-35B-A3B`
- `--ref-load /home/weight/Qwen3.6-35B-A3B_torch_dist`
- 使用物理 NPU `4-15` 共 12 张卡
- 仍然采用 **异步、分卡** 的方式启动训练
- 优先缓解当前 4 卡原型中“response 经常撞 4096 上限，reward 曲线偏弱”的问题

因此，在当前 4 卡异步分卡原型已经跑通主链路的基础上，设计出一个更适合 35B A3B 的 12 卡方案。

### 方案目标

这个 12 卡方案的目标不是“先追求最高吞吐”，而是：

1. **优先保证 35B actor 训练侧并行配置可行且尽量复用已知稳定经验**
2. **保持非 colocate 的异步分卡模式**
3. **优先解决 reward 弱的核心原因：response 预算过小导致高截断率**
4. **避免再次进入 packed HCCL 这条尚未实现的路径**

---

### 为什么先解决 response 截断，而不是先改 reward 逻辑

当前 4 卡异步分卡原型已经证明：

- reward 链路本身已经通了（`raw_reward` 多次非零）
- 主链路不是“完全拿不到 reward”，而是“奖励信号偏弱、偏稀疏”
- 根本原因是：
  - 许多 response 长度已经撞到 `4096`
  - `truncated_ratio` 经常处于 `0.75 ~ 1.0`
  - 数学推理过长时，最终答案（例如 `oxed{}`）来不及输出

所以，针对 35B 的第一优先改动应该是：

- 增大 `--rollout-max-response-len`
- 同步增大 `--vllm-max-model-len`

这比继续换 reward 类型更直接，因为当前已经使用：

- `--rm-type math`

假零门控问题已经绕过。

---

### 推荐拓扑：8 actor + 4 rollout

当前推荐的 12 卡方案是：

- **actor：8 卡**（物理 NPU 4-11）
- **rollout：4 卡**（物理 NPU 12-15）
- **非 colocate**
- **异步 `train_async.py`**

#### 为什么不是 6+6 或 4+8

##### 不选 4 actor + 8 rollout
35B A3B 在现有仓库里更接近一条已知稳定 actor 路径：

- `tensor-model-parallel-size 1`
- `expert-model-parallel-size 8`

这意味着 actor 世界大小天然更适合 8 卡。如果只给 actor 4 卡，就必须重新设计 TP/EP 拆分，风险会明显上升。

##### 不选 6+6
6 卡不是当前 35B A3B 并行策略下一个自然的切分点：

- EP/TP 不整齐
- 仓库里缺少直接经验模板
- bring-up 成本更高

##### 选 8 actor + 4 rollout
这是一个折中的、最稳的 12 卡方案：

- actor 侧尽量复用现有 35B 稳定经验
- rollout 4 卡足够支撑一个 TP=4 的 vLLM engine
- 总卡数恰好 12
- 保持用户要求的异步分卡

---

### rollout 侧为什么推荐 1 个 4 卡 engine，而不是 4 个单卡 engine

对于 35B A3B：

- 模型体量远大于 8B
- 单卡 rollout engine 不现实
- 用 4 卡做一个 TP=4 的单 engine 更接近实际可运行方案

因此推荐：

```bash
--rollout-num-gpus 4
--rollout-num-gpus-per-engine 4
```

这样形成：

- **1 个 4 卡 rollout engine**

而不是：

- 4 个单卡 rollout engine

---

### actor 侧并行参数为什么这样选

当前 12 卡方案中 actor 侧推荐：

```bash
--actor-num-gpus-per-node 8
--tensor-model-parallel-size 1
--expert-model-parallel-size 8
--expert-tensor-parallel-size 1
--pipeline-model-parallel-size 1
--context-parallel-size 1
--sequence-parallel
```

#### 原理

这是尽量沿用仓库里已有 35B A3B 稳定 actor 配置思路：

- `TP=1`
- `EP=8`

相对于“为了追求更高理论吞吐而重构 TP/EP 拆分”，这种方案更适合当前目标：

> **优先让 35B 在 12 卡异步分卡模式下稳定起跑。**

这与当前 4 卡 8B 原型的逻辑不同：

- 8B actor 需要靠 TP=2 避免整模型复制 OOM
- 35B A3B 的已知稳定经验则更偏向 `TP=1 + EP=8`

所以 35B 不能简单照搬 8B 的 actor 并行方式。

---

### 为什么当前推荐 response 长度直接拉到 16K

为了优先缓解当前 reward 弱的问题，推荐：

```bash
--rollout-max-response-len 16384
--vllm-max-model-len 18432
```

#### 原理

当前 4 卡原型中：

- `4096` 已经经常打满
- truncation ratio 仍长期很高
- reward 信号虽然不再全死，但依然偏弱

因此对 35B 数学推理任务来说，`4096` 很可能仍远不够。直接把 response budget 拉到 `16K`，是为了优先验证：

- reward 是否明显改善
- truncation 是否显著下降
- 模型是否能把完整推理和最终答案输出出来

#### 风险

代价是：

- rollout 显存压力上升
- 单轮 rollout 时间可能上升

因此与之配套的 rollout batch 必须更保守。

---

### 为什么 rollout batch / global batch 要先保守设置

当前推荐：

```bash
--rollout-batch-size 2
--n-samples-per-prompt 4
--global-batch-size 8
--vllm-max-num-seqs 8
```

#### 原理

因为现在 rollout 只有：

- 1 个 4 卡 engine
- response 长度提升到了 16K

如果此时仍然保持过大的 batch：

- rollout 时间会暴涨
- KV cache 压力会明显上升
- 更容易碰到新的 OOM / 不稳定问题

因此第一版 12 卡方案强调：

> 先跑稳，再考虑扩 batch。

---

### 为什么继续保留 `--rm-type math`

当前已经确认：

- `deepscaler` 的 `</think>` / `###Response` 门控会带来假零问题
- `math` reward 已经绕过了这个问题
- 当前 reward 弱主要不是因为 reward 类型错了，而是因为 response 太短、答案出不来

因此 12 卡方案继续使用：

```bash
--rm-type math
```

而不是再切回 `deepscaler`。

---

### 为什么必须保留 `--no-vllm-weight-sync-packed`

在当前 runtime 修复状态下：

- non-packed HCCL 权重同步已经打通
- packed HCCL 路径仍未实现完整 producer / consumer

因此 12 卡方案里必须继续带：

```bash
--no-vllm-weight-sync-packed
```

#### 原理

这是一个 bring-up 保护参数：

- 它不一定是未来最优性能方案
- 但它是当前异步分卡训练可持续运行的必要条件之一

---

### 12 卡方案的实际启动脚本

已生成新的启动脚本：

- `/workspace/run_qwen36_35b_a3b_async_npu4_15.sh`

其核心设计为：

- 物理 NPU `4-15`
- actor 8 卡
- rollout 4 卡
- 非 colocate
- `train_async.py`
- `rm-type math`
- `rollout-max-response-len 16384`
- `vllm-max-model-len 18432`
- `--no-vllm-weight-sync-packed`
- `optimization-level 0`

---

### 推荐启动方式

启动命令：

```bash
bash /workspace/run_qwen36_35b_a3b_async_npu4_15.sh
```

#### 第一阶段建议

建议先用：

```bash
--num-rollout 80
```

做第一轮验证，而不是直接长跑。

原因：

- 这一步的重点是验证 35B 12 卡异步分卡 bring-up 和 reward 改善效果
- 不是第一时间追求完整 200 rollout

#### 第二阶段建议

如果前 5~10 个 rollout 已能稳定显示：

- reward 非零明显增多
- truncation ratio 下降
- 权重同步与 train step 都稳定

再把：

```bash
--num-rollout 80
```

提高到：

```bash
--num-rollout 200
```

---

### 当前对 12 卡方案的最终推荐结论

如果目标是：

> **先让 35B 在 12 卡上以异步分卡方式稳定启动，并优先改善 reward 弱问题**

那么当前最推荐的方案是：

- **8 actor + 4 rollout**
- **actor: TP1 + EP8**
- **rollout: 1 个 4 卡 engine（TP=4）**
- **math reward**
- **response 16K**
- **non-packed HCCL sync**
- **先 80 rollout 验证，再 200 rollout 长跑**

这套方案的核心思想是：

1. 训练 bring-up 尽量复用已知稳定 35B actor 经验；
2. rollout 先给足 response 预算，优先解决 reward 信号弱；
3. 分布式权重同步走已经验证成功的 non-packed HCCL 路径；
4. 稳定后再逐步调大 batch / rollout 强度，而不是一开始就追求极限配置。


---

## 当前阶段新增问题与修复总结（12 卡 35B 异步分卡 bring-up 继续推进）

本节补充的是在 4 卡异步分卡原型已跑通之后，把方案切换到：

- `Qwen3.6-35B-A3B`
- 12 张 NPU（物理 4-15）
- 异步、分卡、非 colocate

的过程中，继续遇到并已经分析/修复的问题。

这一阶段的问题已经明显呈现出新的特点：

- 不再主要是通用运行环境问题；
- 也不再主要是 W&B / Ray / HCCL control plane bring-up；
- 而是集中在 **35B A3B 的 MoE 模型路径、配置字段分布、rollout 模型实现选择，以及 35B rollout engine 显存 sizing** 上。

下面按问题逐条记录。

### 15. 35B A3B 方案再次触发 CUDA-only 检查，但原因不是“12 卡”，而是 `--moe-grouped-gemm`

#### 现象

12 卡 35B 脚本首次启动时再次出现：

```text
AssertionError: Torch not compiled with CUDA enabled
```

位置在：

- `/workspace/Megatron-LM/megatron/training/arguments.py:907`
- `/workspace/Megatron-LM/megatron/training/yaml_arguments.py:260`

#### 根因

这次不是之前 `get_device_arch_version()` 那条路径，而是 **MoE grouped GEMM 的能力校验**：

```python
if args.moe_grouped_gemm:
    dc = torch.cuda.get_device_capability()
```

而 35B A3B 的模型脚本：

- `/workspace/vime/scripts/models/qwen3.5-35B-A3B.sh`

明确打开了：

- `--moe-grouped-gemm`

8B 原型没有这个参数，所以 8B 从来不会走到这条分支。

#### 修改点

修改：

- `/workspace/Megatron-LM/megatron/training/arguments.py`
- `/workspace/Megatron-LM/megatron/training/yaml_arguments.py`

把 grouped-gemm 的 CUDA capability 检查包在：

```python
torch.cuda.is_available()
```

之后再执行。

#### 为什么这样能解决

因为这条检查本质上只对 **CUDA grouped GEMM kernel** 是否可用有意义。在 NPU-only 训练里，继续碰 `torch.cuda.get_device_capability()` 只会把训练提前杀死，而不会提供任何真实有效的保护。

#### 原理

这里的原则是：

> **把“设备能力判断”和“设备类型绑定”解耦。**

如果当前根本不是 CUDA 设备，就不应该继续进入 CUDA 特有能力探测。

---

### 16. 35B rollout 路径最初被解析成 `Qwen3MoeForCausalLM`，导致 checkpoint / module tree 不匹配

#### 现象

12 卡 35B bring-up 一路推进后，rollout 侧开始出现大量：

- missing config field
- missing weights
- q_norm / k_norm / layernorm / mlp 结构对不上

并且日志里显示 rollout 侧被解析成：

- `Resolved architecture: Qwen3MoeForCausalLM`

#### 根因

当前这份模型的 HF config / checkpoint 家族更接近：

- `Qwen3_5MoeConfig`
- `Qwen3_5MoeForConditionalGeneration`

而不是 rollout 最初自动选择的：

- `Qwen3MoeForCausalLM`
- 对应实现：`/workspace/vllm/vllm/model_executor/models/qwen3_moe.py`

这两套实现虽然都属于 Qwen MoE 家族，但它们的：

- 模块树
- HF 配置字段组织方式
- 权重键名

并不完全一致。

从 checkpoint 实际 key 可以看出：

- 存在大量 `model.language_model.layers.*...`
- 同时混合了：
  - `linear_attn.*`
  - `self_attn.*`
  - `mlp.experts.*`
  - `shared_expert.*`

而 `qwen3_moe.py` 期待的是它自己那套更接近 `Qwen3MoeForCausalLM` 的参数命名树。

#### 修改点

修改 12 卡启动脚本：

- `/workspace/run_qwen36_35b_a3b_async_npu4_15.sh`

加入：

```bash
--vllm-hf-overrides '{"architectures":["Qwen3_5MoeForConditionalGeneration"]}'
```

#### 为什么这样能解决

它的作用是强制 rollout 侧不要再按默认推断走 `Qwen3MoeForCausalLM`，而是走更贴近当前 checkpoint / config 家族的：

- `Qwen3_5MoeForConditionalGeneration`

#### 原理

这里的原则是：

> **如果 HF config / checkpoint 来自某个特定分支（qwen3_5_moe），优先让 rollout 侧显式匹配到同分支的 vLLM 模型实现，而不是依赖自动推断。**

---

### 17. `qwen3_moe.py` 对 35B HF config 的字段假设不够鲁棒：`decoder_sparse_step`

#### 现象

最早的 rollout vLLM 失败之一是：

```text
AttributeError: 'Qwen3_5MoeTextConfig' object has no attribute 'decoder_sparse_step'
```

位置：

- `/workspace/vllm/vllm/model_executor/models/qwen3_moe.py`

#### 根因

这份 35B config 的字段分布是混合型的：

- root config 上有：
  - `decoder_sparse_step = 1`
- 但 `hf_text_config` 上没有这个字段

而 `qwen3_moe.py` 直接假设 `config.decoder_sparse_step` 一定存在。

#### 修改点

修改：

- `/workspace/vllm/vllm/model_executor/models/qwen3_moe.py`

改成：

```python
decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)
```

#### 为什么这样能解决

因为根据实际读取的 `/home/weight/Qwen3.6-35B-A3B/config.json`：

- `decoder_sparse_step` 的有效值就是 `1`

所以这个 fallback 不是拍脑袋瞎补，而是与当前 checkpoint 语义一致。

#### 原理

这里的原则是：

> **对于 HF config 在 root / text_config 间混合分布的字段，rollout 模型实现不能硬编码“字段一定挂在当前 config 对象上”，必须允许用语义等价的默认值兜底。**

---

### 18. `qwen3_moe.py` 对 `norm_topk_prob` 的假设不成立

#### 现象

下一步 rollout 又报：

```text
AttributeError: 'Qwen3_5MoeTextConfig' object has no attribute 'norm_topk_prob'
```

#### 根因

当前 35B A3B 的 config 中：

- `norm_topk_prob` 在 root 没有
- 在 `text_config` 里也没有

但 `qwen3_moe.py` 直接写：

```python
renormalize=config.norm_topk_prob
```

#### 修改点

改成：

```python
renormalize=getattr(config, "norm_topk_prob", True)
```

#### 为什么这样能解决

vLLM 仓库中其他 MoE 模型（以及 config patch 逻辑）本身就已经广泛使用：

- `getattr(config, "norm_topk_prob", True)`

并且对这个家族的 Qwen MoE 模型来说，默认 `True` 是合理的。

#### 原理

这里的原则是：

> **对于并非强制出现在 config 中、但在模型家族中有清晰默认语义的字段，应优先采用“显式默认值”而不是要求 checkpoint/config 必须完整显式提供。**

---

### 19. `qwen3_moe.py` 中更多 MoE 字段也存在 root / text_config 混合问题

#### 现象

随着逐步推进，发现不仅 `decoder_sparse_step` / `norm_topk_prob` 有问题，`qwen3_moe.py` 对以下字段也有直接访问假设：

- `num_experts`
- `num_experts_per_tok`
- `moe_intermediate_size`
- `intermediate_size`
- `num_hidden_layers`

而这份 35B config 的真实分布是：

- root config：
  - `decoder_sparse_step`
- `text_config`：
  - `num_experts`
  - `num_experts_per_tok`
  - `moe_intermediate_size`
  - `shared_expert_intermediate_size`
- 完全缺失：
  - `norm_topk_prob`

#### 修改点

我继续修改了：

- `/workspace/vllm/vllm/model_executor/models/qwen3_moe.py`

加入更系统的 fallback，包括：

- `top_k=getattr(config, "num_experts_per_tok", getattr(config, "moe_router_topk", 1))`
- `intermediate_size=getattr(config, "moe_intermediate_size", getattr(config, "intermediate_size", 0))`
- `num_hidden_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", None))`
- `num_experts = getattr(config, "num_experts", 0)`
- 稀疏层判断也使用本地 `num_experts` fallback
- dense MLP 分支的 `intermediate_size` 改为允许从 `moe_intermediate_size` fallback

#### 为什么这样能解决

这是把问题从“修一条报错补一个字段”升级为：

- 系统性适配 35B A3B 这份 config 的字段分布方式

#### 原理

这里的原则是：

> **当确认某个模型族的 config 不是“字段完整平铺式”结构时，不能靠逐个 AttributeError 被动修补，而应把所有关键依赖字段一次性梳理成稳定 fallback 体系。**

---

### 20. 35B rollout 再往前推进后，不再死在 config 字段缺失，而是死在 rollout engine 的 KV cache 显存预算

#### 现象

在前面的字段兼容问题修到一定程度后，rollout engine 又推进到了：

```text
ValueError: No available memory for the cache blocks.
Try increasing `gpu_memory_utilization`
```

#### 根因

这说明 rollout engine 已经可以：

- 完成模型结构解析
- 进入模型加载和 KV cache 规划阶段

但在当时的配置里：

- `--vllm-gpu-memory-utilization 0.28`
- `--vllm-max-model-len 18432`
- `35B + TP4 rollout engine`

对于当前 rollout 侧显存预算过于保守，导致可用于 KV cache 的空间为负。

#### 修改点

修改：

- `/workspace/run_qwen36_35b_a3b_async_npu4_15.sh`

把：

```bash
--vllm-gpu-memory-utilization 0.28
```

提高到：

```bash
--vllm-gpu-memory-utilization 0.45
```

#### 为什么这样能解决

rollout 侧当前最大的显存压力来自：

- 35B 模型权重本身
- 18K context 下的 KV cache
- graph / activation 额外开销

提高 rollout 允许使用的显存比例，是最直接的解决方案。

#### 原理

这里的原则是：

> **当 bring-up 已推进到 KV cache 规划阶段时，说明“模型能起”，这时优先应该调 rollout 显存预算与 context 预算，而不是再回头怀疑前面的模型实现。**

---

### 21. Megatron MoE router 仍残留旧的 `slime` 依赖

#### 现象

在 actor 侧继续推进时，又报：

```text
ModuleNotFoundError: No module named 'slime'
```

位置：

- `/workspace/Megatron-LM/megatron/core/transformer/moe/router.py`

#### 根因

当前环境里并没有：

- `slime.utils.routing_replay`

但这条 MoE router 路径里仍然保留了旧的导入：

```python
from slime.utils.routing_replay import register_routing_replay
```

而当前真正存在的是：

- `/workspace/vime/vime/utils/routing_replay.py`

#### 修改点

修改：

- `/workspace/Megatron-LM/megatron/core/transformer/moe/router.py`

改成：

```python
from vime.utils.routing_replay import register_routing_replay
```

#### 为什么这样能解决

当前训练环境的 routing replay 实现实际由 vime 提供，而不是旧的 slime 包。

#### 原理

这里的原则是：

> **当 Megatron 被集成进 vime 作为训练后端时，MoE router 里依赖的 routing replay 也必须指向当前集成栈里的真实实现，而不能保留历史项目的包路径。**

---

### 22. rollout 侧进一步暴露出 checkpoint key / module tree 不匹配问题

#### 现象

在模型路径和字段 fallback 都继续修完之后，rollout engine 继续往前走时又出现：

```text
ValueError: Following weights were not initialized from checkpoint: {...}
```

未初始化权重集中在：

- `input_layernorm.weight`
- `post_attention_layernorm.weight`
- `q_norm.weight`
- `k_norm.weight`
- `model.norm.weight`
- 以及大量 `self_attn.*` / `mlp.*` 参数

#### 根因

这说明 rollout 侧当前选中的模型实现，与 checkpoint 实际 key tree 仍然没有完全对齐。

从 checkpoint 真实 key 看：

- 存在大量 `model.language_model.layers.*...`
- 同时既有：
  - `linear_attn.*`
  - `self_attn.*`
  - `shared_expert.*`
  - `experts.gate_up_proj` / `experts.down_proj`

这更像是一个 **Qwen3.5-MOE / multimodal / GDN 相关实现族** 的权重结构，而不是 upstream `qwen3_moe.py` 默认假设的纯 `Qwen3MoeForCausalLM` 参数树。

#### 已采取的方向性修复

在脚本里已加入：

```bash
--vllm-hf-overrides '{"architectures":["Qwen3_5MoeForConditionalGeneration"]}'
```

目的就是强制 rollout 侧向：

- `Qwen3_5MoeForConditionalGeneration`

这条更接近当前 checkpoint / config 家族的实现靠拢。

#### 为什么这样做是合理的

因为从 registry / config mapping 看：

- `Qwen3_5MoeForConditionalGeneration`
  - 对应 `qwen3_5.py`
  - 绑定 `Qwen3_5MoeConfig`
- 这比继续让 rollout 自动解析成 `Qwen3MoeForCausalLM` 更符合当前模型家族

#### 原理

这里的原则是：

> **如果 checkpoint / config / model_type / architecture 字段都更接近某个特定实现分支，就应显式把 rollout 路径锁到那个分支，而不是继续依赖自动架构推断。**

---

## 当前阶段的总体判断

截至本节更新时，12 卡 35B 异步分卡方案已经经历了三类问题：

1. **NPU-only / MoE 校验问题**
   - 例如 grouped-gemm 的 CUDA-only capability check

2. **MoE rollout 模型 config 兼容问题**
   - `decoder_sparse_step`
   - `norm_topk_prob`
   - `num_experts_per_tok`
   - `moe_intermediate_size`
   - `num_hidden_layers`
   - `num_experts`

3. **更深层的 rollout 模型实现与 checkpoint key tree 不匹配问题**
   - 说明问题已经不是单一字段缺失，而是 rollout 所选模型实现与这份 35B checkpoint 的结构契合度仍不足

同时还叠加：

4. **rollout 侧显存预算问题**
   - 18K context + 35B + TP4 rollout engine 需要更激进的 `gpu_memory_utilization`

5. **历史依赖残留问题**
   - `slime.utils.routing_replay` -> `vime.utils.routing_replay`

---

## 为什么这些修复方式是合理的

这些修复遵循了同一个原则：

> **先确保 bring-up 链路能继续往后走，再根据新的阻塞点收敛到更深层的真实问题。**

换句话说：

- 如果卡在 CUDA-only 检查，就先让 NPU-only 环境能过检查；
- 如果卡在缺字段，就先让 rollout 模型能继续实例化；
- 如果卡在 KV cache 预算，就先让 engine 真正起得来；
- 如果卡在 checkpoint key 对不上，就说明 bring-up 已经推进到了“模型实现选择是否正确”的层面。

这是一个逐层剥离问题的过程，而不是在最开始就试图一次性猜中所有正确参数。

---

## 当前仍未完全解决的问题

截至当前，12 卡 35B 方案还没有像 4 卡原型那样进入稳定 rollout + train 循环。当前剩余问题最可能集中在：

1. **rollout 侧模型实现仍未与 checkpoint 结构完全对齐**
2. **Qwen3.5-MOE / GDN / multimodal 这条 rollout 模型路径仍需进一步适配**
3. **即便模型实现对齐，18K context 下 rollout 显存预算仍需继续打磨**

因此，当前阶段的结论是：

> **12 卡 35B 方案的主阻塞，已经从基础环境问题推进到了“rollout 模型实现与 checkpoint 家族的精确对齐”问题。**

这比最初的状态更接近真正可运行的 35B 异步分卡训练，但还没有完全打通。


---

## 最新新增阻塞：12 卡 actor 侧 HCCL P2P 同平面约束问题

在继续推进 12 卡 35B 异步分卡方案时，最新一次 bring-up 已经不再主要卡在 rollout vLLM 配置字段，而是推进到了 **actor 训练侧 8 卡 HCCL 通信初始化**。

### 现象

训练在 actor 初始化 / optimizer 参数分组阶段失败，核心报错为：

```text
RuntimeError: create_config ... hcclCommInitRootInfoConfig(...), error code is 5
Communication_Error_P2P(EI0010): P2P communication failed.
Reason: Device ID 4 in module 0 and device ID 8 in module 1 are not on the same plane.
Solution: Ensure that the NPU card is normal and entering environment variables 'export HCCL_INTRA_ROCE_ENABLE=1'.
```

对应调用链在：

- `/workspace/vime/vime/backends/megatron_utils/model.py`
- `/workspace/Megatron-LM/megatron/core/optimizer/__init__.py`
- `torch.distributed.all_gather_object(...)`
- `torch_npu` / `HCCLUtils.cpp`

### 根因

这不是 rollout 侧问题，而是 **actor 8 卡训练侧在当前 4-11 这组卡上的 HCCL P2P 平面约束**。

当前 12 卡脚本里：

- `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7,8,9,10,11,12,13,14,15`
- actor 取前 8 张可见卡，也就是物理 `4-11`

而报错明确指出：

- `Device ID 4 in module 0`
- `Device ID 8 in module 1`
- **not on the same plane**

说明当前 actor 8 卡跨越了不同 module / plane，Megatron 初始化优化器参数组时触发的 HCCL object gather / allgather 需要跨这些 plane 建链，但当前环境不满足直连 P2P 约束。

### 为什么这样的问题只在 12 卡 35B actor 上出现

因为：

1. 4 卡原型里 actor 只有 2 卡，根本不会覆盖这么宽的物理卡范围；
2. 35B 12 卡方案的 actor 侧是 8 卡，第一次真正把一个大 MoE actor 放到 `4-11` 这段跨 module 物理拓扑上；
3. 这个错误发生在 actor 的 HCCL 通信初始化，说明当前障碍已经从 rollout bring-up 进一步推进到了 **大规模 actor 并行拓扑与物理连通性** 层面。

### 为什么设置 `HCCL_INTRA_ROCE_ENABLE=1` 可能有帮助

错误信息本身已经给出：

```text
Solution: ... export HCCL_INTRA_ROCE_ENABLE=1
```

这意味着当前驱动/HCCL 栈在处理“不同 module / 不同 plane 间的卡”时，需要打开 intra-node RoCE 通道来承接原本不可用的直接 P2P 路径。

也就是说：

- 默认假设：同 plane / 同 module 间使用更直接的 P2P 拓扑；
- 当前实际：actor 8 卡跨 plane，默认路径不成立；
- 备选路径：启用 `HCCL_INTRA_ROCE_ENABLE=1`，让 HCCL 用 RoCE 风格路径跨 plane 建链。

### 当前结论

截至这个阶段，12 卡 35B 方案的最新主阻塞可以总结为：

> **rollout 模型兼容问题继续在修，但 actor 8 卡训练侧已经先暴露出更底层的 HCCL 物理拓扑限制：当前 4-11 的卡组合跨 plane，默认 P2P 不可用。**

因此后续继续推进 12 卡方案时，优先级上需要同时考虑两类问题：

1. rollout 侧：Qwen3.5-MoE / GDN / checkpoint key tree 适配；
2. actor 侧：8 卡跨 plane HCCL 通信是否需要：
   - 设置 `HCCL_INTRA_ROCE_ENABLE=1`
   - 或调整 actor 选卡方式，尽量落在更一致的物理 plane 上。


---

## 最新新增阻塞：16 卡 actor 侧 torch_dist / dist_checkpointing 加载路径异常

在把方案从 12 卡调整为：

- actor：物理 0-7（8 卡）
- rollout：物理 8-15（8 卡，2 个 4 卡 engine）

之后，新的 bring-up 继续向前推进，但当前最新主阻塞已经从 rollout engine bring-up 转移到了 **actor 8 卡加载 35B torch_dist checkpoint 的分布式 checkpoint 反序列化路径**。

### 现象

当前 16 卡方案下，训练并没有先死在 rollout engine，而是在 actor 初始化、加载 `ref_load` 时失败：

```text
TypeError: object of type '_io.BytesIO' has no len()
```

调用链集中在：

- `/workspace/vime/vime/backends/megatron_utils/model.py:946`
- `/workspace/vime/vime/backends/megatron_utils/checkpoint.py:107`
- `/workspace/Megatron-LM/megatron/training/checkpointing.py:1637`
- `/workspace/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py:977`
- `/workspace/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py:439`

也就是：

- actor 8 卡并行已经开始构建模型和 optimizer
- rollout 侧已有 server 存活
- 但 actor 在 `dist_checkpointing.load(...)` 的 key 替换逻辑 `_replace_sharded_keys_with_state_dict_keys(...)` 中失败

### 根因判断

日志里同时出现了大量：

```text
decoder.layers.* ... from model not in state dict, will skip
```

以及最终：

```text
assert len(tensors) == len(rename_mapping[k])
TypeError: object of type '_io.BytesIO' has no len()
```

这说明当前问题很可能已经不只是“某几个字段名对不上”，而是：

1. **当前 actor 侧 35B Megatron 模型结构 / spec / qwen-gdn-backend 参数，与 `Qwen3.6-35B-A3B_torch_dist` 的 shard 命名和映射预期仍有偏差**；
2. 在 `dist_checkpointing` 做 rename / key replacement 时，本应是 tensor-list 的对象，实际遇到了 `BytesIO` 形式的内容，说明当前 checkpoint 结构和代码路径的配合方式并没有完全对上；
3. 这已经是 **actor 训练侧 checkpoint 读取兼容问题**，而不再是 rollout vLLM 侧 MoE HF 权重兼容问题。

### 为什么 16 卡方案会暴露出这个问题

因为：

- 16 卡方案把 actor 固定到 8 卡（0-7）
- rollout 固定到 8 卡（8-15）
- 这回 actor 不再先被 HCCL plane 问题挡住（至少当前日志里没有再先报那个问题）
- 所以系统第一次更完整地进入了 **8 卡 actor 真正加载 35B torch_dist checkpoint** 这一步

换句话说：

> 16 卡方案让 actor 训练侧的 bring-up 更深入，因此暴露出了新的 checkpoint / dist_checkpointing 兼容问题。

### 为什么当前这个问题和 rollout 侧问题不同

前面一系列 `qwen3_moe.py` / `Qwen3_5MoeForConditionalGeneration` / HF key tree 的问题，都是：

- **rollout vLLM 侧加载 HF checkpoint** 的问题。

而这次新的 `BytesIO has no len()` 问题，发生在：

- **actor 训练侧加载 torch_dist checkpoint** 的问题。

所以当前 16 卡方案的总阻塞已经拆成两层：

1. rollout 侧：Qwen3.5-MOE / GDN / HF 权重映射与模型实现选择问题；
2. actor 侧：35B torch_dist checkpoint 与 Megatron dist_checkpointing / model spec 的匹配问题。

### 当前结论

截至这个阶段，16 卡方案的最新主阻塞可以总结为：

> **rollout 侧仍在推进 Qwen3.5-MOE 路径适配，但 actor 8 卡已经先暴露出更底层的 torch_dist checkpoint 兼容问题。当前训练尚未进入首轮 actor->rollout 权重同步，而是在 actor 初始化加载 checkpoint 时失败。**

这说明 bring-up 已经从：

- 环境依赖
- CUDA-only 校验
- MoE config 字段缺失
- rollout vLLM 模型实现选择
- actor HCCL 物理拓扑

进一步推进到了：

- **actor 侧 35B torch_dist checkpoint 与当前 Megatron/vime 模型构建路径的精确对齐问题。**
