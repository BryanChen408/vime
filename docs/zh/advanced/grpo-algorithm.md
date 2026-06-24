# GRPO 算法：从 rollout 到 opt.step

本文以一个具体场景串起 vime 中 GRPO 的完整更新链路：**一次 rollout 产出 8 题 × 16 样本 = 128 条数据，这些样本是如何被用于一次（或多次）参数更新的**。文末附上与 [verl](https://github.com/volcengine/verl) 的实现对比。

涉及的关键代码位置：

- `vime/ray/rollout.py:_post_process_rewards` —— GRPO 组归一化
- `vime/utils/dp_schedule.py:build_dp_schedule` —— 128 条数据切分到 DP rank / step / microbatch
- `vime/backends/megatron_utils/actor.py:train_actor` —— 训练前的前向（old/ref log_probs）与优势计算调度
- `vime/backends/megatron_utils/loss.py:compute_advantages_and_returns` / `policy_loss_function` —— 优势广播与 PPO 损失
- `vime/backends/megatron_utils/model.py:train` / `train_one_step` —— forward / backward / opt.step
- `vime/utils/ppo_utils.py:compute_policy_loss` / `get_grpo_returns` —— 截断目标与 returns 广播

---

## 1. 数据布局

- `rollout_batch_size = 8`（8 个 prompt/题目）
- `n_samples_per_prompt = 16` → 一次 rollout 产出 `8 × 16 = 128` 条轨迹
- 「组（group）」= 同一个 prompt 的 16 个样本。GRPO 的 baseline 就是在这 16 个样本内部计算的。

## 2. 关键设计：优势在 rollout 阶段就算好

vime 的 GRPO **不是在训练时**计算优势，而是在 rollout 收集数据、做 reward 后处理时就把 reward 变成了组相对优势（`vime/ray/rollout.py:_post_process_rewards`）。训练时只是把这个标量优势广播到 response 的每个 token 上。

## 3. 整体伪代码

```python
# ============ 阶段 1: Rollout（vLLM 推理 + 打分）============
samples = rollout_engine.generate(8 prompts × 16 samples)   # 128 条
raw_rewards = [reward_model(s) for s in samples]            # 128 个标量

# ============ 阶段 2: GRPO 组归一化 → 得到每条样本的标量优势 ============
# vime/ray/rollout.py  _post_process_rewards
r = torch.tensor(raw_rewards).reshape(8, 16)   # [组数, 组内样本数]
r = r - r.mean(dim=-1, keepdim=True)           # 减组内均值（GRPO baseline）
if grpo_std_normalization:                     # grpo/gspo 默认 True
    r = r / (r.std(dim=-1, keepdim=True) + 1e-6)
rewards = r.flatten()                           # 128 个「组相对优势」标量

# ============ 阶段 3: 切分到 DP rank / step / microbatch ============
# vime/utils/dp_schedule.py  build_dp_schedule
# global_batch_size = rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout
# 默认 num_steps_per_rollout=1 → gbs=128 → 整个 128 条做 1 个 optimizer step
num_steps = 128 // global_batch_size           # = 优化步数 = opt.step 次数
# 每个 step 内，该 rank 的样本再被打包成若干 microbatch（梯度累积）

# ============ 阶段 4: 训练前的前向（算 old/ref log_probs + 优势）============
# vime/backends/megatron_utils/actor.py  train_actor
if kl_coef > 0:
    ref_log_probs = forward_only(ref_model, data)      # 给 KL 用
old_log_probs   = forward_only(actor_model, data)      # 重要性比的分母 π_old

compute_advantages_and_returns(args, rollout_data)     # loss.py
#   grpo 分支:
#     returns[i]    = ones_like(kl[i]) * rewards[i]     # 标量优势广播到每个 token
#     advantages[i] = returns[i]
#     若 kl_coef>0: 在 returns 里减 token 级 KL
#   若 normalize_advantages: 再跨 DP 做一次 whiten（减均值/除标准差）

# ============ 阶段 5: 训练循环（forward / backward / opt.step）============
# vime/backends/megatron_utils/model.py  train()  →  train_one_step()
for step_id in range(num_steps):               # 默认只有 1 步
    train_one_step(...)                         # ↓ 见下
```

`train_one_step`（`model.py`）就是 forward / backward / opt.step 三件事：

```python
def train_one_step(data_iterator, model, optimizer, num_microbatches):
    # (1) 清梯度
    model.zero_grad_buffer()
    optimizer.zero_grad()

    # (2) forward + backward: Megatron 的流水线引擎在所有 microbatch 上跑
    #     每个 microbatch: forward → loss_function → loss.backward()（梯度累积进 grad buffer）
    losses_reduced = forward_backward_func(
        forward_step_func = forward_step,   # 内部: logits = model(tokens); 返回 partial(loss_function,...)
        data_iterator     = data_iterator,
        num_microbatches  = num_microbatches,
        forward_only      = False,          # ← False 才会 backward
    )

    # (3) opt.step: 一次参数更新，用的是所有 microbatch（乃至所有 DP rank）累积 + all-reduce 后的梯度
    if valid_step:                          # 检查 grad 没有 nan/inf
        update_successful, grad_norm, _ = optimizer.step()
        opt_param_scheduler.step(increment=global_batch_size)   # 更新 lr

    # (4) 再次清梯度
    model.zero_grad_buffer(); optimizer.zero_grad()
```

其中每个 microbatch 的 `forward_step`（`model.py`）和 loss：

```python
def forward_step(data_iterator, model):
    batch = get_batch(...)                  # 含 tokens / advantages / old log_probs / loss_masks ...
    logits = model(input_ids=batch["tokens"], ...)        # ← 前向
    return logits, partial(loss_function, args, batch, num_microbatches)

# policy_loss_function (loss.py) —— GRPO/PPO 的核心 loss
def policy_loss_function(args, batch, logits, sum_of_sample_mean):
    log_probs = get_log_probs_and_entropy(logits, ...)    # 当前策略 π_θ 的 log_probs（带梯度）
    old_log_probs = batch["log_probs"]                     # π_old（detach，无梯度）
    advantages = cat(batch["advantages"])                  # 阶段2/4 算好的标量优势（已广播到 token）

    ppo_kl = old_log_probs - log_probs
    ratio  = exp(-ppo_kl)                  # = exp(log_probs - old_log_probs) = π_θ / π_old
    # PPO 截断目标（compute_policy_loss, ppo_utils.py）
    pg = max(-ratio * A,  -clip(ratio, 1-ε, 1+ε_high) * A)

    loss = sum_of_sample_mean(pg) - entropy_coef * entropy   # 按 rollout 做 token 加权平均
    if use_kl_loss:  loss += kl_loss_coef * KL(π_θ || π_ref)
    return loss, metrics
```

## 4. 几个容易踩的点

**(1) 一次 rollout 触发几次 `opt.step`？** 取决于 `global_batch_size`（以 rollout 数为单位）：

- `gbs = 128`（默认 `num_steps_per_rollout=1`）→ **1 次 `opt.step`**，梯度是这 128 条的平均。
- `gbs = 32` → `num_steps = 4`，即 **4 次 `opt.step`**，每次用 32 条做一次 mini-batch 更新（PPO 多步、off-policy 的场景，此时 `ratio` 才会明显偏离 1）。

**(2) 128 条之间是「梯度累积」，不是 128 次更新。** 在一个 step 内，128 条被切成多个 microbatch 分散到各 DP rank，forward/backward 逐 microbatch 做、梯度累积，最后 all-reduce 跨 DP 再 **一次** `optimizer.step()`。所以「利用这些样本」= 把 128 条的策略梯度求平均后走一步。

**(3) 单步纯 on-policy 时 `ratio≈1`，但梯度仍然有效。** 默认单步时 `old_log_probs` 是训练前同一个 actor 前向算出来的（detach），`log_probs` 是带梯度重算的，所以 `ratio=1` 处 `∇(-ratio·A) = -A·∇log_probs`，退化成标准的 REINFORCE/策略梯度——这正是 GRPO 想要的：**用组内归一化后的优势 A 给每条样本的 log 概率加权**。

**(4) loss 的缩放**（`loss.py`）：`loss *= num_microbatches / step_global_batch_size * dp_world_size`，目的是抵消 Megatron 内部的梯度累积/缩放，让最终梯度恰好等于「对全局 128 条样本求平均」。`sum_of_sample_mean` 则保证每条 rollout 内部按 token 数加权、rollout 之间等权。

---

## 5. 与 verl 的对比

两边的算法内核是一回事（都是 GRPO：组内减均值、除标准差、广播到 token、PPO 截断目标），区别主要在**工程结构**和**几个默认行为**上。下面同样按 forward/backward/opt.step 这条线来对比。

> verl 参考代码：`verl/trainer/ppo/core_algos.py`（优势/损失）、`verl/trainer/ppo/ray_trainer.py`（driver 编排）、`verl/workers/actor/dp_actor.py`（FSDP 后端训练循环）。

### 5.1 GRPO 归一化：在哪算、怎么算

| | vime | verl |
|---|---|---|
| 计算位置 | **Rollout 侧**（`ray/rollout.py:_post_process_rewards`），打完分立刻算 | **Driver 侧**（`ray_trainer.compute_advantage` → `core_algos.compute_grpo_outcome_advantage`），train 前算 |
| 分组方式 | `reshape(-1, n_samples_per_prompt)`，**靠样本排列顺序**分组 | `index/uid` 显式分组（`defaultdict(list)`），不依赖顺序 |
| 单样本组 | 由 `n_samples_per_prompt==1` 时关掉 std | 显式判断 `len==1` → mean=0/std=1 |

数学上完全一致：`A = (r - mean_group) / (std_group + eps)`，然后广播到 response 每个 token。verl 另有向量化实现 `grpo_vectorized` 和 Dr.GRPO（`norm_adv_by_std_in_grpo=False` 只减均值不除 std）——vime 对应的是 `grpo_std_normalization=False`，二者等价。

**实质差异**：vime 把「归一化」和「广播」拆成两步（rollout 算标量、`get_grpo_returns` 在 loss.py 广播）；verl 在 `compute_grpo_outcome_advantage` 里一次做完。功能等价。

### 5.2 数据切分：这是最大的区别

128 条数据怎么变成几次 `opt.step`，两边的旋钮不一样：

**vime**（`dp_schedule.build_dp_schedule`）：

```text
global_batch_size = rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout
num_steps = num_rollouts // global_batch_size      # = opt.step 次数
```

- 用 `num_steps_per_rollout` 把 128 条**互斥地**切成若干 step，每个 step 一次 `opt.step`。
- **同一条样本在一次 rollout 内只被用一次**，没有「ppo_epochs」这种数据复用。

**verl**（`dp_actor.update_policy`）：

```python
mini_batches = data.split(ppo_mini_batch_size)     # 128 → 切 mini-batch
for _ in range(ppo_epochs):                        # ← 同一批数据重复用 ppo_epochs 遍
    for mini_batch in mini_batches:
        micro_batches = mini_batch.split(ppo_micro_batch_size_per_gpu)
        optimizer.zero_grad()
        for mb in micro_batches:  ...backward()     # 梯度累积
        optimizer_step()                            # 每个 mini-batch 一次 opt.step
```

- verl 用 `ppo_mini_batch_size` 控制每次 `opt.step` 的样本数，`ppo_epochs` 控制**整批数据重复训练几遍**。
- `ppo_epochs > 1` 时，**同一条样本会被多次用于梯度更新**（真正的 off-policy PPO，这时 `ratio` 才显著偏离 1）。

> 对照：vime `num_steps_per_rollout=4` ≈ verl `ppo_mini_batch_size=32, ppo_epochs=1`。但 **verl 能用 `ppo_epochs>1` 复用数据，vime 在单次 rollout 内做不到**（vime 走的是多步 off-policy 的另一套：`keep_old_actor` / `update_weights_interval`，而不是 epoch 复用）。

### 5.3 on-policy 判定与 old_log_probs

两边逻辑几乎相同：

- **verl**：`on_policy = len(mini_batches)==1 and ppo_epochs==1`；若 on-policy，`old_log_prob = log_prob.detach()`，否则用预先 forward 存的 `old_log_probs`。
- **vime**：`can_reuse_log_probs_in_loss`（`actor.py`）判断同样的条件（单 step、kl_coef=0、不 keep_old_actor…），满足就跳过单独的 old_log_prob 前向，在 loss 里 `detach`。

结论一致：单步纯 on-policy 时 `ratio≡1`，退化成优势加权的 REINFORCE。

### 5.4 forward / backward / opt.step 机制

| | vime | verl（FSDP 后端） |
|---|---|---|
| 梯度累积 | Megatron `forward_backward_func` 跨 microbatch 自动累积 | Python 手写 `for mb: loss.backward()` 累积 |
| 跨 DP 同步 | DDP / 分布式 optimizer + `all_reduce` | FSDP reduce-scatter |
| opt.step | `optimizer.step()`，带 grad nan/inf 检查后 `opt_param_scheduler.step` | `_optimizer_step()`，含 grad clip + scaler |
| 后端 | **只有 Megatron**（PP/CP/TP/VPP 全套） | **FSDP 和 Megatron 双后端**（`dp_actor.py` vs `megatron_actor.py`） |

vime 因为绑定 Megatron，流水线/上下文并行的 microbatch 调度交给 Megatron 引擎；verl 的 FSDP 路径是纯手写循环，更直白也更容易读。verl 的 Megatron 后端（`megatron_actor.py`）则与 vime 思路更接近。

### 5.5 loss 聚合默认值（容易被忽略，但影响结果）

这是两者**默认行为真正不同**的地方：

- **verl** 默认 `loss_agg_mode = "token-mean"`（`agg_loss`，`core_algos.py`）：把整个 (mini-)batch 的所有有效 token 放一起求平均 → **长序列权重更大**。另有 `seq-mean-token-sum-norm`（Dr.GRPO 论文做法，用固定常数归一）。
- **vime** 默认 `calculate_per_token_loss=False` → 用 `sum_of_sample_mean`：**每条 rollout 内部按 token 加权平均，rollout 之间等权** → 长短序列等权。打开 `calculate_per_token_loss` 才接近 verl 的 token-mean。

同样一批 128 条数据，默认配置下两边的 loss 归一化分母不一样，梯度尺度会有差异。要严格对齐结果，需要手动把 `loss_agg_mode` / `calculate_per_token_loss` 调成一致。

### 5.6 小结

算法内核（组归一化 + PPO 截断）完全相同；区别在三处工程取舍：

1. **数据复用**：verl 有 `ppo_epochs` 可在一批 rollout 上多遍更新，vime 单次 rollout 内只用一遍；
2. **loss 归一化默认值**：verl 默认 token-mean（长序列加权），vime 默认 per-rollout 等权；
3. **后端与实现风格**：vime 只走 Megatron、靠其引擎做梯度累积；verl 双后端、FSDP 路径手写 backward 循环，归一化放 driver 侧、分组靠显式 uid。

---

## 6. NPU（Ascend）训练基础设施

本节覆盖 vime 对华为 Ascend NPU 的全栈适配（commit `8e2bc806` 起，持续到 `8d6c1841`），涉及三个外部依赖：

| 依赖 | 作用 | 路径 |
|---|---|---|
| **vllm-ascend** | vLLM 的 Ascend NPU 后端（HCCL 通信、CANN 算子） | `/home/g00841271/vllm-ascend` |
| **MindSpeed** | 华为昇腾训练加速库（算子融合、NPU 适配器、GDN 算子） | `/home/g00841271/MindSpeed` |
| **Megatron-LM** | 训练引擎（PP/DP/TP/CP 并行） | `/home/g00841271/Megatron-LM` |

### 6.1 启动时的 NPU 适配

`vime/backends/megatron_utils/__init__.py` 中的 `_ensure_npu_adaptor()` 是 NPU 路径的入口：

```python
def _ensure_npu_adaptor():
    """延迟导入 mindspeed.megatron_adaptor 来 patch torch NPU。
    必须在任何触及 NPU 的 megatron 代码之前调用（模型构建、checkpoint 加载等）。
    设计为延迟加载（非模块级），使得 vLLM 子进程不会拉入 mindspeed——
    mindspeed 会破坏 torch.compile 的 aot_compile 路径从而杀死 cudagraph capture。"""
    if is_npu():
        import mindspeed.megatron_adaptor
```

关键设计决策：**导入被故意延迟** — 因为 `update_weight_from_tensor.py`（colocate IPC 权重同步路径）可能被 vLLM 子进程 import，而 vLLM 子进程需要 cudagraph capture。mindspeed 的 `megatron_adaptor` 会 monkey-patch `torch.compile` 破坏 `aot_compile`，因此仅在 NPU 训练的 megatron 路径才真正 import 它。

在 `actor.py:12-14`，如果 detect 到 NPU，还会额外调用 `mindspeed.megatron_adaptor.repatch(args)` 做二次 patch。

### 6.2 vLLM 子进程环境（vllm-ascend 接口）

`vime/backends/vllm_utils/vllm_engine.py:build_vllm_subprocess_env` 为 vllm-ascend 构建子进程环境：

- **清理 `PYTORCH_NPU_ALLOC_CONF`**：Docker 镜像设置了 `expandable_segments:True` 供训练用，但 vllm-ascend 的 CaMemAllocator 在 sleep 模式下会断言拒绝该参数。让 vllm-ascend 自己的 `platform.py`（sleep-mode-aware）按需重新设置。
- **`ASCEND_RT_VISIBLE_DEVICES`**：传递 NPU 可见设备，对应 CUDA 的 `CUDA_VISIBLE_DEVICES`。
- **colocate 模式**：将 vime 根目录注入 `PYTHONPATH`，使 vLLM 子进程能 import `vLLMColocateWorkerExtension`。

### 6.3 权重同步：两条路径

vime 支持两种权重同步模式，通过 `--colocate` 自动选择：

**路径 A：colocate（Tensor IPC）** — `update_weight_from_tensor.py`

Actor 训练进程与 vLLM 引擎共享同一 NPU 池。权重更新流程：

```
Megatron params → convert_to_hf() → HF chunks → CUDA/NPU IPC handle
   → Ray ObjectRef → vLLMColocateWorkerExtension.update_weights_chunk()
   → layerwise_reload (initialize/finalize) → 权重加载完成
```

关键细节：
- `_copy_vllm_param_attrs(src, dst)`（`update_weight_from_tensor.py:410`）：`torch.nn.Parameter(data)` 创建新 tensor 时会丢弃 vLLM 自定义属性（如 `weight_loader`），必须在 weight sync 后从源 param 拷贝回来。原先此逻辑在 vllm-ascend 的 `worker.wake_up` 中，现移到 `finish_weight_update` 之后执行。
- `vLLMColocateWorkerExtension.init_weight_transfer_engine` 是 no-op — colocate IPC 路径不需要 transfer engine（weights 直接通过 Ray ObjectRef 传递）。
- **layerwise reload**：`start_weight_update` 中调用 `initialize_layerwise_reload(model)` 解除 vLLM 加载后做的 kernel-format fusion，使 params 重新可以被 `load_weights` 写入。这对 NPU sleep 模式特别重要：sleep 后 params 数据可能在 host 侧，reload 保证 d2d copy 有效。

**路径 B：disaggregated（Distributed NCCL/HCCL）** — `update_weight_from_distributed.py`

```
import guard: try import vllm_ascend HCCL engine, fallback NCCL
   → start_weight_update → post process weights → trainer_send_weights via HCCL/NCCL
```

- NPU 上用 `HCCLWeightTransferEngine`（vllm-ascend 提供），GPU 上用 `NCCLWeightTransferEngine`。
- 顶层 import 被 try/except 保护：vllm_ascend 的 HCCL engine 在某些 NPU 构建中不存在，但 colocate path 不需要它——guard 防止模块级崩溃。

### 6.4 Sleep 模式（vllm-ascend 交互）

`vllm_engine.py:release_memory_occupation` 和 `resume_memory_occupation`：

```python
def release_memory_occupation(self, level: int = 1):
    """默认 level=1（weights 卸载到 host，保留 param 对象）而非 level=2。
    NPU（vllm-ascend）上 level=2 会丢弃 weights，wake_up 后重建为普通的
    torch.Tensor 而不带 vllm 的 weight_loader 属性，打破 RLHF 权重更新。"""
    self.flush_cache()
    requests.post(f"{base}/sleep", params={"level": level})

def resume_memory_occupation(self, tags: list[str] | None = None):
    """POST /wake_up，可选 tags 筛选唤醒哪些组件。"""
    requests.post(f"{base}/wake_up", params=wake_params)
```

核心约束：在 vllm-ascend 上必须用 `level=1`。`level=2` 的完全卸载会破坏 `weight_loader` 属性，导致后续 RLHF weight update 时 `'Parameter' object has no attribute 'weight_loader'`。

### 6.5 外部依赖概况

三个外部仓库的状态（截至 2026-06-23）：

**vllm-ascend**（`/home/g00841271/vllm-ascend`）：
- 最近 commit `794e1539`（branch clean），追溯到 v0.13.0 的 cherry-pick
- vime 通过 `vllm_ascend` namespace 使用其 HCCL engine 和 NPU worker

**MindSpeed**（`/home/g00841271/MindSpeed`）：
- 最近 commit `d04da291`（CANN 环境变量文档）
- 包含关键 fix：GDN `dqkwg` 和 `cumsum` bug 修复（`6c931c1d`）、triton op `mbs>1` 修复（`b600ab5a`）
- vime 通过 `mindspeed.megatron_adaptor` 和 `mindspeed.ops.chunk_gated_delta_rule`/`causal_conv1d` 使用

**Megatron-LM**（`/home/g00841271/Megatron-LM`）：
- 最近 commit `a8aa264`（branch clean），基于 `core_r0.12.0`
- 提供 PP/DP/TP/CP 分布式训练引擎

---

## 7. Qwen3.6-35B-A3B GDN（Route B）移植

`8d6c1841` 的核心交付：将 Qwen3.6-35B-A3B 的 GDN（Gated Delta Net）架构从 Qwen 官方 longctx 分支移植到 vime 的 slime-mindspeed 栈。

### 7.1 GDN 后端调度

`vime_plugins/models/qwen_gdn_backend.py` 统一管理三种 GDN 后端：

| 后端 | 使用场景 | 算子来源 |
|---|---|---|
| `"fla"` | GPU（NVIDIA CUDA SM90） | `flash-linear-attention` 库 |
| `"flashqla"` | GPU，需要 PyTorch 2.8+ / CUDA 12.8+ / SM90 | `FlashQLA` 官方库 |
| `"npu"` | NPU（Ascend） | `mindspeed.ops.chunk_gated_delta_rule`（MindSpeed AscendC 混合算子）+ `mindspeed.ops.causal_conv1d`（Triton conv port） |

`get_chunk_gated_delta_rule(backend)` 和 `get_causal_conv1d(backend)` 是延迟分发的入口，每个后端在调用时才完成 import 和版本校验。

NPU 路径的 causal conv1d 可通过 `QWEN36_CAUSAL_CONV1D_IMPL=eager` 回退到纯 PyTorch 的 `F.silu(F.conv1d)` 实现。

### 7.2 核心模型文件

`vime_plugins/models/qwen3_5.py`（+488/-67 行）：fused GDN forward + TP/CP 分发逻辑。包含 GDN 的 forward 路径，处理 ColumnParallelLinear 的 head-split、sharded state dict 等。

`vime_plugins/models/gdn_cp_utils.py`（+257 行）：GDN 在 context-parallel 下的 packed × CP THD 重排。从新 Megatron 栈的 `megatron.core.ssm.gated_delta_net` 移植到老栈的 `megatron.core.ssm.mamba_context_parallel` 原语上。包括：
- `thd_get_partitioned_indices_torch`：纯 torch 实现 THD 分区索引（替代新栈的 `tex.thd_get_partitioned_indices`）
- `tensor_a2a_cp2hp` / `tensor_a2a_hp2cp`：CP→HP / HP→CP all-to-all
- `get_parameter_local_cp`：CP 参数分片
- `pad_packed_for_cp` / `unpad_packed_for_cp`：packed 序列的 pad/unpad

`vime_plugins/mbridge/gdn_param_mapping.py`（+281 行）：6 个纯 torch GDN helper（`adjust_qweight_k_h` 等），作为 mbridge 参数映射的 single source of truth。`mbridge/qwen3_5.py` 从该文件 import，不再包含内联副本。

### 7.3 megatron_to_hf 转换器

`vime/backends/megatron_utils/megatron_to_hf/qwen3_5.py`（+50 行）：处理 fused `in_proj`（ColumnParallelLinear）的拆分和 deinterleave conv1d，在 Megatron 分布式 checkpoint → HF safetensors 转换时正确还原 Qwen3.6 的权重布局。

### 7.4 DAPO 运行配置

`scripts/run_qwen36_35b_a3b_dapo_math_npu.sh` — 单机 16 NPU 完整 DAPO 数学 RL 训练：

```
模型: Qwen3.6-35B-A3B GDN/MoE (40层, 256专家, topk=8, GQA 16h/2q)
并行: TP=2, EP=8, CP=1, PP=1
权重同步: colocate (same NPU pool), sleep mode ON
Rollout: vllm-gpu-memory-utilization 0.30 (训练+推理共享 16 NPU)
算法: GRPO advantage + decoupled clip (eps 0.2 / 0.28), kl_coef=0
数据: DAPO-MATH-17K, deepscaler reward (本地 \boxed{} 检测)
优化器: Adam (lr=1e-6), precision-aware + CPU offload
内存: dynamic batch size, max_tokens_per_gpu=4096, recompute full/uniform
```

关键参数含义：
- `--eps-clip 0.2 --eps-clip-high 0.28`：DAPO（Decoupled Alignment Policy Optimization）的 decoupled clip — 下限 clip 0.2、上限 clip 0.28，相比标准 PPO 的对称 clip 能更精细地控制策略更新幅度。
- `--n-samples-per-prompt 4`：每个 prompt 采样 4 条（非标准 16），与 `--rollout-batch-size 4` 搭配，总共 `4×4=16` 条/step 的 mini-batch。
- `--global-batch-size 16`：每 step 16 个 rollout。
- `--qwen-gdn-backend npu`：使用 MindSpeed 的 AscendC GDN 算子。

### 7.5 未提交改动（工作区）

当前工作区有以下与训练稳定性相关的未提交改动：

**(1) TIS 调试 hook**（`vime/backends/megatron_utils/loss.py`，+11 行）：
```python
# Save per-token logprobs for train-inference consistency analysis
if os.environ.get("VIME_SAVE_TIS_LOGPROBS", ""):
    torch.save({"old_log_probs": old_log_probs.detach().cpu(),
                "rollout_log_probs": rollout_log_probs.detach().cpu()}, save_path)
```
在 `policy_loss_function` 中保存 old/rollout log_probs 到磁盘，用于离线分析训练-推理一致性（TIS = Truncated Importance Sampling 的 importance ratio 漂移）。通过环境变量 `VIME_SAVE_TIS_LOGPROBS=<path>` 触发，非侵入式调试。

**(2) Lazy NPU adaptor**（`vime/backends/megatron_utils/__init__.py`，+14/-2 行）：
将 `import mindspeed.megatron_adaptor` 从模块级下移为 `_ensure_npu_adaptor()` 函数，防止 vLLM 子进程在 colocate 模式下 import 该模块破坏 cudagraph capture。已在 6.1 节详述。

**(3) DAPO 运行脚本微调**（`scripts/run_qwen36_35b_a3b_dapo_math_npu.sh`）和模型配置（`scripts/models/qwen3.5-35B-A3B.sh`）。

**(4) 文档变更**：
- `docs/zh/index.rst`：新增 `advanced/grpo-algorithm.md` 到 toctree（未提交）。
- `docs/zh/advanced/grpo-algorithm.md`：本文档（untracked，未入 git）。

---

## 8. 变更与文档的对应关系总结

| 提交/变更 | 主题 | 本文档对应章节 |
|---|---|---|
| `8d6c1841` | GDN port + sleep mode | §6.2–§6.4, §7.1–§7.3 |
| `00ccc210` | 30B NPU script + mindspeed guard | §6.1 |
| `d2b5f7c6` | NPU setup guide + 4B script | — (不涉及算法) |
| `b1a57185` | Ascend A3 Dockerfile | — (不涉及算法) |
| `c304f5bc` | vllm_engine bugfix | §6.4 (sleep level 选择) |
| `8e2bc806` | NPU 基础支持 | §6.1, §6.2 |
| 未提交 (`__init__.py`) | lazy mindspeed import | §6.1 |
| 未提交 (`loss.py`) | TIS debug hook | §7.5 |
| 未提交 (`docs/`) | GRPO 算法文档 | §1–§5 |
