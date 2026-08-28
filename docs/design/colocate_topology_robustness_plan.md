# 共卡拓扑鲁棒性方案：同构 / 异构 × 单机 / 多机

> 状态：待评审
> 日期：2026-08-28
> 基座：`a3-pd`
> 起因：单机异构（T4）首跑崩在 `NPUIPCWeightTransferInitInfo.__init__() got an unexpected keyword argument 'master_address'`

---

## 1. 诊断：一个语义，三个各自为政的代理量

「这台引擎是否与 trainer 共卡」这一个事实，代码里有 **三处独立判定，判据互不相同**：

| # | 消费者 | 位置 | 判据 |
|---|---|---|---|
| 1 | 引擎启动的 weight-transfer backend（`npu_ipc` / `nccl`） | `vllm_engine.py:566-569` | **节点**：`host_for_subprocess in actor_nodes` |
| 2 | 权重同步走 IPC 还是 HCCL | `update_weight_from_tensor.py` `count_colocated_engines` | **总 actor 卡数**：`gpu_offset + count > total_actor_gpus` |
| 3 | 训练窗口哪些引擎 sleep | `rollout.py` `ServerGroup._offload_engine_indices` | **share 卡数**：`gpu_offset + i*per < rollout_shared_num_gpus` |

另有两处次级判定沿用同族代理量：`ServerGroup.needs_offload`（组级，比 `megatron_num_gpus`）、`actor.py` 的 `_rollout_shares_actor_devices`（`colocate or spec.rollout_has_share`）。

### 1.1 实测矩阵（用真实代码跑出，非推演）

`C` = 判为共卡，`.` = 判为专用/远程：

| 拓扑 | 引擎侧 | 更新器 | offload | |
|---|---|---|---|---|
| T2 同构单机（actor=share=16） | `CCCCCCCC` | `CCCCCCCC` | `CCCCCCCC` | ✅ |
| T3 异构跨机（.56 share16 + .64 dedi8） | `CCCCCCCC....` | `CCCCCCCC....` | `CCCCCCCC....` | ✅ |
| **T4 异构单机**（actor 4-11, share 4-11, dedi 12-15） | `CCCCCC` | `CCCC..` | `CCCC..` | ❌ |
| **T5 部分共卡跨机**（actor 16, share 8, dedi 8@n2） | `CCCC....` | `CCCCCCCC` | `CCCC....` | ❌ |

**只有跑过的两个拓扑是对的。** 两个没跑过的各自在不同的消费者上错：

- **T4**：引擎侧按节点判 —— 专用引擎与 actor 同节点 → 误判共卡 → 起成 `npu_ipc`；而更新器按槽位正确判为远程 → 发带 `master_address` 的 HCCL init info → IPC 的 init info 不认该 kwarg → 500。
- **T5**：更新器按 `total_actor_gpus=16` 判 —— 远程引擎槽位仍 < 16 → 误判共卡 → 会对 n2 上的引擎尝试 IPC 直传。

修 T4 不会顺带修 T5：它们坏在不同的消费者上。

### 1.2 根因

三个代理量都不是「共卡」的定义，只是在特定拓扑下与它等价：

- **节点**：假设「专用段一定在别的节点」 → 单机异构证伪
- **总 actor 卡数**：假设「share 段覆盖 actor 全部卡」 → 部分共卡证伪
- **share 卡数**：最接近，但只在 layout 路径下有定义（`--colocate` 与位置式路径下 `spec is None`）

共卡的**定义**是：该引擎占用的全部 `(node, device)` 都落在 actor 的 `(node, device)` 集合内。这是一个集合包含关系，任何一维标量都无法无损表达。

---

## 2. 方案

### P0 — 单一真源：`resolve_engine_roles`

新增 `vime/ray/engine_roles.py`：

```python
@dataclass(frozen=True)
class EngineRole:
    index: int                      # 全局引擎序号
    gpu_slot: int                   # 在 rollout 卡序列中的起始槽位
    placement: tuple[tuple[str, int], ...]   # 该引擎占用的 (node, device)
    colocated: bool                 # 唯一判据

def resolve_engine_roles(args) -> tuple[EngineRole, ...]:
    """唯一判据:引擎占用的 (node, device) 是否全部 ⊆ actor 的 (node, device)。"""
```

必须同时支持三条现存路径，不能只覆盖 layout：

| 路径 | actor 卡集合来源 |
|---|---|
| `--resource-layout` | `spec.actor` 展开 |
| `--colocate` | actor 全部卡；rollout 与之完全重合 → 全 colocated |
| 位置式（异步 PD） | actor 占 `[0, actor_gpus)`，rollout 从 `actor_gpus` 起 → 全非 colocated |

该判据在四个拓扑上均正确（T2 全 C；T3/T4/T5 按实际包含关系分叉），无需特例。

### P1 — 消费者改读真源

| # | 改动点 | 现判据 → 新 |
|---|---|---|
| 1 | `vllm_engine.py` 启动 backend | 节点 → `role.colocated` |
| 2 | `count_colocated_engines` | 前缀计数 → 读 roles（保留函数名与前缀语义，内部改为按 role 计数并**断言前缀连续**） |
| 3 | `_offload_engine_indices` | `shared_num_gpus` → `role.colocated` |
| 4 | `ServerGroup.needs_offload` | `megatron_num_gpus` → 组内是否存在 colocated role |
| 5 | `actor._rollout_shares_actor_devices` | `colocate or has_share` → `any(r.colocated)` |

**引擎侧拿不到 role 的问题**：`VLLMEngine.__init__` 目前只收 `rank / base_gpu_id / num_gpus_per_engine`，没有槽位。不要在引擎侧用 `rank * per_engine` 反推 —— 多 group（PD 的 prefill/decode 分组）下 `rank_offset` 与 `gpu_offset` 不同步，反推会错。**显式把 `EngineRole` 传进构造函数**。

### P2 — CPU 回归矩阵（无需 NPU）

`tests/test_engine_role_consistency.py`：

- 拓扑：T1（纯分离）× T2 × T3 × T4 × T5 × `--colocate` × 位置式
- 每引擎卡数：`per_engine ∈ {1, 2, 4}`
- 断言 1：**五个消费者对每台引擎的判定完全一致**（这次 T4/T5 就是回归用例）
- 断言 2：colocated 引擎构成**前缀**（`count_colocated_engines` 的前缀语义、以及 layout 校验「共卡段在前」共同依赖它）
- 断言 3：HCCS 域 —— 单台引擎占用的卡不得跨 `0-7 / 8-15`（当前只靠人工把 share 段拆两条，无校验；`per_engine=4` 时写成一条 `"4-11"` 就会产出 `(6,7,8,9)` 跨域）

### P3 — 分阶段真机，按风险递增

| 阶段 | 拓扑 | 新变量 | 前置 |
|---|---|---|---|
| S1 | T2 同构单机 16 卡 | 同步 rollout 路径本身 | 无（绕开本 bug） |
| S2 | T4 异构单机 | dedicated 段 + HCCL 通道 + CP4 | P0/P1/P2 |
| S3 | T3 异构跨机 | 跨机 HCCL（已验证过，回归） | S2 |
| S4 | T5 部分共卡 | share < actor | S3 |

S1 不依赖任何代码改动，**现在就能跑**。

---

## 3. 立刻可用的绕行

T2（同构）三处判据天然一致，不碰这个 bug，且并行度 `TP2×CP8=16` 是 `.56` 已验证的配置：

```
actor:   0-15
rollout: 4-7 (share) + 8-15 (share)     → 6 台 TP2 引擎,全部共卡
polar:   0-3(与 actor 重叠,layout 中不声明)
```

polar 与训练共卡是可行的，且**由同步路径保证**：`generate()` 返回时所有 polar task 已终态（含 judge），而 offload/训练发生在其后，judge 与训练天然不重叠。卡位必须让 polar 落在「有训练常驻、无引擎」的 0-3 上 —— 4-15 被引擎占到 util 0.70 后没有余量给 judge。

⚠ 代价：`polar_reserved` 与 actor 重叠会被 `_validate_layout` 拒绝，只能省掉该角色 → **那 4 张卡失去冲突校验**，profile 里 pool 写错不会报错。

---

## 4. 不做什么

- 不在引擎侧用 `rank * per_engine` 反推槽位（PD 多 group 下会错）
- 不给 T4 打点对点补丁（不修 T5，且第四个消费者迟早再分叉一次）
- 不动异步 PD 路径的既有行为（P1 的每处改动都必须在 T1/T3 上逐位等价，由 P2 断言）
