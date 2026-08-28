# a3-pd 共卡基座 · polar-bridge 能力迁移方案

> 状态：待开发
> 日期：2026-08-28
> 基座分支：`a3-pd`（HEAD `b892640d`）
> 来源分支：`feature/polar-bridge`（HEAD `b71f3f62`）
> 范围：仅 `vime` 仓。不涉及 `polar`、`vllm-023`、`vllm-ascend-023`、`Megatron-LM`、`MindSpeed`
> 相关：`vime_polar_sync_colocate_design.md`（工作区顶层，同步 rollout 路径的设计依据）

---

## 0. 决策与依据

**基座选 a3-pd。** 两个分支解决的是正交的两个问题：

| | A. 资源 / 显存 / 权重通道 | B. rollout 调度语义 |
|---|---|---|
| a3-pd | ✅ 已解决，含异构 | ❌ |
| polar-bridge | ❌ 语义上禁止异构 | ✅ |

polar-bridge 的 `resource_layout` 对四个 role 做严格互斥重叠校验，**无法表达** rollout ∩ actor ≠ ∅ 的异构共卡；a3-pd 用 `share="actor"` 打开了这个例外（`vime/ray/resource_layout.py:335-336`）。在 polar-bridge 上重做异构共卡等于重走 a3-pd 的 149 个 commit。

### 0.1 经复核的能力矩阵

| 能力 | a3-pd | polar-bridge |
|---|:--:|:--:|
| 异构共卡（rollout ∩ actor ≠ ∅） | ✅ `share="actor"` | ❌ 严格互斥校验 |
| 同构共卡 | ✅ `--colocate`；全 share layout 亦可 | ✅ `--colocate` |
| storage-resize offloader | ✅ 含 `param_restorer`，不多占 host 副本 | ⚠️ 无，无条件多分配一份 model-sized pinned CPU |
| TMS 后端（NPU） | ❌ 硬编码封死 | ✅ 参数 + region |
| 显存交接编排 | ✅ `memory_handoff` 21 处引用 | ❌ 0 处 |
| `expandable_segments` 动态开关 / `aggressive_empty_cache` / `cpu_tensor_breakdown` | ✅ | ❌ |
| 交接点内存探针（driver → actor） | ❌ | ✅ |
| 同步一次性 rollout 路径 | ❌ | ✅ |
| KV 容量感知路由 | ✅ `scripts/dp_load_balance_proxy_server.py` | ❌ 文件不存在 |
| NPU 真机雷修复 + CPU 回归（带 run id） | ✅ | ❌ |

**结论**：polar-bridge 独有的仅三样——同步 rollout 路径、交接点探针、TMS 后端选择权。全部可移植，无一是基座属性。

### 0.2 待解问题（本方案不承诺解决）

rollout 窗口下训练侧常驻 ~9.8G/卡，压住共卡引擎 `gpu_memory_utilization` 到 0.70，256K 长序列 KV 不足。

**该数字从未被逐项拆解过**，仓内无任何账本。Phase A 的目的就是产出这个账本。在账本出来之前，不对"能否把 0.70 推到 0.85"做任何承诺。

若拆账结果显示残留主要是驱动侧（CANN 图缓存 / HCCL context），则 TMS 亦无解——那时的正解是接受共卡段 0.70，靠既有的 KV 容量路由把长会话压到 `.64` 专用段（util 0.85 / KV 16.5G）。异构布局本就是为此设计。

---

## 1. 工作区前置状态

```
HEAD                b892640d fix(rollout): 按 KV 容量维护 session 缓存亲和
未提交改动          scripts/start_sync_hybrid.sh  VLLM_GPU_MEM_UTIL 0.70 → 0.80
未跟踪              vime_64_memory_offload_{incremental,rootfix}_*.patch（已应用的存档）
```

⚠️ 那处未提交改动与**同文件 18-23 行的显存账自相矛盾**（0.80 → 引擎 48.8G + trainer 9.8G + 杂项 8G = 66.6G > 61G 总量），且未经真机验证。

**处置：保留在工作区，不纳入本方案任何 commit。** 每个 commit 显式指定路径，不用 `git add -A`。

---

## 2. Commit 结构

### Phase A — 拆账：把探针打到交接点

a3-pd 已有的（**不重复造**）：

- `VIME_MEM_PROBE` 经 `runtime_env` 转发进 train actor —— `vime/ray/actor_group.py:65`
- `_log_npu_mem(tag, step_id)` —— 区分 torch 池内 / 池外占用（`model.py:42`）
- `_log_npu_expandable(tag, step_id)` —— expandable 是否真激活、覆盖率、`fully_free` 可归还量（`model.py:72`）

**缺的是 driver 侧**：`train.py` 的循环跑在无设备的 driver 上，交接点的显存必须经 Ray 往返到 actor 内才读得到。a3-pd 现有探针只覆盖 train step 内部（`post-fwd-mb` / `pre-fwd-bwd` / `post-fwd-bwd` / `pre-empty`）与 `post-init`，**交接点是盲区**。

> 相对 polar-bridge 的改进：它的 `probe_memory` 只调 `print_memory`。本方案改调 a3-pd 已有的 `_log_npu_mem` + `_log_npu_expandable`，直接给出「torch 没还」vs「驱动侧不放」的判据——这正是 §0.2 要的账本。

| # | Commit | 内容 | 状态 |
|---|---|---|---|
| A0 | `refactor(mem): NPU 显存探针搬出 megatron 模块` | `_log_npu_mem` / `_log_npu_expandable` 移入 `memory_utils.py`；函数体逐字未改，`model.py` 重新导出，5 处旧调用点与 `actor.py:135` 局部导入行为不变 | ✅ `606dcf6b` |
| A1 | `feat(mem): 让 train driver 能读共卡交接点的显存` | `mem_probe_enabled()`；`TrainRayActor.probe_memory(tag)` 同时报池内/池外拆分、可归还量、host 侧；`RayTrainGroup.probe_memory(tag)` 带门控扇出（关闭时零 Ray 往返） | ✅ `ada7a669` |
| A2 | `feat(train): 在每个共卡交接点埋显存探针` | `train.py` 七点埋点；`onload_weights` / `onload_kv` 埋进 handoff 辅助函数内部，启动期那次交接一并覆盖（两函数各多一个 `tag` 参数） | ✅ `84067f27` |
| A3 | `test(mem): 覆盖共卡交接点探针的门控与覆盖面` | 14 项新测试 + 修正 `test_npu_memory_handoff.py` 因 `tag` 参数而失效的 2 处调用 | ✅ `def45dcb` |

A0 → A1 → A2 → A3 顺序依赖。

**拆账时要看的两个数**（`VIME_MEM_PROBE=1`）：

- `handoff:rollout N after train offload` —— rollout 窗口的**起始残留**。它直接决定共卡引擎 util 的天花板。
- 该行的 `non_torch = device_used - torch_reserved`：若 `non_torch` 占大头 ⇒ 残留在驱动侧（CANN 图缓存 / HCCL context），**TMS 与 storage-resize 都够不到**，Phase C 无意义；若 `torch_reserved` 占大头且 `[MEM-EXP]` 的 `cached_free` / `fully_free_seg` 很大 ⇒ 是 torch 没归还，Phase C 的 TMS region 有机会。

### Phase B — 同步 rollout 路径

**依据**：`vime_polar_sync_colocate_design.md` §2 已论证薄壳方案不可行（`_async_session_pool_loop` 的结构性 run-ahead），必须走独立的一次性提交路径。

**现状**：`start_sync_hybrid.sh` 入口是 `train.py`（同步），但 rollout 函数仍是 `generate_rollout_polar_async`（经 `run-qwen36-35b-polar-multi-pd.sh:241`），并带 `--rollout-scheduler-mode session_pool`、`--rollout-max-async-level`、`--use-tis`、`POLAR_DRAIN_SESSIONS=0`。同步循环配异步 worker ⇒ `generate()` 返回时 polar 侧仍有在飞 session，紧接着 vLLM sleep。

| # | Commit | 内容 |
|---|---|---|
| B1 | `refactor(bridge): lift task rejection check to module scope` | `_task_rejection_reason` 提升到模块级，worker 方法改为一行委托。行为逐位相同 |
| B2 | `feat(bridge): add synchronous one-shot polar rollout path` | `_submit_train_groups` + `generate_rollout_polar_sync` + `--rollout-sync-oversubscribe-factor`（默认 1.0）+ `_abort_inflight` no-op 占位；补数逻辑不可省 |
| B3 | `test(bridge): cover sync rollout path and async non-regression` | 设计文档 §5.3 的 7 项，含「同步路径调用图不含 `AsyncPolarRolloutWorker`」的静态断言 |
| B4 | `feat(scripts): give the sync hybrid run a synchronous rollout contract` | 切 `..._sync`；去 `--use-tis` / session_pool 四件套 / `max-async-level` |

B1 必须先于 B2。B4 依赖 B2。

### Phase C — offload 后端可选

a3-pd **保留**了完整 TMS 管线（`actor.py:20` import、`:243` pause、`:260` resume、`update_weight/common.py:141-143` `get_cpu_backup`、`actor_group.py:70-84` LD_PRELOAD、`megatron_utils/__init__.py:22-32` deep_ep 门控），只是 `actor.py` 里 NPU 分支硬编码为 storage-resize。

不选 tms 时零影响，三层保证：

1. `TorchMemorySaver.__init__` 只设 `_impl_ctor_kwargs={}` / `_impl=None`；`torch.npu.MemPool` 与 .so 加载都在 `_ensure_initialized()`，仅由 `region()` / `cuda_graph()` 触发。**不进 region ⇒ 不装分配器**
2. `megatron_utils/__init__.py` 的 `if _impl is not None` 守卫已在生产中每次启动都走（NPU 上恒 no-op）
3. `expandable_segments` 冲突检查本身 tms-gated；`auto` → NPU 解析成 `storage-resize`，路径逐位不变

| # | Commit | 内容 |
|---|---|---|
| C1 | `feat(npu): make the offload backend selectable and restore the TMS path` | `--npu-offload-backend {auto,tms,storage-resize}` + `_resolve_npu_offload_backend`；`actor.py` 三处 `if is_npu()` 换成单一 `self._use_tms`；NPU+tms 时用 `torch_memory_saver.region(tag="training", enable_cpu_backup=True)` 包住 `initialize_model_and_optimizer` |
| C2 | `test(npu): cover offload backend resolution` | `auto` 在 NPU 上解析为 `storage-resize`；tms + `expandable_segments:True` 报错；`_use_tms` 单一开关同时决定 region 与 pause/resume |

⚠️ **`_use_tms` 必须是单一表达式**。`entrypoint.py` 的 `pause()`/`resume()` 内部直接 `self._impl.pause()`，**没有** `_ensure_initialized()`；两处各写各的 `if` 一旦不一致，报的是 `AttributeError: 'NoneType'` 而非可读错误。

**不搬**：polar-bridge 的 `NPUWeightOffloader`（无 `param_restorer`，多占一份 model-sized host 副本）、它的 `memory_utils.py`（缺 a3-pd 的四个函数）、它的同构共卡脚本（改用全 share layout 表达）。

### Phase D — layout 统一（本次不做）

a3-pd 的 `share="actor"` 已能表达同构共卡（share 段覆盖 actor 全部卡、无专用段），经复核可通过 `_validate_layout` 全部校验，且 `_offload_engine_indices` / `_compute_rollout_offset` / `_build_layout_bundles` 在该退化情形下行为正确。

但 `--colocate` 与 `--resource-layout` 互斥（`arguments.py:1837`）仍在，`start_cola.sh` 走的是第二条代码路径。维护两条共卡路径本身是 bug 来源。

| # | Commit | 内容 |
|---|---|---|
| D1 | `feat(layout): express homogeneous colocate as an all-share layout` | 新增全 share layout + 与 `--colocate` 路径对拍 |
| D2 | `refactor(args): retire --colocate into layout sugar` | `--colocate` 降级为生成等价 layout 的语法糖 |

Phase D 需真机对拍，排在拿到 Phase A 账本之后。

---

## 3. 执行顺序与验证

```
A1 → A2 → A3        CPU 测试可验证
B1 → B2 → B3 → B4   CPU 测试可验证
C1 → C2             CPU 测试可验证
──────── 以上无需机器 ────────
拆账（真机 NUM_ROLLOUT=2，VIME_MEM_PROBE=1）
   ↓
据账本决定是否试 tms；决定 util 天花板
   ↓
D1 → D2             需真机对拍
```

真机验证按设计文档 §5.4 分阶段：`--debug-rollout-only` → `--debug-train-only` → `NUM_ROLLOUT=2` 全循环（核心验收点：日志中 staleness 恒 0）→ 长跑。

**真机 run 由用户自己起，本方案不代跑。**

---

## 4. 回滚

各 commit 之间除标注的依赖外无耦合，可逆序 revert。Phase A 纯观测；Phase C 默认路径逐位不变；Phase B 为新增路径，回滚 = 启动脚本切回 `..._async`。

---

## 5. 已知缺口（不阻塞，记账）

| 项 | 说明 |
|---|---|
| `--polar-gateway-url` 缺失 | `arguments.py:706` 只注册 `--polar-url`，`_resolve_gateway_url` 恒返回 `None`。gate 未来的超订 + abort（设计文档 §7）。polar 侧端点已核对存在：`src/polar/gateway/server.py:505` inflight 探针、`:681` `DELETE /sessions/{id}` |
| `CeilEpochRolloutDataSourceWithBuffer` 未接线 | 两个启动脚本均未设 `--data-source-path`，epoch 长度走 floor，尾部 prompt 被跳过 |
| 9.8G 未拆解 | 见 §0.2，Phase A 的产出 |
