# PD 分离 sleep/wake KV 传输失效(根因 B)修复设计方案

日期:2026-09-12
状态:探针验证完成,待评审后实施
关联文档:`pd_sync_precision_handoff_20260911.md`(整体问题背景)、`pd_sync_debug_handoff_20260910.md`(完整调试史)
探针脚本:`tools/mooncake_sleep_wake_probe.py`(本仓库)

---

## 0. 问题一句话

PD 分离 + 同步训推下,vLLM level-2 sleep 释放 KV 物理页、wake 时同 VA 重映射新物理页;RoCE 路径上 Mooncake/ADXL 网卡侧的 VA→PA 翻译**在注册时固化、进程内不可刷新**,导致 wake 后 KV 传输"成功返回"但读旧物理页(生产表现为全零 → rollout 乱码/语义串线)。

当前在线的 workaround 是 `VIME_KV_SLEEP_PERSISTENT=1`(KV 常驻不释放,~8G/卡),用户已明确否决,要求根修。

## 1. 根因定论(最小探针实证,非推断)

探针环境:NPU 4→8(RoCE,生产同构)、8→9(HCCS 同域),CaMem 池分配 + 生产同款 `TransferEngine.initialize(ip, "P2PHANDSHAKE", "ascend", "")`,`batch_transfer_sync_read` D 拉 P。

### 1.1 RoCE 路径(生产路径)

| 实验 | 结果 | 结论 |
|---|---|---|
| sleep/wake 后不做任何刷新 | ret=0,读到**旧页残留数据** | 根因最小复现 |
| 同 VA 重新注册(remap 后注销+注册) | ret=0,仍旧数据 | 无效 |
| 同 VA、**unmap 前先注销**、wake 后注册 | ret=0,仍旧数据 | 时序无关,无效 |
| P 重注册 + D 侧重建引擎(清 segment 缓存) | ret=0,仍旧数据 | 双端同 VA 刷新无效 |
| **wake 后换全新 VA 注册** | ret=0,**新数据正确到达** | **唯一有效路径** |
| 对照:再读旧 VA | 仍旧数据 | 陈旧按 VA 永久存在 |
| D 侧 remap 后同 VA 作接收方 | ret=0,数据落进**已释放旧页**,当前页零改动 | **目的侧同病,双侧都必须换 VA** |

机理结论:RoCE NIC 的 VA→PA 翻译在 `adxl_->RegisterMem` 时固化,`DeregisterMem+RegisterMem`(任意时序)、TransferEngine 重建、对端元数据刷新均不能使其重绑。该绑定为驱动/网卡层、按 VA 键控、进程级。ADXL 公开 API(`adxl_engine.h`)只有 Register/Deregister,无 refresh/rebind 原语。

### 1.2 HCCS 路径(对照实验)

| 实验 | 结果 |
|---|---|
| sleep/wake 后同 VA、**零刷新** | **完全正确,读到新数据** |
| 换新 VA | 正确 |

**HCCS 免疫**:HCCS 传输跟随设备侧页表(remap 时同步更新),不存在 RoCE 的固化翻译表。

HCCS 的附带发现:长连接模式下 HCCS 传输会在对端建立 peer 映射并钉住物理页,之后 `aclrtUnmapMem` 报 507899(DRV_INTERNAL_ERROR)——即 HCCS 长连接与 level-2 sleep 直接冲突;`ASCEND_USE_SHORT_CONNECTION=1`(传完即断链)可解除。

### 1.3 索引配对不变式(实现约束)

ADXL 传输按**注册顺序索引**配对两侧区域(P 区域[i] ↔ D 区域[i]):P 注册 2 个区域、D 只注册 1 个时,对 P 第 2 个区域的访问必败(503900);D 补齐第二个接收区后全通。生产 P/D 同模型同 KV 布局,各注册 10 个区域(pd_e2e_full2 日志实锤),天然满足。**任何修复必须维持两侧等长同序注册。**

### 1.4 上游 bug 报告口径(精确版)

> ADXL RoCE 路径:`RegisterMem` 建立的 VA→PA 翻译在底层 buffer 经历 `aclrtUnmapMem + aclrtFreePhysical + aclrtMallocPhysical + aclrtMapMem`(同 VA 换 PA)后,无法通过 `DeregisterMem + RegisterMem` 刷新,后续 TransferSync ret=SUCCESS 但搬运旧物理页数据;同流程 HCCS 路径不受影响。

---

## 2. 方案 A:RoCE 环境(当前生产)—— wake 换 VA + 双侧重注册 + 图重捕获

### 2.0 分层决策(2026-09-12 评审结论)

**主体实现在 vllm-ascend-023,vime 只留开关/编排/验证。** 决策依据:本修复操作的全是引擎内部结构(`model_runner.kv_caches`、`static_forward_context`、CaMem 单例、connector 注册表、graph dispatcher)——代码必须和这些内部结构同仓维护,否则 vllm-ascend 侧任何重构会让修复静默失效且晚炸。且这本质上是 vllm-ascend 自身"sleep mode × Mooncake connector"的组合缺陷,应以上游可合入的形态实现。曾评估过"vllm-ascend 只留薄钩子、主体放 vime 仓"(引擎 PYTHONPATH 含 vime_root,技术上可行),因依赖方向反转被否。

### 2.1 总体思路

wake 时 KV/state 池**不在旧 VA 上 remap**,而是全新分配(新 VA reservation),双侧按原顺序重新注册,decode cudagraph 清掉重捕获。sleep 行为不变(物理页真释放,无 8G 常驻)。

```
sleep(不变):
  allocator.sleep() → KV 物理页释放,VA 保留
  [新增] 解除 Mooncake 注册、释放 KV 的 VA 引用(见 2.3)

wake_up(改造):
  1. KV/state 池全新分配(新 VA)— 不走 CaMem 同 VA remap
  2. 更新全部引用:model_runner.kv_caches / static_forward_context 各层 /
     (若开 MTP)drafter 侧
  3. connector 双侧按原顺序注册新区域(维持索引配对不变式)
  4. D 侧丢弃缓存的 P segment 元数据(重建 TransferEngine,
     或 Mooncake 补丁暴露 closeSegment/openSegmentNoCache)
  5. CUDAGraphWrapper.clear_all_graphs() + model_runner.capture_model()
     重捕获 decode 图(FULL_DECODE_ONLY 保住)
```

### 2.2 为什么必须重捕获图(已核实)

- vLLM v1 attention 从 `static_forward_context[layer].kv_cache` 取 tensor,捕获时 kernel 启动参数烧死 `data_ptr()`(VA 常量);换 VA 后旧图必读写废地址。
- `torch_npu NPUGraph.update()` 存在但要求捕获时 `auto_dispatch_capture=True`,vLLM 不走该模式,不可用作指针热更新。
- 重捕获路径是上游现成的:`CUDAGraphWrapper.clear_all_graphs()`(vllm/compilation/cuda_graph.py:172)+ `model_runner.capture_model()`(内部会开 `set_cudagraph_capturing_enabled(True)`;运行时惰性补捕获被 `validate_cudagraph_capturing_enabled` 挡住,必须显式调)。
- 成本:上游注释全量捕获 5~20s;FULL_DECODE_ONLY 只有 decode 档,预计每引擎每次 wake 数秒,P/D 各引擎并行。

### 2.3 改动点清单(逐文件)

**vllm-ascend-023(主体):**

1. `vllm_ascend/worker/worker.py`
   - `sleep()`:level-2 且开关开时,connector 注销 KV 区域 → drop kv_caches 引用 → gc + empty_cache,借 CaMem free 回调释放 VA reservation(sleep 的 unmap+freePhysical 之外,把 VA 预留也还掉);
   - `wake_up()`:约 262 行处现有调试 hook(`global_te.recreate_engine` 调用)替换为新流程编排;**开关关时必须维持现状**(sleep-persistent workaround 路径不受扰);
   - `initialize_from_config()`(~1046-1058):KV 池 tag 分支随新路径调整(现有 sleep_persistent / kv_cache / nullcontext 三分支不动,新增分支只在开关开时生效)。
2. `vllm_ascend/worker/model_runner_v1.py`(新增方法,~150-200 行,工作量主体)
   - `reinitialize_kv_cache_at_wake()`:用缓存的 `self.kv_cache_config` 重跑 `initialize_kv_cache_tensors()`(新 VA)→ 全量重绑引用:`self.kv_caches`、`compilation_config.static_forward_context[layer].kv_cache`(含 GDN mamba state)、实施时 grep 见底的其余持有点(attn metadata builder、`_mamba_bufs`、MTP drafter);
   - 末尾 `CUDAGraphWrapper.clear_all_graphs()` + `self.capture_model()` 重捕获 decode 图。
3. `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`
   - `register_kv_caches` 增加重注册模式:区域推导逻辑复用(保证与 init 同序——索引配对不变式),更新 `self.kv_caches` 与新 base_addr 列表;调试期加的 epoch 失效机制复用,使对端丢弃旧元数据。
4. `vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py`
   - `GlobalTE` 增加 `registered_regions` 更新接口;`recreate_engine()` 已实现且语义正好(新引擎 → 新 handle_map_ → 重拉对端 segment → 注册新 VA),复用。
5. 视验证结果而定:`vllm_ascend/device_allocator/camem.py`(+`csrc/camem_allocator.cpp`)——优先纯 Python(drop 引用触发 free 回调释放 VA);走不通再给 KV tag 加"sleep 时连 VA 一并释放"语义(唯一可能的 C 层改动)。

**vllm-023(上游):零改动。** `clear_all_graphs` / `capture_model` 均为现成公开 API,只调用。

**vime(编排侧,三行量级):** 启动脚本/YAML 透传 `VIME_PD_FRESH_VA_ON_WAKE`;sleep/wake 编排不变;验证探针(`VIME_PD_SLEEP_WAKE_RELOAD_DIAG`、`VIME_PD_KV_XFER_HASH`)现成复用。

### 2.3.1 VA 换新机制(两个候选,实施第一步先验)

- **候选 a(倾向)**:KV 从 CaMem remap 语义剥离——sleep 时 connector 注销 + model_runner 释放 kv_caches 引用 + gc(free 回调连带释放 VA reservation);wake 时重跑 `initialize_kv_cache_tensors`(新 reserve → 必然新 VA)。语义最干净,先验证 `python_free_callback` 能正确释放 VA。
- **候选 b**:KV 留在 CaMem 池但打"不 remap"标记,wake 时池内重新分配。改动更局部但 CaMem 语义更绕。

**开关语义**:`VIME_PD_FRESH_VA_ON_WAKE=1` 开新路径;默认关;与 `VIME_KV_SLEEP_PERSISTENT=1`(现 workaround)**互斥,同开即 assert**。另注意:P 引擎 prefill 不走 FULL_DECODE_ONLY 的 decode 图,图重捕获成本集中在 D 侧。

### 2.3.2 影响面复核(2026-09-12 评审)

双重门控:`enable_sleep_mode`(纯推理不睡觉,新代码物理不执行)× `VIME_PD_FRESH_VA_ON_WAKE`(默认关)× `has_kv_transfer_group()`(非 PD 无连接器,误开开关也 no-op + warning)。

| 改动点 | 纯推理(无 sleep) | 非 PD 训推(有 sleep 无 PD) | PD rollout-only |
|---|---|---|---|
| sleep 释放 VA | 不执行 | 开关关→不执行 | 不睡觉 |
| wake 新流程 | 不执行 | 开关关→不执行 | 不睡觉 |
| KV 池 tag 分支 | 原 nullcontext 分支不变 | 原 kv_cache 分支不变 | 不变 |
| connector 重注册 | connector 不存在 | 不存在+开关关 | init 注册不变 |
| 图重捕获 | 不触发 | 不触发 | 不触发 |

结论:**纯推理 / 非 PD 训推 / PD rollout-only 三条对照路径零影响**;唯一行为变化面 = PD + 训推 + 显式开新开关。两个实施注意点:①wake_up 里调试期 recreate_engine hook 的替换必须保证"开关关=现状";②开关判定必须联合 `has_kv_transfer_group()`,防止非 PD 误开白付图重捕获。

### 2.4 风险与开放项

| 风险 | 评估 | 缓解 |
|---|---|---|
| 重捕获每 wake 数秒,影响 step 时间 | 中 | 实测;若不可接受再评估捕获集裁剪 |
| 重捕获后 graph pool 内存复用是否正确 | 中 | 全局共享 pool 设计上支持;E2E 验证显存曲线 |
| mamba/GDN state 张量引用遗漏 | 中 | 实施时全量 grep `kv_cache`/`_mamba_bufs` 引用点;E2E hash 校验 |
| D 侧 segment 刷新不彻底 | 低 | recreate_engine 已验证可拉新元数据(探针 phase4) |
| 新 VA 分配与 torch 缓存分配器交互 | 低 | CaMem 池外分配走标准 VMM,探针已验证 |

### 2.5 验证计划(按序)

1. 探针回归:`tools/mooncake_sleep_wake_probe.py` 保持绿(run15 形态)。
2. 引擎级 diag:`VIME_PD_LIFECYCLE_DIAG_ONLY=1 VIME_PD_SLEEP_WAKE_RELOAD_DIAG=1` + `VIME_PD_KV_XFER_HASH=1`,**不开** `VIME_KV_SLEEP_PERSISTENT` → 期望 wake+reload 后 P/D 40 层 hash 全一致、direct-vs-PD token 一致。
3. 生产形态 E2E:`start_sync_pd_single52.sh` 小批量(NUM_ROLLOUT=2),确认真实 session 输出正常、无 OOM、step 时间可接受。

---

## 3. 方案 B:HCCS 全联通环境(未来,如 A3 超节点)

### 3.1 原理

HCCS 传输跟随设备页表,remap 后同 VA 传输天然正确(探针实证)。全 HCCS 互联环境下**根因 B 不存在**,无需换 VA、无需重捕获图、无需 sleep-persistent。

### 3.2 需要做的事

1. **传输层走 HCCS**:去掉 `HCCL_INTRA_ROCE_ENABLE=1`(或置 0),ADXL 默认在 A2/A3 内走 HCCS。
2. **解决 unmap 冲突**(HCCS 长连接 pin 页导致 sleep unmap 507899),二选一:
   - B1(纯配置):`ASCEND_USE_SHORT_CONNECTION=1`。代价:每次传输重建链(实测建链 25~90ms),KV 传输热路径上不可忽略;
   - B2(推荐,需 Mooncake 小补丁):sleep 钩子前显式拆除 ADXL 连接(`Disconnect`/暴露 `closeSegment` 到 python),wake 后惰性重建。保长连接性能,sleep 前一次性清场。
3. **元数据/路由**:kv_port、拓扑 extra_config 按 HCCS 形态核对(参考 verl PR #7616 的 Ascend 配置模式);LocalCommRes 不需要。
4. **布局**:P/D 任意摆(全联通),无需域内配对约束。

### 3.3 待验证项(换 HCCS 环境后首批实验)

1. 探针 run:B2 断链方案下完整阶梯(含 D 侧 remap 作接收方——本轮因探针自身重复注册 artifact 未跑到,环境具备后补)。
2. 长请求 KV 传输带宽/延迟对比(HCCS 预期优于 RoCE)。
3. sleep/wake 全周期 + 真实 E2E。

### 3.4 与方案 A 的关系

互斥环境、可共代码:方案 A 的换 VA 逻辑用 `VIME_PD_FRESH_VA_ON_WAKE` 开关控制,HCCS 环境下关掉即可;两方案共享同一套探针与 hash 校验工具链。

---

## 4. 时间线与已投入

- 2026-09-11:根因 A(GDN recompute 分诊链)修复并 E2E 通过;根因 B 定位到注册失效,sleep-persistent workaround 上线。
- 2026-09-12 上午:最小探针(6 组实验)完成机制定论;HCCS 对照完成;cudagraph 重捕获可行性核实(NPUGraph.update 不可用 / clear+recapture 可用);verl PR #7616 评估(仅 level-1 sleep,不触及本问题)。

## 5. 附:探针使用

```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# RoCE(生产路径):
export HCCL_INTRA_ROCE_ENABLE=1
python3 tools/mooncake_sleep_wake_probe.py P 4 <host_ip> /tmp/probe_dir &
python3 tools/mooncake_sleep_wake_probe.py D 8 <host_ip> /tmp/probe_dir &
# HCCS 同域:unset HCCL_INTRA_ROCE_ENABLE; export ASCEND_USE_SHORT_CONNECTION=1; 设备对换同域(8/9)
# 结果:<dir>/results.jsonl,逐 phase JSON 行
```
