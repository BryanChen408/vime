# 特性叠加结果存档 — vime+polar qwen3.6-35B-A3B RL rollout

> 打结时间:2026-07-15。本文是**删除运行时产物(/home/docker/logs、output/polar_bridge)、打包镜像前**的结论存档。
> 原始日志(将删):`train_qwen36_polar_20260715-{135826,140921,144301,150552}.log`。
>
> **标注约定**:【实测】= 本工况真机跑出的数;【推论】= 读码/理论,未 profile/未跑,**别当事实**;【文档事实】= ref.md/ref2.md 或代码里白纸黑字。

---

## 0. 工况(所有结论的前提)

- 任务:**polar 算子 RL rollout**,**突发式 agentic**(session 频繁为工具/算子执行暂停,生成只占每 session 墙钟一小片)。
- 拓扑:双机,88=train(8 卡 0-7),52=infer(16 卡 = 4×TP4),polar 占 88 的 8-15(不在 ray)。
- 规模:`ROLLOUT_BATCH_SIZE=2 N_SAMPLES=2 GLOBAL_BATCH_SIZE=4`(debug 小 batch)。
- 栈:vllm 0.21.0 + vllm-ascend + Megatron/MindSpeed,GRPO+TIS。

## 1. 一句话结论

**这个突发 agentic 工况下,ref 那条"饱和 serving"路线(external-LB DP + Balance + 跨 DP EP)是净损或无收益;正解是独立副本(external-LB 关)+ 引擎内 EP。DP 实现本身正确(TIS=1.0046),错的是把饱和场景的配置搬进了突发场景。** 详见 §4 机制与 §5 regime。

## 2. 逐特性结果表

| 特性 (开关) | 本次状态 | 结果 | 当前建议 |
|---|---|---|---|
| **external-LB 分布式 DP** (`FEAT_DP_EXTERNAL_LB`) | 开→关做过 A/B | 【实测】开=净损 **2.3×**(e2e 15→35,关掉即回),唯一变量。见 §3.1 | **关**(用独立副本) |
| **Balance scheduling** (`FEAT_BALANCE_SCHED`) | 开/关都跑过 | 【实测】中性(external-LB 开时开关都 ~15)。【文档事实】其目的是跨 DP EP 的 rank 对齐,EP 关时无收益 | **关**(除非开跨 DP EP) |
| **跨 DP EP** (`FEAT_CROSS_DP_EP`) | **从未跑** | 代码就绪(开关 + vLLM c04b1dea4 两机都在 + 自动展平 ep_size=dp×tp)。【推论】继承 DP 锁步税;大 batch 翻盘未验 | **暂不开**,见 §6 |
| **LB proxy 透传** (`FEAT_LB_PROXY`) | 开 | 【实测】工作正常(4 路/独立副本分发、保 return_token_ids + 会话亲和) | 开 |
| **multistream 共享专家** (`FEAT_MULTISTREAM_SHARED_EXPERT`) | 开 | 【实测】生效(日志 "Multistream overlap shared expert is enabled") | 开(ref2 默认 false,我们保留,已过 token-faith) |
| **static kernel** (`FEAT_STATIC_KERNEL`) | 开(名义) | 【实测】**运行时被静默关掉**——mp 起的引擎无 `LOCAL_WORLD_SIZE` → "static kernel feature will be disabled"(8 次)。**实际 inert** | 待修(§7) |
| **prefix cache** (`FEAT_PREFIX_CACHE`) | 开 | 【实测】vLLM 引擎层命中 80-88%,但 **rollout 层 `prefix_cache_hit_rate=0.0`**(待查,§7) | 开 |
| **npugraph_ex / HCCL AIV** (`FEAT_HCCL_AIV`) | 开 | 本次未单独 A/B,随栈跑通无崩 | 保持 |
| **rollout EP / FlashComm1** (`FEAT_ROLLOUT_EP`/`FEAT_FLASHCOMM1`) | **本次关** | FlashComm1=1 会拉起引擎内 EP=4(experts/card=64)。是历史基线档,本次为做 DP-smoke 压成 0 | 见 §5 结论:引擎内 EP 才是本工况正解 |
| **训练 expandable** (`FEAT_TRAIN_EXPANDABLE`) | 开 | 【实测】修好训练 actor OOM(reserved ratchet),静态显存对齐 slime,长跑站住。见 §3.3 | 开 |
| **每步 empty_cache** (`VIME_EMPTY_CACHE_PER_STEP`) | 开 | 【实测】每步收回 ~10G、reserved 复位,配合上条防 OOM | 开 |
| **opt-level 2** (`FEAT_OPT2`) | 关 | 未验;MindSpeed level-2 fusion。blast-radius 项,待单独验 token-faith+GDN | 保持关,单列 |

## 3. 关键实测数据(存档)

### 3.1 external-LB DP 净损(核心结果)
- 干净相对 A/B,**唯一变量 = `FEAT_DP_EXTERNAL_LB`**,其余(polar、batch、EP 关)全同:
  - external-LB **开**(140921/144301):observer e2e ≈ **15**;`perf/wait_time_ratio ≈ 0.866`(训练 86.6% 时间在等 rollout)。
  - external-LB **关**=独立副本(150552):observer e2e ≈ **35**(**2.3×**);`perf/wait_time_ratio ≈ 0.740`(等待占比下降 → rollout 变快,日志佐证 observer)。
- Balance 开(140921)vs 关(144301):e2e 都 ≈15 → **税不在 Balance,在 external-LB DP 本身**。

### 3.2 token-faith(DP+Balance 数值正确性,step 0,140921)
```
train/tis                          = 1.0045503     (≈1.00 ✓)
train/train_rollout_logprob_abs_diff = 0.0379850   (≲0.05 ✓)
train/ois                          = 1.0           (✓)
train/pg_clipfrac                  = 0.0           (✓)
train/ppo_kl                       = 0.0           (✓)
train/tis_clipfrac                 = 0.0019283
train/tis_abs                      = 0.0309491
train/grad_norm                    = 1.4926
```
权重同步:`external_lb=True world_size=17`(16 rollout + 1),128 张量全量广播干净收尾。→ **external-LB DP + Balance 结构+数值双闭合;实现正确,只是这个工况用不上它。**

### 3.3 训练侧 OOM 修复(实测有效)
- `FEAT_TRAIN_EXPANDABLE=1`(把 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 经 `--train-env-vars` 灌到训练 actor)+ `VIME_EMPTY_CACHE_PER_STEP=1`:step 0 reserved 爬到 ~45.5 后每步收回 cached_free ~10.5G,dev_free 保持 7-19G,长跑**无 OOM**。详见 memory `vime-oom-reserved-ratchet-emptycache`。

## 4. 为什么 external-LB DP 在此工况有害(机制)

【推论,未 profile,读码得】vLLM 原生 DP busy-loop `vllm/v1/engine/core.py:1794-1846`:
- DP 组按 **wave 锁步**:只要任一 peer rank 有活(`engines_running=True`),本步**没活**的 rank 被迫 `execute_dummy_batch()`(**core.py:1821**)——跑一次全量假前向,产出为 0;wave 要**全部 rank 同时清空**才结束(`_has_global_unfinished_reqs` all-reduce **每 32 步**一次,非主因)。
- 突发 agentic → 各 rank duty-cycle 低、极少同忙 → **~3/4 卡持续烧 dummy** → 16 卡只交付约 1 个引擎的有效吞吐 → 2.3× 损。
- 独立副本(external-LB 关)`data_parallel_size=1`:**无 DP 组、无 wave、无 dummy**,空闲零成本,各引擎全速 → 拿回 2.3×。

## 5. 为什么 ref 用这套、我们不该照搬(regime)

【文档事实】ref.md/ref2.md = **GLM-5 W4A8 一体机饱和 serving 压测**(ref2.md:139 aisbench `--concurrency 16 --request_rate 0 --input_len 32768 --output_len 500 --prefix 90% --dp 2`),目标 TPOT≤50ms、固定卡榨吞吐。
- 满载 → 各 rank 队列**永远非空** → `not executed` 几乎不真 → **dummy 税≈0** → DP 把连续大 batch 并行化(赚)+ Balance 守 TPOT SLO(赚)+ 跨 DP EP 分专家省显存换 KV(赚)。
- **我们是"走走停停的 rollout",站在吞吐曲线另一端**:同一套配置,满载天堂 / 突发地狱,分水岭 = **rank 占用率(duty-cycle)**。

**推论(未验)**:大 batch 下 DP+EP 赢独立副本靠 ① expert 权重分片 256→16/卡 省显存换 KV ② per-expert GEMM 跨 rank 聚合提 MFU——但这俩**只在饱和时兑现**,且 agentic 突发性是否 batch 大了也消不掉**没测**。→ **本工况正解 = 独立副本 + 引擎内 EP(FlashComm1 那套,各副本 TP 内分 expert,无跨引擎锁步)**;要更狠分片走加大单引擎 TP。

## 6. 跨 DP EP 就绪状态(代码在,未跑)

- ✅ 脚本开关 `FEAT_CROSS_DP_EP`(run 脚本,含护栏:强制 DP、建议 Balance)。
- ✅ vLLM EP 权重 loader 补丁 **c04b1dea4**("re-attach FusedMoE weight_loader for EP RL weight-sync")—— **88 和 52 都确认在**。
- ✅ vLLM 自动展平 `ep_size=dp×tp`(`config.py:1202`)+ `determine_expert_map` 自动跳非本地 expert → 无需 vime 改权重映射。
- ❌ **从未真机 EP=16 实跑,token-faith 未验**。开之前先看 §5:很可能背着 2.3× DP 税翻不了盘。

## 7. 未决 / 下一步

1. **饱和度探针**:拉高 `POLAR_MAX_ACTIVE_SESSIONS`(如 16→64)+ 加并发 rollout,看 `tokens_per_gpu_per_sec`/KV 占用能否顶起来。上得去 → DP/EP 才有戏;上不去 → 本工况就是 agent-bound,DP/EP 只当容量手段。
2. **static_kernel 掉线**:给 mp 引擎注入 `LOCAL_WORLD_SIZE`,否则 `FEAT_STATIC_KERNEL` 永远 inert。
3. **prefix cache rollout 层 0.0**:vLLM 引擎层命中 80%+ 但 `rollout/prefix_cache_hit_rate=0.0`,查这两个口径为何不一致。
4. **跨 DP EP 实跑**:仅当 §7.1 证明能饱和、且愿承担 2.3× 起点时再做。

## 8. 本工况推荐配置(结论落地)

```
# start.sh 特性行(突发 agentic RL rollout 正解)
FEAT_DP_EXTERNAL_LB=0 FEAT_BALANCE_SCHED=0 FEAT_CROSS_DP_EP=0 FEAT_LB_PROXY=1 \
FEAT_ROLLOUT_EP=? FEAT_FLASHCOMM1=?  # 引擎内 EP:建议开回(见 §5),但需重验 token-faith \
FEAT_PREFIX_CACHE=1 FEAT_MULTISTREAM_SHARED_EXPERT=1 FEAT_STATIC_KERNEL=1 FEAT_HCCL_AIV=1 \
FEAT_TRAIN_EXPANDABLE=1 VIME_EMPTY_CACHE_PER_STEP=1
```

相关 memory:`external-lb-dp-harmful-agentic-rollout`、`feature-eval-scaling-regime`、`follow-ref-before-diverging`、`vime-oom-reserved-ratchet-emptycache`、`vime-optlevel-fusion-gating`。

---

## 附录 A:逐 log 挖掘时间线(2026-07-15 挖,**原始/待整理**)

> 从 119 个 `train_*.log` 里挖的。目的是在删日志前把"哪个 run 开了什么 + 当时吞吐"落盘。
> **注意**:`[feat]` 行没进 tee(被 `set -x` 吞),特性从 args dump 点线格式还原(`vllm_data_parallel_external_lb .. True` 等);吞吐取 `perf/tokens_per_gpu_per_sec` 中位、已过滤 <500 的缓存瞬时垃圾值。**xLB/vEP/gpu/吞吐 可信;BAL/MS 当时 grep 有糊,待复核。**

| 日期-时间 | xLB | vEP | gpu | tok/gpu/s(中位) |
|---|---|---|---|---|
| 0713 fullstack | . | Y | 4 | 63 |
| 0713 layout | . | Y | 4 | 78 |
| 0713-154108 | . | Y | 4 | 147 |
| 0714-021642 | . | Y | 4 | 96 |
| 0714-063401 | . | Y | 4 | 84 |
| **0714-131706** | . | Y | **16** | **28** ← 切双机 16 卡 |
| 0714-143714 | . | Y | 16 | 36 |
| 0715-023942 | . | Y | 16 | 36 |
| 0715-031359 | . | Y | 16 | 13 |
| 0715-073624 | . | Y | 16 | 7 |
| 0715-092134 | . | Y | 16 | 9 |
| **0715-140921** | **Y** | . | 16 | **7.0**（DP external-LB 开，EP 关，Balance 开） |
| **0715-150552** | . | . | 16 | **13.0**（DP 关＝独立副本，EP 关） |

**可靠结论(有数支撑)**：
1. **最大吞吐变化 = 4→16 卡切换(0714-13:17)**：每卡吞吐 ~90→~20(掉 4×);乘卡数**聚合几乎不变**(≈360→≈320)→ **小 batch 喂不饱 16 卡,scale-out 白扩**。
2. **历史叠加 run 全 EP 开(vEP=Y),只有今天两个 DP 实验关了 EP** → "DP-smoke vs 历史"同时差 external-LB + EP 两项(混淆坐实)。
3. **唯一干净同日同 batch A/B**:140921(xLB 开)7.0 → 150552(xLB 关)13.0 = **~1.85×**,与 observer 15→35(2.3×)同向、量级吻合 → log 佐证 external-LB 净损。

**为什么给不出干净的逐特性增量(诚实限制)**：卡数中途 4→16、EP 历史开今天关、batch/workload 漂移(07-15 的 EP-on run 只 7-13,比 07-14 的 28-36 低是工况漂非特性差)、`tokens_per_gpu_per_sec` 噪声大、**叠加时判收益用的 observer 实时数没进 log**。→ 除 external-LB 这一个受控 A/B,其余步骤的 "+X%" 无法从 log 诚实还原。

**待整理 TODO**:① 复核 BAL/MS/其余特性列(逐 log args dump)；② 若有 observer 逐步记录,对齐到本表配置；③ 指认每个特性的受控对照 run-pair,抠完整 perf dict。
