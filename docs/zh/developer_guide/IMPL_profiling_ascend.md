# NPU 精细 Profiling —— 实现方案(代码实锤版)

> 本文是 PLAN_profiling.md 的**可执行落地版**。所有结论均来自读 vime/vllm/vllm-ascend 源码,
> 不是文档措辞推测。凡与 PLAN 冲突处以本文为准(见 §0 更正表)。
> 目标:在 polar 生产拓扑(start-ascendc.sh:12 卡 = 3 engine×TP4,端口 15000/15002/15004)上,
> 采到 T1–T8 端到端之外的**引擎内部算子/调度细节**。

## 0. 相对 PLAN 的更正(读代码后)

| 项 | PLAN 旧述 | 代码实测(文件:行) | 影响 |
|---|---|---|---|
| analyze_profile.py | 用它分析 NPU trace | 纯 CUDA:解析 `.trace.json.gz`/cudaGraphLaunch/NCCL/numSms(analyze_profile.py:186-334) | NPU 弃用,改 §4 |
| `/start_profile` payload | num_steps/output_dir 等生效 | 零参调用 `engine_client.start_profile()`(vllm entrypoints/serve/profile/api_router.py:22-26) | payload 全忽略;配置只能启动时给 |
| max_iterations 自动落盘 | (含糊) | **有效**:WorkerProfiler.step→到 max 自动 stop(vllm profiler/wrapper.py:83-113);NPU 子类 _profiler_step 返 True(torch_npu_profiler.py:84) | 需 `ignore_frontend=true`;仅 delay/max 生效,warmup/active 被 NPU wrapper 忽略 |
| sleep_rollout | 生产也要 | 生产由 polar gateway 打流量,引擎只服务(start-ascendc.sh DEBUG_ROLLOUT_ONLY=1 actor=0) | 生产**不需要**;仅无 polar 的独立压测用 |
| RPC_TIMEOUT 注入 | 写 ray job runtime-env-json | 生产 `python3 train_async.py` 直跑(run-...-ascendc.sh:307);引擎=Ray actor 子进程,env=os.environ.copy()(vllm_engine.py:368) | 在 start-ascendc.sh 顶部、ray start 前 export 即可贯穿 |
| AI Core 指标 | level1 够 | wrapper 写死 `AiCMetrics.AiCoreNone`(torch_npu_profiler.py:53) | verl 的 ≥50μs→流水线≥80% 判据采不到,要 §5 patch |

## 1. 关键机制(接线依据)

- **参数透传**:vLLM `--profiler-config` 在 vime 里就是 `--vllm-profiler-config`(arguments.py 整包导入 AsyncEngineArgs 加前缀)。
  它是非原始值,`_forward_vllm_cli_args` 用 `_vllm_raw_values` **原样字符串转发**(vllm_engine.py:333-337) → 传什么 JSON 就转发什么。
- **ProfilerConfig 可用字段**(vllm/config/profiler.py):`profiler`、`torch_profiler_dir`、
  `torch_profiler_with_stack`(→NPU `with_modules`)、`torch_profiler_with_memory`(→`profile_memory`)、
  `torch_profiler_record_shapes`、`ignore_frontend`、`delay_iterations`、`max_iterations`。
  ⚠ NPU wrapper 未接 torch_npu schedule,故 `warmup/active_iterations` 无效。
- **产物路径**:`torch_profiler_dir` 下每 worker 一个 `<host>_<pid>_<ts>_ascend_pt/`(tensorboard_trace_handler)。
- **落盘慢→超时**:stop 时落盘可达数分钟,`VLLM_RPC_TIMEOUT` 默认 10s 会截断。必须调到 1800000(30min)。
- **互斥**:torch profiler 与 MS Service Profiler(msmonitor daemon)不能同时(torch_npu_profiler.py:46-47 assert)。故方案①②**分开跑**。

## 2. 方案①:算子级(Ascend PyTorch Profiler,推荐先做)

### 2.1 改 start-ascendc.sh(两处)
```bash
# (a) 顶部、ray start 之前:让 RPC 超时贯穿到引擎子进程
export VLLM_RPC_TIMEOUT=1800000

# (b) 传 profiler 配置(经 run 脚本进 VLLM_ARGS 或直接加在 train_async.py 参数)
#     token 区间控制我们没有(那是 verl profile_token_start/end),用 max_iterations 控量
PROFILE_DIR=/mnt/share/polar_engine_metrics/opprof/$(date +%Y%m%d-%H%M%S)
mkdir -p "$PROFILE_DIR"
--vllm-profiler-config '{"profiler":"torch","torch_profiler_dir":"'"$PROFILE_DIR"'","ignore_frontend":true,"max_iterations":20,"torch_profiler_with_stack":false,"torch_profiler_record_shapes":true}'
```
- `ignore_frontend:true`:只 profile worker(用 delay/max 时必须,否则 profiler.py:127 报错)。
- `max_iterations:20`:每 worker 记 20 个 engine step 自动 stop 落盘(先小后调;step≈调度步,非 token)。
- `with_stack:false`:关 python 调用栈,避免数据爆炸(vLLM 默认开,膨胀极大)。
- `record_shapes:true`:留 shape 供 verl 的 shape 亲和分析(512B 对齐)。

### 2.2 启动后三项自检(缺一即未生效)
1. 日志有 `--profiler-config {"profiler":"torch",...}`(vllm serve 命令行,vllm_engine.py:514 打印)。
2. curl 引擎 `POST /start_profile` 返回 200(非 404 → 路由已注册,attach_router:40)。
3. 发流量后 `$PROFILE_DIR` 下出现 `*_ascend_pt/`。

### 2.3 采集(生产:polar 已在打流量)
```bash
# 直接对三个引擎 start(payload 无所谓,vLLM 忽略;用 vime 脚本或 curl 均可)
for p in 15000 15002 15004; do curl -sS -X POST http://127.0.0.1:$p/start_profile; done
# polar 正常跑 → max_iterations=20 到步各 worker 自动 stop 落盘
# 若要提前停:for p in ...; do curl -sS -X POST http://127.0.0.1:$p/stop_profile; done
# 无 polar 时:sleep_rollout + 直连 worker 发 2~4 条 completion(见 PLAN §方案①)
```
注:vime `tools/profile_rollout.py` 可代替循环(经 router /workers 播 start/stop),但它带的
`--num-steps/--output-dir` 对 vLLM **无效**(§0),仅 start/stop 动作有用。

### 2.4 只采一个引擎(降开销,推荐首次)
只对 15000 一个引擎 start;或 profiler_dir 只配在需要的 engine。TP4 会出 4 个 rank 的 ascend_pt。

## 3. 方案②:服务框架级(MS Service Profiler)

与①互斥,**单独一次 run**。按 vllm-ascend service_profiling_guide.md:
```bash
# start-ascendc.sh 顶部(引擎子进程继承):
export SERVICE_PROF_CONFIG_PATH=/mnt/share/polar_engine_metrics/msprof/ms_service_profiler_config.json
# 首次自动生成默认;开采集:
sed -i 's/"enable":\s*0/"enable": 1/' "$SERVICE_PROF_CONFIG_PATH"
# 关键字段:domain="Request;KVCache;ModelExecute;BatchSchedule" acl_task_time=1 timelimit>=120
```
解析:`cd ~/.ms_server_profiler/<ts> && msserviceprofiler parse --input-path=./ --output-path output`
→ `request.csv`/`kvcache.csv`/`batch.csv`/`chrome_tracing.json`。补的是引擎**对内**调度视图
(T4 是对外逐请求结果,不重复)。

## 4. 分析(analyze_profile.py 不能用于 NPU!)

NPU 产物是 `*_ascend_pt/`,不是 CUDA 的 `.trace.json.gz`。三条路:
1. **离线 analyse 生成 CSV**(export_type 已是 Text):
   ```python
   from torch_npu.profiler.profiler import analyse
   analyse("<PROFILE_DIR>/<host>_<pid>_<ts>_ascend_pt/")
   ```
   → `ASCEND_PROFILER_OUTPUT/` 下:`op_statistic.csv`(算子占比)、`kernel_details.csv`、
   `step_trace_time.csv`、`trace_view.json`。
2. **Perfetto**:`trace_view.json` 拖进 https://ui.perfetto.dev/。
3. **MindStudio Insight**:载入 ascend_pt,看 Timeline/Overlap(verl 分析指南主推)。
> 若要复刻 analyze_profile.py 的算子分类汇总,需**新写**一个读 op_statistic.csv 的脚本
> (NPU 算子名与 CUDA 完全不同,不能复用 classify_kernel)。列为后续 TODO,不阻塞采集。

## 5. (可选)打开 AI Core 流水线指标 —— patch

verl《性能分析指南》的核心判据(算子 ≥50μs 时 vec/mac 流水线占用应 ≥80%)依赖 AI Core metrics,
当前 wrapper 写死关闭。要采需改 `vllm-ascend/vllm_ascend/profiler/torch_npu_profiler.py:53`:
```python
# 原: aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,   # 或 AiCoreAndMemoryBandwidth
```
代价:数据量与开销上升,采集步数要更保守(max_iterations 调小)。**先按 §2 默认跑通,确认要看流水线再打。**

## 6. 执行顺序

1. 只对 infer-0(15000)按 §2 采一次、max_iterations=20,确认 ascend_pt 产出 + analyse 出 CSV —— **打通链路优先**。
2. 通了再对齐 T1–T8 窗口(同一轮 rollout)全 3 引擎采,内外交叉验证。
3. 需要引擎内部调度视图时,单独跑 §3(与①互斥)。
4. 需要流水线占用判据时,§5 打 patch 重采。
5. 分析口径照 verl:先整体(Timeline 空泡/负载均衡)后细粒度(算子/通信/内存)。

## 7. 参考(见 PLAN_profiling.md §参考文档,补充本文用到的代码位置)
- vime 透传/env:`vime/backends/vllm_utils/vllm_engine.py`(build_vllm_cmd_and_env / _forward_vllm_cli_args / build_vllm_subprocess_env)
- vLLM profiler:`vllm/config/profiler.py`、`vllm/profiler/wrapper.py`、`vllm/entrypoints/serve/profile/api_router.py`
- NPU wrapper:`vllm-ascend/vllm_ascend/profiler/torch_npu_profiler.py`
- 生产脚本:`vime/scripts/start-ascendc.sh`、`scripts/run-qwen36-35b-polar-ascendc.sh`
- 分析陷阱:`vime/tools/analyze_profile.py`(CUDA-only,NPU 勿用)

---
*2026-08-05 读码定稿。max_iterations 自动落盘经 wrapper.py 确认有效;analyze_profile.py 经读码确认 CUDA-only。*
