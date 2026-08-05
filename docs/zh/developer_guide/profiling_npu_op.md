# 昇腾算子级 Profiling(Ascend PyTorch Profiler)

> 面向 **NPU** 推理引擎内部的**算子/内核级**性能采集,补端到端遥测(卡负载/逐请求耗时)之外的细节:
> 哪些算子最耗时、prefill/decode 内部构成、调度 step 时间。
> 全程走**昇腾原生**工具(`torch_npu.profiler` → `ascend_pt` → `analyse`/MindStudio),不涉及 CUDA。
>
> 与本目录 `profiling.md` 的区别:那份讲通用 vLLM profiler 流程(措辞沿用上游 slime,偏 CUDA/trace.json.gz);
> 本文专讲昇腾 NPU 落地,并纠正几个上游遗留脚本在 NPU 上的不适用点(见 §陷阱)。

## 机制(为什么这么配)

- vLLM 的 `--profiler-config` 在 vime 里前缀为 `--vllm-profiler-config`,作为非原始值经
  `_forward_vllm_cli_args` **原样字符串转发**给 `vllm serve` 子进程。
- vllm-ascend 的 `TorchNPUProfilerWrapper`(vllm_ascend/profiler/torch_npu_profiler.py)把它落成
  `torch_npu.profiler`,产物为每 worker 一个 `<host>_<pid>_<ts>_ascend_pt/`。
- `/start_profile` 是**零参调用**(entrypoints/serve/profile/api_router.py),POST body 被忽略 →
  配置只能在**启动时**由 `--profiler-config` 决定,运行时只能 start/stop。
- `max_iterations=N` 自动落盘**有效**:`WorkerProfiler.step()` 到 N 步自动 stop(profiler/wrapper.py)。
  前提 `ignore_frontend=true`。注意 NPU wrapper 未接 torch_npu schedule,故 `warmup/active_iterations` 无效。

## 采集流程

### 1. 带 profiler 启动引擎

在 `scripts/start-ascendc.sh` 命令前加开关(默认关,不影响正式跑):

```bash
PROFILE_OP=1 \
PROFILE_DIR=/mnt/share/polar_engine_metrics/opprof/$(date +%Y%m%d-%H%M%S) \
PROFILE_MAX_ITERS=20 \
bash scripts/start-ascendc.sh
```

`run-qwen36-35b-polar-ascendc.sh` 里的 `PROFILE_OP` 段会:
- `export VLLM_RPC_TIMEOUT=1800000`(30min;stop 落盘慢,默认 10s 会截断)
- 追加 `--vllm-profiler-config '{"profiler":"torch","torch_profiler_dir":"$PROFILE_DIR","ignore_frontend":true,"max_iterations":20,"torch_profiler_with_stack":false,"torch_profiler_record_shapes":true}'`

### 2. 确认路由已注册

```bash
bash tools/op_profile_collect.sh check 15000 15002 15004
# 200=已注册(带了 profiler 配置);404=启动时没带 PROFILE_OP=1,需重启
```

### 3. 采集

生产由 polar gateway 打流量,引擎只服务,**不需要 sleep_rollout**。有流量时:

```bash
# 首次冒烟建议只采一个引擎,降开销:
bash tools/op_profile_collect.sh start 15000
# 到 max_iterations 步该 worker 自动落盘;要提前停:
bash tools/op_profile_collect.sh stop 15000
```

无 polar 时可用 sleep_rollout + 直连 worker 发少量请求压测,见 `profiling.md`。

### 4. 解析成 CSV

```bash
bash tools/op_profile_collect.sh analyse "$PROFILE_DIR"
```

调 `torch_npu.profiler.profiler.analyse()`,在每个 `*_ascend_pt/ASCEND_PROFILER_OUTPUT/` 下生成:
- `op_statistic.csv` — 算子耗时占比(**看谁是大头**)
- `kernel_details.csv` — 内核级明细
- `step_trace_time.csv` — 调度 step 时间
- `trace_view.json` — 拖进 https://ui.perfetto.dev/ 或 MindStudio Insight 看时间线

## 陷阱(上游遗留,NPU 勿用)

1. **`tools/analyze_profile.py` 不能用于 NPU**:它是纯 CUDA 分析器(解析 `.trace.json.gz`、
   cudaGraphLaunch、NCCL、numSms),昇腾产物是 `ascend_pt/` + CSV,格式对不上。NPU 用本文 §4。
2. **`tools/profile_rollout.py` 的 payload 对 vLLM 无效**:`num_steps/output_dir/activities` 是
   SGLang 接口形状,vLLM 的 `/start_profile` 忽略。该脚本仅 start/stop 动作可用(本文 collect.sh 已替代)。
3. **必须 `VLLM_RPC_TIMEOUT` 调大**:否则 stop 落盘超时中断、trace 不完整(PROFILE_OP 段已自动设)。

## 进阶:AI Core 流水线指标(默认关)

要看"算子该不该这么慢"(算子 ≥50μs 时 vec/mac 流水线占用应 ≥80%,verl 判据),需 AI Core 指标,
当前 wrapper 写死关闭(torch_npu_profiler.py:`aic_metrics=AiCMetrics.AiCoreNone`)。需要时改为
`AiCMetrics.PipeUtilization` 重采(数据量与开销上升,`max_iterations` 要调小)。先按默认跑通再决定。
