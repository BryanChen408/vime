# Profiling 采集方案(推理侧内部细节)

> 背景:T1–T8 遥测已覆盖**端到端**负载(卡/引擎/请求/时间拆解,见 GUIDE_perf_report.md)。
> 本方案补**内部细节**两层:① 算子/内核级 trace;② 引擎服务框架级(请求/调度/KVCache 内部流转)。
> 调研结论:verl main、vime、slime-ascend、vllm-ascend 四方文档已比对(见文末参考),verl 的
> `global_profiler`/`discrete` 体系**不适用于我们**——那是 verl 管 rollout 的架构;
> vime 是 server-based 引擎(vLLM 子进程),profiling 走 HTTP API,天然离散。

---

## 方案总览

| 层 | 工具 | 采集对象 | 产物 | 分析工具 |
|---|---|---|---|---|
| ① 算子/内核级 | Ascend PyTorch Profiler(torch_npu,经 vLLM `/start_profile` API) | NPU kernel、算子耗时、调度 step 时间 | `*ascend_pt/` → `kernel_details.csv` `op_statistic.csv` `step_trace_time.csv` `trace_view.json` | MindStudio Insight / Perfetto / 自研 csv 分析 |
| ② 服务框架级 | MS Service Profiler(msserviceprofiler) | vLLM 服务框架函数级:Request / KVCache / BatchSchedule / ModelExecute 域 | `request.csv` `kvcache.csv` `batch.csv` `chrome_tracing.json` `profiler.db` | MindStudio Insight / pandas |

两层互补:①回答"算力花在哪些算子上、prefill/decode 内部长什么样";②回答"引擎内部调度、KV 管理、batch 组排哪里耗时"。与 T1–T8(外部负载)拼成完整链路。

---

## 方案①:算子级 trace(vime 现成流程 + vllm-ascend NPU profiler)

### 链路(每步均有文档/代码实锤)

1. **启动 train 时让 rollout 等待**:`--rollout-function-path vime.rollout.sleep_rollout.sleep`
   (vime 自带 `vime/rollout/sleep_rollout.py`,rollout 初始化后挂起,便于手动发请求)
2. **透传 profiler 配置给 vllm serve 子进程**:
   ```bash
   --vllm-profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/logs/vllm_profile","max_iterations":3,"ignore_frontend":true}'
   ```
   - `max_iterations`:worker 记录超过 N 步后自动 stop 落盘,发完请求通常无需手动 stop
   - `ignore_frontend: true`:只 profile worker,降低前端开销
   - vllm-ascend 侧由 `vllm_ascend/profiler/torch_npu_profiler.py` 落地为 torch_npu 采集(本地代码已确认存在)
   - 注意:vLLM 默认开 python stack,数据膨胀大;不需要则加 `"torch_profiler_with_stack": false`
3. **`VLLM_RPC_TIMEOUT` 必须调大**(vime 文档实测坑):默认 10s,stop_profile 落盘耗时数分钟会被截断。
   仅 shell export 不一定进 Ray job,须写进 `ray job submit --runtime-env-json`:
   ```json
   {"env_vars": {"VLLM_RPC_TIMEOUT": "1800000", "PYTHONPATH": "/root/Megatron-LM"}}
   ```
4. **启动后三项自检**(缺任一即未生效):
   - 日志有 `vllm_profiler_config ... profiler='torch'`
   - 日志有 `Launching vLLM server: ... --profiler-config {...}`
   - vLLM 路由列表含 `/start_profile` `/stop_profile`(否则 POST 返回 404)
5. **抓 trace**(train 就绪后另开终端):
   ```bash
   cd /root/vime
   python tools/profile_rollout.py --router-url http://127.0.0.1:<router_port> --action start
   # router 端口每次 job 随机(3000-4000),从日志 'Router launched at' 取;
   # curl http://127.0.0.1:<port>/workers 确认 worker 健康
   # 直连 worker 发 2~4 条请求即可(trace 很大):
   curl -X POST http://127.0.0.1:15000/v1/completions -H "Content-Type: application/json" \
     -d '{"model":"<HF ckpt 路径>","prompt":"Hello","max_tokens":32}'
   # max_iterations 到步自动落盘;需提前结束再 --action stop
   ```
6. **解析与查看**:
   ```python
   from torch_npu.profiler.profiler import analyse
   analyse("/root/logs/vllm_profile/localhost.localdomain_*_ascend_pt/")
   ```
   - `trace_view.json` → Perfetto(https://ui.perfetto.dev/)或 MindStudio Insight
   - `op_statistic.csv` / `kernel_details.csv` → 算子/内核耗时占比
   - vime 自带聚合分析:`python tools/analyze_profile.py --profile-dir /root/logs/vllm_profile --all-ranks`

### 完整可运行示例
vime 文档 §8 给了两段式脚本(launch / profile 两个终端),存为 `run_profiling_demo.sh` 直接用,
按其注释改模型路径与卡布局即可。参考:`/workspace/vime/docs/zh/developer_guide/profiling.md`。

### 采集级别建议(对照 verl 阈值方法论)
- 默认 level1 才能看到 AI Core 流水线指标;算子性能判据:单算子 ≥50μs 时 bound 流水线占用应 ≥80%
  (FA vec ~88%、Matmul mac ~90% 为合格参考);Host bound = host 耗时 > device 的 top 算子
- 我们的场景是 vLLM 引擎内采集,level 由 vllm-ascend 固定(Agent Loop 约束:不支持 analysis/自定义 level,
  默认 level1、离线 analyse)——这点 verl 文档与 vllm-ascend 行为一致

---

## 方案②:服务框架级(MS Service Profiler)

算子级之外,补"引擎内部在干什么":Request 生命周期、KVCache 分配/释放、Batch 组排耗时。

1. 工具随 CANN Toolkit 预装;源码升级:
   ```bash
   git clone https://gitcode.com/Ascend/msserviceprofiler.git
   cd msserviceprofiler && bash scripts/build_and_upgrade.sh
   ```
2. 启动服务前设环境变量(对 vllm serve 进程生效,故要在 vime 拉起 vLLM 的环境注入):
   ```bash
   export SERVICE_PROF_CONFIG_PATH=ms_service_profiler_config.json   # 不存在会自动生成默认
   export PROFILING_SYMBOLS_PATH=service_profiling_symbols.yaml      # 可选,默认按 vllm 版本加载
   ```
3. 开启采集:`sed -i 's/"enable":\s*0/"enable": 1/' ms_service_profiler_config.json`
   关键配置:`domain`(Request;KVCache;ModelExecute;BatchSchedule;Communication)、
   `acl_task_time`(0 关 / 1 ACL_TASK_TIME_L0 / 2 MSPTI / 3 Torch Profiler)、`timelimit`(建议 ≥120s)
4. 正常发请求(可复用真实 rollout 流量或手动 curl)
5. 解析:`cd ~/.ms_server_profiler/<启动时间戳目录> && msserviceprofiler parse --input-path=./ --output-path output`
6. 产物:`request.csv`(逐请求内部各阶段)、`kvcache.csv`、`batch.csv`、`chrome_tracing.json`

> 与现有 T4(engine-*.jsonl 逐请求)的关系:T4 是引擎**对外**的逐请求结果;②是引擎**对内**的
> 调度/KV/批排时间线,两者互补不重复。

---

## 执行顺序建议

1. 先在一台 infer 引擎(infer-0,端口 15000)按方案①跑一次最小验证(sleep_rollout + 3 条请求),
   确认 `ascend_pt` 能产出、能 analyse——链路打通优先于数据量
2. 通之后,用真实 rollout 流量抓 1 轮窗口(对齐 T1–T8 的 W0/W1,内外数据可交叉验证)
3. 方案②单独起一次服务采集(它与 ① 都注入在 vLLM 启动环境,建议分开跑,避免互相干扰归因)
4. 分析口径照 verl《性能分析指南》:先整体(Timeline 空泡、负载均衡)后细粒度(算子/通信/内存)

---

## 参考文档

### 本地(已核实存在)
| 路径 | 内容 |
|---|---|
| `/workspace/vime/docs/zh/developer_guide/profiling.md` | vime profiling 全流程(方案①主依据,含 §8 完整脚本、VLLM_RPC_TIMEOUT 等坑) |
| `/workspace/vime/tools/profile_rollout.py` | 经 router 对所有 worker start/stop profile |
| `/workspace/vime/tools/analyze_profile.py` | trace 聚合分析 |
| `/workspace/vllm-ascend/docs/source/developer_guide/performance_and_debug/service_profiling_guide.md` | vllm-ascend 两套 NPU 方案(方案①②的官方依据) |
| `/workspace/vllm-ascend/vllm_ascend/profiler/torch_npu_profiler.py` | NPU profiler 落地代码 |
| `deploy/ascend_operator/telemetry/GUIDE_perf_report.md` | 既有 T1–T8 端到端遥测口径(本方案的上层) |

### 线上
| 网址 | 内容 |
|---|---|
| https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/dev_guide/performance/ascend_profiling_zh.rst | verl 最新 profiling 采集(v0.7.1 的更新版;两级配置/level/contents/离散模式;其 global_profiler 架构不适用于 vime,仅作 NPU 采集参数语义参考) |
| https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/dev_guide/performance/ascend_performance_analysis_guide.md | verl 性能分析方法论(指标阈值:算子 ≥50μs 流水线占用 ≥80%;Host bound;Shape 512B 对齐) |
| https://github.com/THUDM/slime/blob/main/docs/en/developer_guide/profiling.md | slime 上游 profiling(vime 文档的源头,SGLang 版) |
| https://gitcode.com/Ascend/slime-ascend | slime 昇腾官方 fork(基于 slime v0.3.0;其 profiling.md 无 NPU 定制,已核实) |
| https://gitcode.com/Ascend/msserviceprofiler | 方案② 工具源码 |
| https://ui.perfetto.dev/ | trace 查看 |

---
*2026-08-05 调研整理。所有结论均有上述文档/本地代码实锤;vime 为 slime fork(训练栈 Megatron,rollout 换 vLLM+vllm-router),verl 体系不适用已说明理由。*
