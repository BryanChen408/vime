# vLLM Metrics Monitor - 自动启动集成

## 概述

训练脚本现已集成 vLLM metrics 监控面板自动启动功能。启动训练时,监控面板会自动在后台运行,无需手动操作。

## 功能特性

- ✅ **自动启动**: 训练脚本启动时自动启动监控面板
- ✅ **智能检测**: 自动检测并跳过已运行的监控实例
- ✅ **端口自适应**: 根据训练配置自动适配 vLLM 端口
- ✅ **日志记录**: 监控日志保存在 `logs/metrics/` 目录
- ✅ **依赖安装**: 首次运行自动安装 Flask/requests 依赖

## 快速开始

### 1. PD (Prefill/Decode) 模式训练

```bash
cd /workspace/vime
bash scripts/run-qwen36-35b-polar-minimal-single-rollout-only-pd.sh
```

启动后自动监控:
- **Engine 1 (Prefill)**: `http://<IP>:18100/metrics`
- **Engine 2 (Decode)**: `http://<IP>:18200/metrics`
- **Dashboard**: `http://<IP>:5000`

### 2. 非 PD 模式训练

```bash
cd /workspace/vime
bash scripts/run-qwen36-35b-polar-minimal-single-rollout-only.sh
```

启动后自动监控:
- **Engine 1**: `http://<IP>:18000/metrics`
- **Engine 2**: `http://<IP>:18002/metrics`
- **Dashboard**: `http://<IP>:5000`

### 3. 访问监控面板

训练启动后,在浏览器打开:

```
http://<训练机IP>:5000
```

**示例**:
```
http://80.48.5.64:5000
```

## 自定义配置

### 方式 1: 环境变量覆盖

```bash
# PD 模式
PREFILL_PORT=19100 \
DECODE_PORT=19200 \
METRICS_DASHBOARD_PORT=6000 \
bash scripts/run-qwen36-35b-polar-minimal-single-rollout-only-pd.sh

# 非 PD 模式
METRICS_ENGINE1_PORT=19000 \
METRICS_ENGINE2_PORT=19002 \
METRICS_DASHBOARD_PORT=6000 \
bash scripts/run-qwen36-35b-polar-minimal-single-rollout-only.sh
```

### 方式 2: 手动启动监控面板

如果需要独立启动监控面板(不依赖训练脚本):

```bash
# 使用默认端口
source /workspace/vime/scripts/common/start_metrics_monitor.sh

# 自定义端口
source /workspace/vime/scripts/common/start_metrics_monitor.sh 18100 18200 5000 80.48.5.64
#                                                            ^^^^^ ^^^^^ ^^^^ ^^^^^^^^^^
#                                                            E1端口 E2端口 面板 主机IP
```

### 方式 3: 直接运行 Python 脚本

```bash
python3 /workspace/vime/scripts/vllm_metrics_monitor_v2.py \
    --host 0.0.0.0 \
    --port 5000 \
    --engine1 http://80.48.5.64:18100/metrics \
    --engine2 http://80.48.5.64:18200/metrics \
    --interval 5
```

## 监控指标说明

### 组合指标 (Combined Metrics)

| 指标 | 说明 |
|------|------|
| **Total Generation Throughput** | 两个引擎的总生成吞吐量 (tokens/s) |
| **Total Running Requests** | 正在处理的请求总数 |
| **Total Waiting Requests** | 等待队列中的请求总数 |
| **Average Throughput (5min)** | 最近 5 分钟的平均吞吐量 |

### 单引擎指标

- **Engine #1**: 通常是 Prefill 实例(PD 模式)或第一个引擎
- **Engine #2**: 通常是 Decode 实例(PD 模式)或第二个引擎

每个引擎显示:
- Generation Throughput (生成吞吐量)
- Running Requests (运行中请求)
- Waiting Requests (等待中请求)

### 可视化图表

1. **Generation Throughput Over Time**: 总吞吐量时间序列(最近 300 个采样点,约 25 分钟)
2. **Concurrent Requests Over Time**: 两个引擎的并发请求数对比

## 文件结构

```
/workspace/vime/
├── scripts/
│   ├── vllm_metrics_monitor_v2.py          # 监控面板主程序
│   ├── common/
│   │   └── start_metrics_monitor.sh        # 自动启动脚本
│   ├── run-qwen36-35b-polar-minimal-single-rollout-only-pd.sh    # PD 训练(已集成)
│   └── run-qwen36-35b-polar-minimal-single-rollout-only.sh       # 非 PD 训练(已集成)
└── logs/
    └── metrics/                             # 监控日志目录
        └── vllm_metrics_YYYYMMDD_HHMMSS.log
```

## 日志和调试

### 查看监控日志

```bash
# 查看最新日志
tail -f /workspace/vime/logs/metrics/vllm_metrics_*.log

# 查看所有日志文件
ls -lht /workspace/vime/logs/metrics/
```

### 检查监控进程

```bash
# 查看监控进程
pgrep -af vllm_metrics_monitor_v2.py

# 停止监控进程
pkill -f vllm_metrics_monitor_v2.py
```

### 常见问题

#### 1. 监控面板无法访问

**症状**: 浏览器无法打开 `http://<IP>:5000`

**排查**:
```bash
# 检查监控进程是否运行
ps aux | grep vllm_metrics_monitor_v2.py

# 检查端口是否被占用
netstat -tlnp | grep 5000

# 检查防火墙
curl http://localhost:5000
```

**解决**:
- 确保监控进程正在运行
- 检查防火墙规则允许 5000 端口
- 如果端口冲突,使用 `METRICS_DASHBOARD_PORT` 环境变量更改端口

#### 2. 显示全零或无数据

**症状**: 面板显示,但所有指标都是 0 或 `--`

**排查**:
```bash
# 手动测试 vLLM metrics 端点
curl http://80.48.5.64:18100/metrics
curl http://80.48.5.64:18200/metrics
```

**解决**:
- 确认 vLLM 引擎已启动并正常运行
- 检查 vLLM 端口配置是否正确
- 查看监控日志中的错误信息:
  ```bash
  tail -100 /workspace/vime/logs/metrics/vllm_metrics_*.log | grep ERROR
  ```

#### 3. 吞吐量为负或异常跳动

**症状**: 吞吐量显示负数或剧烈波动

**原因**: vLLM 引擎重启导致累计计数器重置

**解决**: 
- 重启监控面板(会重新初始化计数器):
  ```bash
  pkill -f vllm_metrics_monitor_v2.py
  source /workspace/vime/scripts/common/start_metrics_monitor.sh
  ```

#### 4. 依赖安装失败

**症状**: 启动时提示 `ModuleNotFoundError: No module named 'flask'`

**解决**:
```bash
pip install flask requests
```

## 技术细节

### vLLM 0.23+ 兼容性

监控面板使用 **delta 计算法** 从累计计数器计算吞吐量:

```python
# vLLM 0.23+ 使用累计计数器
current_total = vllm:generation_tokens_total

# 计算 delta
delta_tokens = current_total - previous_total
throughput = delta_tokens / interval_seconds

# 更新状态供下次采样
previous_total = current_total
```

### 采样间隔

默认每 5 秒采样一次。可通过 `--interval` 参数调整:

- **更短间隔(2-3s)**: 响应更快,但网络开销略高
- **更长间隔(10-15s)**: 减少开销,但更新延迟
- **推荐**: 5-10 秒,平衡实时性和开销

### REST API

监控面板提供 JSON API 供其他工具集成:

```bash
curl http://localhost:5000/api/metrics
```

响应示例:
```json
{
  "last_update": "2026-08-04T15:30:45.123456",
  "engine1": {
    "generation_throughput": 1234.5,
    "running_reqs": 16,
    "waiting_reqs": 0
  },
  "engine2": {
    "generation_throughput": 2345.6,
    "running_reqs": 24,
    "waiting_reqs": 2
  },
  "combined": {
    "generation_throughput": 3580.1,
    "running_reqs": 40,
    "waiting_reqs": 2
  },
  "avg_throughput": 3520.3,
  "history": {...}
}
```

## 参考文档

- 原始参考: `/home/l00830933/workspace/vllm_metrics/vllm_metrics_monitor_v2_integration_guide.md`
- vLLM metrics 文档: `https://docs.vllm.ai/en/latest/observability/metrics.html`
- PD 训练流程: `/workspace/vime_pd_mooncake_migration_guide.md`

## 贡献

脚本基于 `/home/l00830933/workspace/vllm_metrics/` 中的参考实现,已集成到 vime 训练框架。

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
