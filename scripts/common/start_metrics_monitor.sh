#!/bin/bash
# 自动启动 vLLM metrics 监控面板（旁路能力：任何失败都不得影响训练）。
#
# 用法（env 驱动，source 进训练脚本）：
#   METRICS_DASHBOARD_PORT=5000 \
#   METRICS_HOST_IP=80.48.5.56 \
#   source scripts/common/start_metrics_monitor.sh
#
# 可选：
#   METRICS_ENGINE_URLS  显式指定 engine /metrics URL(逗号分隔)，给定后跳过自动发现
#   METRICS_SCAN_PORTS   自动发现的端口扫描范围，默认 15000-15200
#                        (vime 在 rollout.py:165 从 base_port=15000 起逐个探测空闲端口分配,
#                         端口既不固定也不连续，且训练启动前不可知 → 只能运行时发现)
#   METRICS_SKIP_PORTS   扫描时跳过的端口(逗号分隔,支持区间 15002-15005)。
#                        扫描区间和 Mooncake KV 握手端口重叠时用它显式排除
#   METRICS_ENGINE_INTERNAL_PORTS
#                        每个 engine 在 API port 之后占用的内部端口数 = 1(nccl)+tp(bootstrap)。
#                        PD+Mooncake 场景建议设为 $((1 + TP))(如 tp4 → 5),
#                        发现 engine 后自动隔离该窗口内的非 HTTP 端口,避免刷
#                        readString/SocketHandShakePlugin 报错
#   METRICS_INTERVAL     采样间隔秒，默认 5
#
# 设计约束：本脚本被 `set -ex` 的训练脚本 source。绝不能让任何非零返回码
# 经 set -e 冒泡杀死训练进程 —— 早期版本 `VAR=$(pgrep ...)` 在"无已运行实例"
# 这一正常路径上返回 1，直接终止了整个训练。故全程 set +e，末尾恢复调用方 flags。

# ── 保存并关闭调用方的 -e/-x，避免旁路逻辑污染训练日志或杀死训练 ──
__mm_flags=""
case "$-" in *e*) __mm_flags="${__mm_flags}e" ;; esac
case "$-" in *x*) __mm_flags="${__mm_flags}x" ;; esac
set +ex

__mm_finish() {
   # 恢复调用方 shell flags；无论如何返回 0。
   [ -n "${__mm_flags}" ] && set -"${__mm_flags}"
   unset __mm_flags
   return 0
}

__mm_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
__mm_vime_root="$(cd -- "${__mm_script_dir}/../.." &>/dev/null && pwd)"
__mm_monitor="${__mm_vime_root}/scripts/vllm_metrics_monitor_v2.py"

__mm_dash_port="${METRICS_DASHBOARD_PORT:-5000}"
__mm_host_ip="${METRICS_HOST_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
__mm_scan="${METRICS_SCAN_PORTS:-15000-15200}"
__mm_skip="${METRICS_SKIP_PORTS:-}"
__mm_internal="${METRICS_ENGINE_INTERNAL_PORTS:-0}"
__mm_interval="${METRICS_INTERVAL:-5}"

if [ ! -f "${__mm_monitor}" ]; then
   echo "[metrics][WARN] monitor script not found: ${__mm_monitor} — 跳过监控,训练继续" >&2
   __mm_finish
   return 0 2>/dev/null || true
fi

# ── 依赖检查(缺失只警告,绝不 fail) ──
if ! python3 -c "import flask, requests" >/dev/null 2>&1; then
   echo "[metrics] installing flask/requests ..."
   pip install -q flask requests >/dev/null 2>&1
   if ! python3 -c "import flask, requests" >/dev/null 2>&1; then
      echo "[metrics][WARN] flask/requests 不可用 — 跳过监控,训练继续" >&2
      __mm_finish
      return 0 2>/dev/null || true
   fi
fi

# ── 已有实例检测。注意 `|| true`:pgrep 无匹配返回 1,是正常路径 ──
__mm_existing="$(pgrep -f "vllm_metrics_monitor_v2\.py.*--port[= ]${__mm_dash_port}" 2>/dev/null || true)"
if [ -n "${__mm_existing}" ]; then
   echo "[metrics] 已在运行 (PID: ${__mm_existing}) → http://${__mm_host_ip}:${__mm_dash_port}"
   __mm_finish
   return 0 2>/dev/null || true
fi

__mm_log_dir="${__mm_vime_root}/logs/metrics"
mkdir -p "${__mm_log_dir}" 2>/dev/null || true
__mm_log="${__mm_log_dir}/vllm_metrics_$(date +%Y%m%d_%H%M%S).log"

__mm_args=(
   --host 0.0.0.0
   --port "${__mm_dash_port}"
   --interval "${__mm_interval}"
)
if [ -n "${METRICS_ENGINE_URLS:-}" ]; then
   __mm_args+=(--engines "${METRICS_ENGINE_URLS}")
   echo "[metrics] engine 端点(显式): ${METRICS_ENGINE_URLS}"
else
   __mm_args+=(--discover-host "${__mm_host_ip}" --discover-ports "${__mm_scan}")
   [ -n "${__mm_skip}" ] && __mm_args+=(--skip-ports "${__mm_skip}")
   [ "${__mm_internal}" != "0" ] && __mm_args+=(--engine-internal-ports "${__mm_internal}")
   echo "[metrics] engine 端点: 自动发现 ${__mm_host_ip}:${__mm_scan}"
   echo "[metrics]   (engine 端口由 vime 运行时分配,启动前不可知 → 持续发现,引擎起来后自动纳入)"
   [ -n "${__mm_skip}" ] && echo "[metrics]   跳过端口: ${__mm_skip}"
   [ "${__mm_internal}" != "0" ] && \
      echo "[metrics]   engine 内部端口窗口: +1..+${__mm_internal} (nccl+mooncake bootstrap,自动隔离)"
fi

nohup python3 "${__mm_monitor}" "${__mm_args[@]}" >"${__mm_log}" 2>&1 &
__mm_pid=$!

sleep 2
if kill -0 "${__mm_pid}" 2>/dev/null; then
   echo "[metrics] ✓ 面板已启动 (PID ${__mm_pid}) → http://${__mm_host_ip}:${__mm_dash_port}"
   echo "[metrics]   日志: ${__mm_log}"
else
   echo "[metrics][WARN] 面板启动失败,见 ${__mm_log} — 训练继续" >&2
fi

unset __mm_script_dir __mm_vime_root __mm_monitor __mm_dash_port __mm_host_ip \
      __mm_scan __mm_skip __mm_internal __mm_interval __mm_existing __mm_log_dir \
      __mm_log __mm_args __mm_pid
__mm_finish
return 0 2>/dev/null || true
