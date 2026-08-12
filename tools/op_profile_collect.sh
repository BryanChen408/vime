#!/usr/bin/env bash
# 昇腾算子级 profiling —— 采集 + 解析(Ascend PyTorch Profiler)。
#
# 前提:引擎已用 PROFILE_OP=1 启动(见 scripts/start-ascendc.sh 注释),
#       即 vllm serve 带了 --profiler-config,/start_profile 路由已注册。
#
# 与 tools/profile_rollout.py 的区别:那份的 payload(num_steps/output_dir)是 SGLang 形状,
#   vLLM 的 /start_profile 零参调用、全忽略;配置只在启动时由 --profiler-config 决定。
#   本脚本只做 vLLM 真正支持的动作:start / stop / 解析。
#
# 用法:
#   bash tools/op_profile_collect.sh start [PORTS...]     # 对指定引擎开始采(默认 15000)
#   bash tools/op_profile_collect.sh stop  [PORTS...]     # 提前停(不设也会 max_iterations 到步自动停)
#   bash tools/op_profile_collect.sh analyse <PROFILE_DIR> # 把 *_ascend_pt/ 解析成 CSV
#   bash tools/op_profile_collect.sh check [PORTS...]     # 探测 /start_profile 是否注册(200/404)
set -euo pipefail

ACTION="${1:-}"; shift || true
HOST="${PROFILE_HOST:-127.0.0.1}"

_default_ports() { echo "${@:-15000}"; }

case "${ACTION}" in
  check)
    for p in $(_default_ports "$@"); do
      code=$(curl -s --noproxy "*" -o /dev/null -w '%{http_code}' -X POST "http://${HOST}:${p}/start_profile" || echo 000)
      # 立刻停掉(check 只探路由,不真采)
      [ "${code}" = "200" ] && curl -s --noproxy "*" -o /dev/null -X POST "http://${HOST}:${p}/stop_profile" || true
      case "${code}" in
        200) echo "[check] :${p} 路由已注册(200) — 引擎带了 --profiler-config" ;;
        404) echo "[check] :${p} 未注册(404) — 引擎启动时没带 PROFILE_OP=1,需重启" ;;
        000) echo "[check] :${p} 连不上 — 引擎没起或端口不对" ;;
        *)   echo "[check] :${p} 返回 ${code}" ;;
      esac
    done
    ;;
  start)
    for p in $(_default_ports "$@"); do
      code=$(curl -s --noproxy "*" -o /dev/null -w '%{http_code}' -X POST "http://${HOST}:${p}/start_profile" || echo 000)
      echo "[start] :${p} -> ${code}"
      [ "${code}" = "404" ] && echo "        (404=引擎没带 profiler 配置,用 PROFILE_OP=1 重启)" >&2
    done
    echo "已开始采集。到 max_iterations 步各 worker 自动落盘;或运行 stop 提前结束。"
    ;;
  stop)
    for p in $(_default_ports "$@"); do
      code=$(curl -s --noproxy "*" -o /dev/null -w '%{http_code}' -X POST "http://${HOST}:${p}/stop_profile" || echo 000)
      echo "[stop] :${p} -> ${code}(落盘可能耗时数分钟,勿中断)"
    done
    ;;
  analyse)
    DIR="${1:?用法: analyse <PROFILE_DIR>(含 *_ascend_pt/ 的目录)}"
    python3 - "$DIR" <<'PY'
import glob, os, sys
d = sys.argv[1]
pts = sorted(glob.glob(os.path.join(d, "*_ascend_pt")) + glob.glob(os.path.join(d, "**", "*_ascend_pt"), recursive=True))
if not pts:
    sys.exit(f"[analyse] {d} 下没有 *_ascend_pt/ 目录(采集是否落盘?)")
from torch_npu.profiler.profiler import analyse
for p in pts:
    print(f"[analyse] {p}")
    analyse(p)  # 生成 ASCEND_PROFILER_OUTPUT/{op_statistic,kernel_details,step_trace_time}.csv + trace_view.json
    out = os.path.join(p, "ASCEND_PROFILER_OUTPUT")
    if os.path.isdir(out):
        for f in sorted(os.listdir(out)):
            print("   ", os.path.join(out, f))
print("完成。op_statistic.csv=算子占比;trace_view.json 可拖进 https://ui.perfetto.dev/ 或 MindStudio Insight。")
PY
    ;;
  *)
    grep '^#' "$0" | sed 's/^# \?//'
    exit 1 ;;
esac
