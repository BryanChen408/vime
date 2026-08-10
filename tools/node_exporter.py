#!/usr/bin/env python3
# 极简节点指标 exporter(纯标准库,零依赖)。
# 用途:141 的 rl_dashboard(6007)跨机拉取本节点(如 140)的 NPU/CPU/内存。
#
# 用法(在被监控的机器上跑,如 140):
#   setsid nohup python3 tools/node_exporter.py --port 6010 >/tmp/node_exporter.log 2>&1 &
# 面板侧(141)默认拉 http://80.5.25.140:6010/node_metrics,可用 PEER_METRICS_URL 改。
import argparse
import json
import re
import socket
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def npu_info():
    """A3 兼容解析(8 模组 × 2 die = 16 逻辑卡),与 rl_dashboard.npu_info 同步。"""
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True,
                             text=True, timeout=5).stdout
    except Exception:
        return []
    cards, module_temp = {}, {}
    r1 = re.compile(r"\|\s*(\d+)\s+\S+\s*\|\s*\w+\s*\|\s*(?:[\d.]+|-)\s+(\d+)\s")
    r2 = re.compile(r"\|\s*(\d+)(?:\s+(\d+))?\s*\|\s*[\w:.]+\s*\|"
                    r"\s*([\d.]+)\s+[\d.]+\s*/\s*[\d.]+\s+(\d+)\s*/\s*(\d+)")
    for line in out.splitlines():
        m = r1.search(line)
        if m:
            module_temp[int(m.group(1))] = int(m.group(2))
            continue
        m = r2.search(line)
        if m:
            cid = int(m.group(2)) if m.group(2) is not None else int(m.group(1))
            cards[cid] = {"id": cid, "power": 0.0,
                          "temp": module_temp.get(cid // 2, 0),
                          "aicore": int(float(m.group(3))),
                          "hbm_used": int(m.group(4)),
                          "hbm_total": int(m.group(5))}
    return [cards[i] for i in sorted(cards)]


def mem_info():
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                p = line.split()
                if p and p[0].rstrip(":") in ("MemTotal", "MemAvailable"):
                    info[p[0].rstrip(":")] = int(p[1])
    except Exception:
        return {}
    total = info.get("MemTotal", 0) / 1048576
    avail = info.get("MemAvailable", 0) / 1048576
    return {"total_gb": round(total, 1), "used_gb": round(total - avail, 1),
            "avail_gb": round(avail, 1)}


_prev_cpu = None


def cpu_pct():
    """两次抓取间 /proc/stat 差值;首次返回 None。"""
    global _prev_cpu
    try:
        with open("/proc/stat") as f:
            vals = list(map(int, f.readline().split()[1:]))
        idle, total = vals[3] + vals[4], sum(vals)
    except Exception:
        return None
    if _prev_cpu is None:
        _prev_cpu = (idle, total)
        return None
    d_idle, d_total = idle - _prev_cpu[0], total - _prev_cpu[1]
    _prev_cpu = (idle, total)
    return round(100 * (1 - d_idle / d_total), 1) if d_total > 0 else None


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/node_metrics"):
            body = json.dumps({
                "host": socket.gethostname(), "ts": time.time(),
                "npu": npu_info(), "cpu_pct": cpu_pct(), "mem": mem_info(),
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/health"):
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6010)
    a = ap.parse_args()
    print(f"[node_exporter] serving http://0.0.0.0:{a.port}/node_metrics", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
