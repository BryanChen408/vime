#!/usr/bin/env python3
# 极简节点指标 exporter(纯标准库,零依赖)。
# 用途:141 的 rl_dashboard(6007)跨机拉取本节点(如 140)的 NPU/CPU/内存。
#
# 用法(在被监控的机器上跑,如 140):
#   setsid nohup python3 tools/node_exporter.py --port 6010 >/tmp/node_exporter.log 2>&1 &
# 面板侧(141)默认拉 http://80.5.25.140:6010/node_metrics,可用 PEER_METRICS_URL 改。
import argparse
import json
import os
import re
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


NPU_CACHE_PATH = os.environ.get("NPU_METRICS_CACHE", "/tmp/vime_npu_metrics.json")
NPU_CACHE_MAX_AGE = float(os.environ.get("NPU_METRICS_MAX_AGE", "180"))


def npu_info():
    """Read the sampler's atomic local cache; never invoke npu-smi here."""
    now = time.time()
    try:
        with open(NPU_CACHE_PATH, encoding="utf-8") as cache_file:
            cached = json.load(cache_file)
        cards = cached.get("npu", [])
        if not isinstance(cards, list):
            raise ValueError("npu cache field is not a list")
        last_success = float(cached.get("last_success_ts") or 0.0)
        age = round(max(0.0, now - last_success), 1) if last_success else None
        status = {
            "status": cached.get("status", "unknown"),
            "ok": bool(cached.get("ok")),
            "stale": (not cached.get("ok")) or age is None or age > NPU_CACHE_MAX_AGE,
            "age": age,
            "error": cached.get("error"),
            "last_success_ts": last_success or None,
            "collector_pid": cached.get("collector_pid"),
            "child_pid": cached.get("child_pid"),
        }
        return cards, status
    except Exception as exc:
        return [], {"status": "cache_unavailable", "ok": False, "stale": True,
                    "age": None, "error": type(exc).__name__}


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


# ─── vllm 引擎指标(TTFT/TPOT/队列),2026-08-12 新增 ───
import urllib.request

_ENG_CACHE = {"ts": 0.0, "ports": []}
_ENG_SNAPSHOT = {"ts": 0.0, "data": [], "refreshing": False}
_ENG_LOCK = threading.Lock()
# 集群内网必须绕过 http_proxy(代理网关会劫持→连接失败)
_NOPROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _local_ip():
    """本机主网卡 IP(引擎绑的是节点 IP 而非 127.0.0.1,别用 loopback 探)。

    [2026-08-14] 优先取 env。原先只有下面的 UDP-connect 探法,且目标写死旧集群的
    80.5.25.141:换机后该地址无路由 → connect 抛异常 → 回落 127.0.0.1 →
    _discover_engine_ports 扫 loopback → 引擎绑的是节点 IP,一个都扫不到,
    面板的 engines 恒为空。目标地址只用于让内核选出口网卡,不发包。
    """
    for key in ("NODE_EXPORTER_HOST_IP", "VLLM_HOST_IP", "VIME_HOST_IP", "CURRENT_IP"):
        ip = (os.environ.get(key) or "").strip()
        if ip:
            return ip
    for target in (os.environ.get("MASTER_ADDR", "").strip(), "8.8.8.8"):
        if not target:
            continue
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((target, 80))
            ip = s.getsockname()[0]
            s.close()
            if ip and not ip.startswith("127."):
                return ip
        except Exception:
            continue
    return "127.0.0.1"


def _discover_engine_ports():
    """扫 15000-15220,找 /metrics 里有 vllm: 前缀的端口。60s 缓存。"""
    now = time.time()
    if now - _ENG_CACHE["ts"] < 60:
        return _ENG_CACHE["ports"]
    host = _local_ip()
    cands = []
    for p in range(15000, 15221):
        try:
            s = socket.create_connection((host, p), timeout=0.15)
            s.close()
            cands.append(p)
        except Exception:
            pass
    ports = []
    for p in cands:
        try:
            body = _NOPROXY.open(f"http://{host}:{p}/metrics", timeout=2).read().decode(errors="ignore")
            if "vllm:" in body:
                ports.append(p)
        except Exception:
            pass
    _ENG_CACHE.update(ts=now, ports=ports)
    return ports


def _hist_stats(body, name):
    """从 prometheus histogram 提取 mean/p50/p90。"""
    msum = re.search(re.escape(name) + r'_sum[^ ]* ([\d.eE+-]+)', body)
    mcnt = re.search(re.escape(name) + r'_count[^ ]* ([\d.eE+-]+)', body)
    if not msum or not mcnt or float(mcnt.group(1)) <= 0:
        return None
    total, mean = float(mcnt.group(1)), float(msum.group(1)) / float(mcnt.group(1))
    buckets = []
    for m in re.finditer(re.escape(name) + r'_bucket\{[^}]*le="([\d.eE+-]+)"[^}]*\} ([\d.eE+-]+)', body):
        buckets.append((float(m.group(1)), float(m.group(2))))
    buckets.sort()

    def q(x):
        for le, cum in buckets:
            if cum >= total * x:
                return le
        return buckets[-1][0] if buckets else 0.0

    return {"mean": round(mean, 3), "p50": q(0.5), "p90": q(0.9), "n": int(total)}


def _gauge(body, name):
    m = re.search(re.escape(name) + r'(?:\{[^}]*\})? ([\d.eE+-]+)', body)
    return int(float(m.group(1))) if m else 0


def _gauge_float(body, name):
    m = re.search(re.escape(name) + r'(?:\{[^}]*\})? ([\d.eE+-]+)', body)
    return round(float(m.group(1)), 4) if m else None


def _ratio(body, num, den):
    """hits/queries 类比率;无数据返回 None。"""
    n = re.search(re.escape(num) + r'(?:\{[^}]*\})? ([\d.eE+-]+)', body)
    d = re.search(re.escape(den) + r'(?:\{[^}]*\})? ([\d.eE+-]+)', body)
    if not n or not d or float(d.group(1)) <= 0:
        return None
    return round(float(n.group(1)) / float(d.group(1)), 4)


def _collect_engine_metrics():
    """Collect vLLM metrics. This may scan ports and must stay off HTTP threads."""
    host = _local_ip()
    out = []
    for p in _discover_engine_ports():
        try:
            body = _NOPROXY.open(f"http://{host}:{p}/metrics", timeout=3).read().decode(errors="ignore")
        except Exception:
            continue
        out.append({
            "port": p,
            "ttft": _hist_stats(body, "vllm:time_to_first_token_seconds"),
            "tpot": _hist_stats(body, "vllm:request_time_per_output_token_seconds"),
            "queue_t": _hist_stats(body, "vllm:request_queue_time_seconds"),
            "e2e": _hist_stats(body, "vllm:e2e_request_latency_seconds"),
            "prefill_t": _hist_stats(body, "vllm:request_prefill_time_seconds"),
            "decode_t": _hist_stats(body, "vllm:request_decode_time_seconds"),
            "kv_usage": _gauge_float(body, "vllm:kv_cache_usage_perc"),
            "prefix_hit": _ratio(body, "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"),
            "running": _gauge(body, "vllm:num_requests_running"),
            "waiting": _gauge(body, "vllm:num_requests_waiting"),
        })
    return out


def _refresh_engine_metrics():
    try:
        data = _collect_engine_metrics()
        with _ENG_LOCK:
            _ENG_SNAPSHOT.update(ts=time.time(), data=data)
    finally:
        with _ENG_LOCK:
            _ENG_SNAPSHOT["refreshing"] = False


def engine_metrics():
    """Return cached engine metrics and start at most one background refresh."""
    start_refresh = False
    with _ENG_LOCK:
        if not _ENG_SNAPSHOT["refreshing"] and time.time() - _ENG_SNAPSHOT["ts"] >= 5:
            _ENG_SNAPSHOT["refreshing"] = True
            start_refresh = True
        data = list(_ENG_SNAPSHOT["data"])
    if start_refresh:
        threading.Thread(target=_refresh_engine_metrics, daemon=True).start()
    return data


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
            npu, npu_status = npu_info()
            body = json.dumps({
                "host": socket.gethostname(), "ts": time.time(),
                "npu": npu, "npu_status": npu_status,
                "cpu_pct": cpu_pct(), "mem": mem_info(),
                "engines": engine_metrics(),
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
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
    ap.add_argument("--npu-cache", default=NPU_CACHE_PATH)
    ap.add_argument("--npu-cache-max-age", type=float, default=NPU_CACHE_MAX_AGE)
    a = ap.parse_args()
    NPU_CACHE_PATH = a.npu_cache
    NPU_CACHE_MAX_AGE = a.npu_cache_max_age
    print(f"[node_exporter] serving http://0.0.0.0:{a.port}/node_metrics", flush=True)
    engine_metrics()  # warm asynchronously; the HTTP endpoint remains non-blocking
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
