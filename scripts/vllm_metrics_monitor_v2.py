#!/usr/bin/env python3
"""vLLM Rollout Metrics Monitor — 兼容 vLLM 0.23+ 的累计计数器语义。

要点（源自实跑日志分析）：

* **N engine，不是固定 2 个。** rollout 可能是 4 engine(tp4) 或 8 engine(tp2)。
  只聚合 2 个会漏掉大部分负载，且看不出跨 engine 不均。

* **engine 端口运行时分配。** vime 在 ``rollout.py:165`` 从 ``base_port=15000``
  起逐个探测空闲端口，端口既不固定也不连续（实测 tp4: 15000/15002/15004/15035；
  tp2: 15000/15002/15008/15039/15070/15101...），训练启动前不可知。
  因此默认**持续扫描**端口区间，引擎起来后自动纳入、消失后自动标记 down。

* **吞吐必须用 delta。** 0.23+ 的 ``vllm:generation_tokens_total`` 是单调累计
  计数器，直接读快照是没有意义的。engine 重启导致计数器回退时钳到 0。

* **per-engine 的 running/waiting 是关键诊断信号。** 实跑对比中，区分
  "KV 卡住准入"和"负载不足"靠的就是 ``waiting>0 且 running<=1`` 的占比
  （tp2 34% vs tp4 0%）。面板直接算出这个比例。
"""

import argparse
import threading
import time
from collections import deque
from datetime import datetime

import requests
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

HISTORY_LEN = 300

_lock = threading.Lock()

# engine_key -> {url, alive, last_seen, tokens, throughput, running, waiting, ...}
engines: dict[str, dict] = {}
# engine_key -> deque[{throughput, running, waiting}]
engine_history: dict[str, deque] = {}

combined_history: deque = deque(maxlen=HISTORY_LEN)
timestamps: deque = deque(maxlen=HISTORY_LEN)

# waiting>0 & running<=1 的采样计数 → KV 准入受限的直接证据
stall_stats = {"samples": 0, "starved": 0}

config = {
    "engine_urls": [],
    "discover_host": None,
    "discover_ports": (15000, 15200),
    "interval": 5,
    "discover": True,
}


def parse_prometheus_metrics(text: str) -> dict:
    """解析 Prometheus 文本格式。同名多 label 的 series 求和（如多 DP rank）。"""
    metrics: dict[str, float] = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            if "{" in line:
                name = line.split("{")[0]
                rest = line.split("}", 1)[1].strip()
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                name, rest = parts[0], parts[1]
            value = float(rest.split()[0])
        except (ValueError, IndexError):
            continue
        metrics[name] = metrics.get(name, 0.0) + value
    return metrics


def fetch_metrics(url: str, timeout: float = 2.0) -> dict | None:
    """返回 None 表示端点不可达（区别于「可达但全零」）。"""
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return parse_prometheus_metrics(resp.text)
    except Exception:
        pass
    return None


def probe_engines() -> list[str]:
    """扫描端口区间，返回暴露 vLLM metrics 的 URL 列表。

    判据是 payload 里出现 ``vllm:`` 前缀 —— 端口区间内可能混有 Ray/router
    等其他 HTTP 服务，靠 200 状态码无法区分。
    """
    host = config["discover_host"]
    lo, hi = config["discover_ports"]
    found = []
    for port in range(lo, hi + 1):
        url = f"http://{host}:{port}/metrics"
        try:
            resp = requests.get(url, timeout=0.25)
        except Exception:
            continue
        if resp.status_code == 200 and "vllm:" in resp.text:
            found.append(url)
    return found


def _blank_engine(url: str) -> dict:
    return {
        "url": url,
        "alive": False,
        "throughput": 0.0,
        "running": 0,
        "waiting": 0,
        "generation_tokens_total": 0.0,
        "prompt_tokens_total": 0.0,
        "_prev_tokens": None,  # None = 尚未建立基线，首轮不产出吞吐
        "_prev_time": None,
    }


def update_engine(key: str, raw: dict | None) -> dict:
    eng = engines[key]

    if raw is None:
        eng["alive"] = False
        eng["throughput"] = 0.0
        eng["running"] = 0
        eng["waiting"] = 0
        # 保留 _prev_tokens：短暂抓取失败后恢复，仍能按真实时间跨度算 delta
        return eng

    gen_total = raw.get("vllm:generation_tokens_total", 0.0)
    now = time.time()

    prev_tokens, prev_time = eng["_prev_tokens"], eng["_prev_time"]
    if prev_tokens is None or prev_time is None:
        throughput = 0.0  # 首轮只建基线
    else:
        elapsed = now - prev_time
        delta = gen_total - prev_tokens
        if delta < 0:
            delta = 0.0  # engine 重启 → 计数器回退，钳到 0
        throughput = delta / elapsed if elapsed > 0 else 0.0

    eng.update(
        alive=True,
        throughput=throughput,
        running=int(raw.get("vllm:num_requests_running", 0)),
        waiting=int(raw.get("vllm:num_requests_waiting", 0)),
        generation_tokens_total=gen_total,
        prompt_tokens_total=raw.get("vllm:prompt_tokens_total", 0.0),
        _prev_tokens=gen_total,
        _prev_time=now,
    )
    return eng


def collector():
    print(f"[metrics] collector 启动, interval={config['interval']}s")
    last_probe = 0.0

    while True:
        try:
            # 周期重扫：engine 是训练启动后才陆续起来的
            if config["discover"] and time.time() - last_probe > 30:
                for url in probe_engines():
                    key = url.split("//")[1].split("/")[0]  # host:port
                    with _lock:
                        if key not in engines:
                            engines[key] = _blank_engine(url)
                            engine_history[key] = deque(maxlen=HISTORY_LEN)
                            print(f"[metrics] 发现 engine: {url}")
                last_probe = time.time()

            with _lock:
                targets = [(k, e["url"]) for k, e in engines.items()]

            fetched = {key: fetch_metrics(url) for key, url in targets}

            now = datetime.now()
            with _lock:
                for key, raw in fetched.items():
                    if key not in engines:
                        continue
                    eng = update_engine(key, raw)
                    engine_history[key].append(
                        {
                            "throughput": eng["throughput"],
                            "running": eng["running"],
                            "waiting": eng["waiting"],
                        }
                    )

                live = [e for e in engines.values() if e["alive"]]
                total_tp = sum(e["throughput"] for e in live)
                total_run = sum(e["running"] for e in live)
                total_wait = sum(e["waiting"] for e in live)

                # KV 准入受限判据：有请求排队，但引擎并发压不上去
                for e in live:
                    stall_stats["samples"] += 1
                    if e["waiting"] > 0 and e["running"] <= 1:
                        stall_stats["starved"] += 1

                timestamps.append(now.strftime("%H:%M:%S"))
                combined_history.append(
                    {
                        "throughput": total_tp,
                        "running": total_run,
                        "waiting": total_wait,
                        "engines_alive": len(live),
                    }
                )

            n_live = len(live)
            per_eng = total_tp / n_live if n_live else 0.0
            print(
                f"[{now.strftime('%H:%M:%S')}] engines={n_live} "
                f"total={total_tp:.1f} tok/s (avg {per_eng:.1f}/engine) "
                f"running={total_run} waiting={total_wait}"
            )

        except Exception as exc:  # 采集线程绝不能死
            print(f"[metrics][ERROR] collector: {exc}")

        time.sleep(config["interval"])


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>vLLM Rollout Metrics</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family:'Segoe UI',sans-serif; background:#0d1117; color:#c9d1d9; padding:20px; }
        .container { max-width:1500px; margin:0 auto; }
        h1 { font-size:26px; margin-bottom:6px; color:#58a6ff; }
        .subtitle { color:#8b949e; margin-bottom:20px; font-size:13px; }
        .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:14px; margin-bottom:26px; }
        .card { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:18px; }
        .card h3 { font-size:12px; color:#8b949e; margin-bottom:8px; text-transform:uppercase; letter-spacing:.4px; }
        .val { font-size:30px; font-weight:700; color:#58a6ff; }
        .unit { font-size:12px; color:#8b949e; margin-top:3px; }
        .warn .val { color:#f0883e; }
        .bad .val { color:#f85149; }
        .chart { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:18px; margin-bottom:18px; }
        .chart h2 { font-size:15px; margin-bottom:12px; color:#c9d1d9; }
        table { width:100%; border-collapse:collapse; font-size:13px; }
        th,td { text-align:right; padding:8px 10px; border-bottom:1px solid #21262d; }
        th:first-child, td:first-child { text-align:left; }
        th { color:#8b949e; font-weight:600; font-size:11px; text-transform:uppercase; }
        .dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:7px; }
        .up { background:#3fb950; } .down { background:#f85149; }
        .starved td { background:rgba(248,81,73,.08); }
        .hint { color:#8b949e; font-size:12px; margin-top:10px; line-height:1.6; }
        .last { text-align:right; color:#8b949e; font-size:12px; margin-bottom:16px; }
    </style>
</head>
<body>
<div class="container">
    <h1>vLLM Rollout Metrics</h1>
    <p class="subtitle">N-engine 聚合 · 运行时端口发现 · delta 吞吐 (vLLM 0.23+)</p>
    <div class="last">Last update: <span id="lastUpdate">--</span></div>

    <div class="grid">
        <div class="card"><h3>Total Throughput</h3><div class="val" id="tp">--</div><div class="unit">tokens/sec (all engines)</div></div>
        <div class="card"><h3>Engines Alive</h3><div class="val" id="alive">--</div><div class="unit">discovered endpoints</div></div>
        <div class="card"><h3>Running</h3><div class="val" id="run">--</div><div class="unit">requests</div></div>
        <div class="card"><h3>Waiting</h3><div class="val" id="wait">--</div><div class="unit">requests queued</div></div>
        <div class="card" id="starveCard"><h3>KV-Starved Samples</h3><div class="val" id="starve">--</div><div class="unit">waiting&gt;0 且 running&le;1</div></div>
        <div class="card"><h3>Avg / Engine</h3><div class="val" id="perEng">--</div><div class="unit">tokens/sec</div></div>
    </div>

    <div class="chart"><h2>Total Generation Throughput (tok/s)</h2><canvas id="tpChart" height="70"></canvas></div>
    <div class="chart"><h2>Running vs Waiting Requests</h2><canvas id="reqChart" height="70"></canvas></div>

    <div class="chart">
        <h2>Per-Engine Breakdown</h2>
        <table>
            <thead><tr><th>Engine</th><th>Throughput</th><th>Running</th><th>Waiting</th><th>Gen tokens</th></tr></thead>
            <tbody id="engRows"></tbody>
        </table>
        <p class="hint">
            <b>KV-Starved</b>：<code>waiting&gt;0</code> 但 <code>running&le;1</code> 的采样占比。偏高说明请求在排队而引擎并发压不上去,
            通常是 KV cache 不足导致准入受限（而非负载不足）。实测参照：tp4/4engine ≈ 0%,tp2/8engine ≈ 34%。<br>
            engine 端口由 vime 运行时分配（rollout.py:165 从 15000 起探测空闲端口）,故每 30s 重扫一次,新引擎自动纳入。
        </p>
    </div>
</div>
<script>
const base = { type:'line', options:{ responsive:true, maintainAspectRatio:true,
    interaction:{mode:'index',intersect:false},
    plugins:{legend:{labels:{color:'#c9d1d9',boxWidth:12}}},
    scales:{x:{ticks:{color:'#8b949e',maxTicksLimit:12},grid:{color:'#21262d'}},
            y:{ticks:{color:'#8b949e'},grid:{color:'#21262d'},beginAtZero:true}} } };

const tpChart = new Chart(document.getElementById('tpChart'), {...base, data:{labels:[],datasets:[
    {label:'Total tok/s',data:[],borderColor:'#238636',backgroundColor:'rgba(35,134,54,.15)',fill:true,tension:.35,borderWidth:2,pointRadius:0}]}});

const reqChart = new Chart(document.getElementById('reqChart'), {...base, data:{labels:[],datasets:[
    {label:'Running',data:[],borderColor:'#1f6feb',backgroundColor:'rgba(31,111,235,.18)',fill:true,tension:.35,pointRadius:0},
    {label:'Waiting',data:[],borderColor:'#f85149',backgroundColor:'rgba(248,81,73,.18)',fill:true,tension:.35,pointRadius:0}]}});

function fmt(n,d=1){ return (n===null||n===undefined)?'--':Number(n).toFixed(d); }

function update(){
  fetch('/api/metrics').then(r=>r.json()).then(d=>{
    document.getElementById('lastUpdate').textContent = d.last_update || '--';
    document.getElementById('tp').textContent      = fmt(d.combined.throughput);
    document.getElementById('alive').textContent   = d.engines_alive + '/' + d.engines_total;
    document.getElementById('run').textContent     = d.combined.running;
    document.getElementById('wait').textContent    = d.combined.waiting;
    document.getElementById('perEng').textContent  = fmt(d.avg_per_engine);

    const pct = d.starved_pct;
    document.getElementById('starve').textContent = fmt(pct,0) + '%';
    const c = document.getElementById('starveCard');
    c.className = 'card' + (pct >= 20 ? ' bad' : (pct >= 5 ? ' warn' : ''));

    const rows = d.engines.map(e => `<tr class="${e.waiting>0&&e.running<=1?'starved':''}">
        <td><span class="dot ${e.alive?'up':'down'}"></span>${e.key}</td>
        <td>${fmt(e.throughput)}</td><td>${e.running}</td><td>${e.waiting}</td>
        <td>${Number(e.generation_tokens_total).toLocaleString()}</td></tr>`).join('');
    document.getElementById('engRows').innerHTML = rows ||
        '<tr><td colspan="5" style="text-align:center;color:#8b949e">尚未发现 engine — 等待 vLLM 启动…</td></tr>';

    if(d.history){
      tpChart.data.labels = d.history.timestamps;
      tpChart.data.datasets[0].data = d.history.combined.map(x=>x.throughput);
      tpChart.update('none');
      reqChart.data.labels = d.history.timestamps;
      reqChart.data.datasets[0].data = d.history.combined.map(x=>x.running);
      reqChart.data.datasets[1].data = d.history.combined.map(x=>x.waiting);
      reqChart.update('none');
    }
  }).catch(e=>console.error('fetch failed',e));
}
setInterval(update, 2000); update();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/metrics")
def api_metrics():
    with _lock:
        eng_list = [
            {
                "key": key,
                "url": e["url"],
                "alive": e["alive"],
                "throughput": e["throughput"],
                "running": e["running"],
                "waiting": e["waiting"],
                "generation_tokens_total": e["generation_tokens_total"],
            }
            for key, e in sorted(engines.items())
        ]
        live = [e for e in eng_list if e["alive"]]
        latest = combined_history[-1] if combined_history else {
            "throughput": 0.0, "running": 0, "waiting": 0
        }
        samples = stall_stats["samples"]
        starved_pct = 100.0 * stall_stats["starved"] / samples if samples else 0.0
        payload = {
            "last_update": timestamps[-1] if timestamps else None,
            "engines": eng_list,
            "engines_total": len(eng_list),
            "engines_alive": len(live),
            "combined": {
                "throughput": latest["throughput"],
                "running": latest["running"],
                "waiting": latest["waiting"],
            },
            "avg_per_engine": (latest["throughput"] / len(live)) if live else 0.0,
            "starved_pct": starved_pct,
            "starved_samples": stall_stats["starved"],
            "total_samples": samples,
            "history": {
                "timestamps": list(timestamps),
                "combined": list(combined_history),
                "per_engine": {k: list(v) for k, v in engine_history.items()},
            },
        }
    return jsonify(payload)


def main():
    p = argparse.ArgumentParser(description="vLLM Rollout Metrics Monitor (N-engine)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument(
        "--engines",
        default="",
        help="逗号分隔的 engine /metrics URL；给定则关闭自动发现",
    )
    p.add_argument("--discover-host", default="127.0.0.1")
    p.add_argument(
        "--discover-ports",
        default="15000-15200",
        help="自动发现的端口区间，默认 15000-15200（vime base_port=15000）",
    )
    p.add_argument("--interval", type=int, default=5)
    args = p.parse_args()

    config["interval"] = args.interval

    if args.engines.strip():
        config["discover"] = False
        for raw in args.engines.split(","):
            url = raw.strip()
            if not url:
                continue
            if not url.endswith("/metrics"):
                url = url.rstrip("/") + "/metrics"
            key = url.split("//")[1].split("/")[0]
            engines[key] = _blank_engine(url)
            engine_history[key] = deque(maxlen=HISTORY_LEN)
    else:
        config["discover"] = True
        config["discover_host"] = args.discover_host
        lo, _, hi = args.discover_ports.partition("-")
        config["discover_ports"] = (int(lo), int(hi or lo))

    threading.Thread(target=collector, daemon=True).start()

    mode = (
        f"explicit ({len(engines)} engines)"
        if not config["discover"]
        else f"auto-discover {config['discover_host']}:{args.discover_ports}"
    )
    print("=" * 64)
    print("vLLM Rollout Metrics Monitor — N-engine, vLLM 0.23+")
    print("=" * 64)
    print(f"Web UI  : http://{args.host}:{args.port}")
    print(f"Engines : {mode}")
    print(f"Interval: {args.interval}s")
    print("=" * 64)

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
