#!/usr/bin/env python3
"""
RL training live dashboard (zero external deps, self-hosted).

Reads the LIVE slime training log + npu-smi + host mem + Polar session dir and
serves an auto-refreshing web page.

Panels:
  - Reward (reward_mean / reward_mean_completed / std band), success_rate
  - grad_norm, loss (pg_loss), entropy  (per train step)
  - context sequence length (response_len mean/max), rollout_time, staleness
  - NPU load (per-card AICore% + HBM used/total), host CPU memory
  - sglang prefill/decode (running-req, token usage, throughput, queue)
  - operator solve status (session COMPLETED/TIMEOUT/ERROR/running, current op, mean turns)

Usage:
    python3 tools/rl_dashboard.py                # port 6007, follow locally launched run
    PORT=6010 python3 tools/rl_dashboard.py
    python3 tools/rl_dashboard.py --log /path/to/train_*.log --polar /path/to/runs

Then open  http://<host>:6007/
"""
import os
import re
import glob
import json
import time
import ast
import subprocess
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# ----------------------------- config -----------------------------
# 本机训练启动器会把它实际使用的 LOG_FILE 原子发布到 LOG_POINTER。只要指针
# 存在，dashboard 就严格跟随该文件，不再从共享 /mnt 中按 mtime 误选其他机器的日志。
# 指针尚未生成时页面保持空白；只有显式把 RL_LOG_POINTER 设为空，才回退到
# 原来的目录自动发现行为。
LOG_POINTER = os.environ.get(
    "RL_LOG_POINTER", "/tmp/vime_rl_dashboard_train_log"
).strip()
# 回退日志目录:默认两个候选(pd 时代的 /home/docker/logs + 混合部署的
# /mnt/pipeline-data/train_log);可用 RL_LOG_DIRS="dir1:dir2" 覆盖。
LOG_DIRS = os.environ.get("RL_LOG_DIRS", "/home/docker/logs:/mnt/pipeline-data/train_log").split(":")
POLAR_RUNS = "/home/docker/polar_can/ProRL-Agent-Server/output/ascend_operator/runs"
PORT = int(os.environ.get("PORT", "6007"))
LOG_TAIL_INIT = 256 * 1024   # bytes shown on first paint (tail)
LOG_PAGE = 256 * 1024        # bytes fetched per scroll-up page

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PERF_RE = re.compile(r"rollout\.py:\d+ - perf (\d+): (\{.*\})\s*$")
STEP_RE = re.compile(r"model\.py:\d+ - step (\d+): (\{.*\})\s*$")
# per-rollout training-data summary; carries the FULL per-session lengths
DATA_RE = re.compile(r"data\.py:\d+ - rollout (\d+): (\{.*\})\s*$")
DATA_LEN_KEYS = ("rollout/total_lengths", "rollout/response_lengths")
DECODE_RE = re.compile(
    r"Decode batch, #running-req: (\d+),.*?full token usage: ([\d.]+).*?"
    r"gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)")
PREFILL_RE = re.compile(
    r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)")
TS_RE = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
TRAIN_PROCESS_PATTERN = r"(^|[ /])train(_async)?\.py( |$)"
# named phase timers: "Timer <name> start" / "Timer <name> end (elapsed: Ns)"
TIMER_RE = re.compile(
    r"timer\.py:\d+ - Timer (\w+) (start|end)(?:.*?elapsed:\s*([\d.]+))?")
# held-out eval line: "rollout.py:1288 - eval <rollout_id>: {'eval/<ds>': <reward>, ...}"
EVAL_RE = re.compile(r"rollout\.py:\d+ - eval (\d+): (\{.*\})\s*$")
EVAL_KV_RE = re.compile(r"'(eval/[^']+)':\s*([-\d.eE+]+)")
_NUM_ROLLOUT = None   # cached --num-rollout from the running process

# ----------------------------- state ------------------------------
STATE = {
    "log_path": None,
    "offset": 0,
    "rollouts": {},   # step -> perf metrics dict (per-generation-call lengths)
    "data": {},       # step -> per-session length summary (total/response)
    "train": {},      # step -> metrics dict
    "eval": {},       # rollout_id -> held-out eval/* metrics dict
    "sglang": {},     # latest decode/prefill snapshot
    "op_cache": {"ts": 0.0, "data": {}},
    "opsrun_cache": {"ts": 0.0, "data": {}},  # cumulative operators/samples run
    "solved_cache": {"ts": 0.0, "data": {}},  # solved (evaluation.success) vs total
    "tps_cache": {"ts": 0.0, "data": {}},  # per-session decode tok/s distribution
    "evalop_cache": {"ts": 0.0, "data": {}},  # per-op eval verification status (latest eval round)
    "oprounds_cache": {"ts": 0.0, "data": {}},  # per-op per-round solve rate (ops run >=2 rounds)
    "evalset_cache": {"ts": 0.0, "data": 0},  # total ops in the eval set (dynamic denominator)
    "health_cache": {"ts": 0.0, "data": {}},  # zero-variance groups / per-op EMA / failure taxonomy
    "epoch_cache": {"ts": 0.0, "data": 0},  # rollouts-per-epoch (ceil(train_set/rollout_batch))
    "trainset_cache": {"ts": 0.0, "data": 0},  # training-set size = --prompt-data line count
    "holdout_cache": {"ts": 0.0, "data": {}},  # holdout pass@1 vs step (offline ckpt eval jsonl)
    # current training phase: open named-timers + last-seen event timestamps
    "phase": {"open": {}, "sglang_ts": 0.0, "latest_ts": 0.0,
              "elapsed": {}, "round_ts": []},
    "host_peak": 0.0,
    # host RAM history: [(ts, used_gb), ...] sampled >=HOST_HIST_EVERY apart.
    # Purpose: a leak shows as the *troughs* creeping up run-over-run; a spike
    # (e.g. weight sync gathering full tensors) returns to the same baseline.
    "host_hist": [],
}
HOST_HIST_EVERY = 15.0      # seconds between samples
HOST_HIST_MAX = 1440        # keep ~6h at 15s


def _epoch_from(line):
    """Parse the leading [YYYY-MM-DD HH:MM:SS] to epoch (only used for diffs,
    so local-tz offset cancels out)."""
    m = TS_RE.search(line)
    if not m:
        return 0.0
    try:
        return time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return 0.0


def _pointed_log():
    """Return ``(claimed, path)`` for the local launcher log pointer.

    ``claimed`` stays true when the pointer exists but its target has not been
    created yet. In that short startup window we deliberately return no log
    instead of falling back to an unrelated, newer file in the shared mount.
    """
    if not LOG_POINTER:
        return False, None
    try:
        with open(LOG_POINTER, encoding="utf-8") as f:
            path = f.readline().strip()
    except FileNotFoundError:
        # A configured local-launch pointer is an ownership boundary, not merely
        # a preference. Before the first local run, show no log instead of a
        # stranger's file from the shared fallback directories.
        return True, None
    except OSError:
        return True, None
    if not path:
        return True, None
    path = os.path.abspath(os.path.expanduser(path))
    try:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return True, path
    except OSError:
        pass
    return True, None


def latest_log():
    claimed, path = _pointed_log()
    if claimed:
        return path
    files = []
    for d in LOG_DIRS:
        files.extend(glob.glob(os.path.join(d, "train_qwen36_polar_*.log")))
    files = [f for f in files if os.path.getsize(f) > 0]
    return max(files, key=os.path.getmtime) if files else None


def latest_polar_run():
    dirs = glob.glob(os.path.join(POLAR_RUNS, "polar_*"))
    dirs = [d for d in dirs if os.path.isdir(d)]
    return max(dirs, key=os.path.getmtime) if dirs else None


def _training_pids():
    try:
        out = subprocess.run(["pgrep", "-f", TRAIN_PROCESS_PATTERN],
                             capture_output=True, text=True, timeout=3)
        if out.returncode != 0:
            return []
        return [pid for pid in out.stdout.split() if pid.isdigit()]
    except Exception:
        return []


def train_alive():
    return bool(_training_pids())


def train_uptime():
    """Seconds since the active train.py/train_async.py process started.

    Uses the process elapsed time (ps etimes) so it reflects the *currently
    running* script, not a log timestamp. When several matches exist (ray
    workers also match the pattern) the oldest one is the launcher process.
    Returns None when nothing is running.
    """
    try:
        pids = _training_pids()
        if not pids:
            return None
        best = None
        for pid in pids:
            r = subprocess.run(["ps", "-o", "etimes=", "-p", pid],
                               capture_output=True, text=True, timeout=3)
            s = r.stdout.strip()
            if s.isdigit():
                v = int(s)
                if best is None or v > best:
                    best = v
        return best
    except Exception:
        return None


def eval_running():
    """True while the RolloutManager is executing eval(): the Ray worker's task
    name becomes 'ray::RolloutManager.eval' during held-out validation."""
    try:
        out = subprocess.run(["pgrep", "-f", "RolloutManager.eval"],
                             capture_output=True, text=True, timeout=3)
        return out.returncode == 0 and out.stdout.strip() != ""
    except Exception:
        return False


# --------------------- incremental log parse ----------------------
def update_log():
    path = latest_log()
    if path is None:
        return
    if path != STATE["log_path"]:
        # A new run: drop every per-run series. host_hist/host_peak included —
        # a curve carrying the dead run's history makes the trough comparison
        # (leak vs spike) read against the wrong baseline.
        STATE.update(log_path=path, offset=0, rollouts={}, data={}, train={},
                     eval={}, sglang={}, host_hist=[], host_peak=0.0,
                     integrity_rejects=0)
        STATE["phase"] = {"open": {}, "sglang_ts": 0.0, "latest_ts": 0.0,
                          "elapsed": {}, "round_ts": []}
    try:
        size = os.path.getsize(path)
        if size < STATE["offset"]:          # file rotated/truncated
            STATE["offset"] = 0
        with open(path, "rb") as f:
            f.seek(STATE["offset"])
            chunk = f.read()
        # only advance to the last full line
        last_nl = chunk.rfind(b"\n")
        if last_nl == -1:
            return
        STATE["offset"] += last_nl + 1
        text = chunk[:last_nl].decode("utf-8", errors="ignore")
    except OSError:
        return

    for raw in text.splitlines():
        line = ANSI.sub("", raw)
        # [CoT-watch] count logprob-integrity rejections (adapter guard fires
        # when a trace's rollout logprobs are misattributed/missing — the risk
        # surfaces once CoT reasoning is unmasked and its logprobs enter TIS).
        if "LOGPROB-INTEGRITY-REJECT" in line:
            STATE["integrity_rejects"] = STATE.get("integrity_rejects", 0) + 1
            continue
        m = PERF_RE.search(line)
        if m:
            try:
                d = ast.literal_eval(m.group(2))
                STATE["rollouts"][int(m.group(1))] = d
            except Exception:
                pass
            continue
        m = DATA_RE.search(line)
        if m:
            # targeted extraction (the full dict may hold nan/arrays that break
            # literal_eval); we only need the two per-session length means.
            step, body = int(m.group(1)), m.group(2)
            rec = STATE["data"].setdefault(step, {})
            for key in DATA_LEN_KEYS:
                km = re.search(r"'" + re.escape(key) + r"':\s*([-\d.eE+]+)", body)
                if km:
                    try:
                        rec[key] = float(km.group(1))
                    except ValueError:
                        pass
            ep = _epoch_from(line)
            if ep:
                rtlist = STATE["phase"].setdefault("round_ts", [])
                rtlist.append((step, ep))
                del rtlist[:-12]
                STATE["phase"]["latest_ts"] = max(STATE["phase"]["latest_ts"], ep)
            continue
        m = STEP_RE.search(line)
        if m:
            try:
                d = ast.literal_eval(m.group(2))
                STATE["train"][int(m.group(1))] = d
            except Exception:
                pass
            continue
        m = EVAL_RE.search(line)
        if m:
            # targeted float extraction (dict may hold nan/arrays that break
            # literal_eval); we only need the eval/* scalar metrics.
            rid, body = int(m.group(1)), m.group(2)
            rec = STATE["eval"].setdefault(rid, {})
            for km in EVAL_KV_RE.finditer(body):
                try:
                    rec[km.group(1)] = float(km.group(2))
                except ValueError:
                    pass
            continue
        m = DECODE_RE.search(line)
        if m:
            STATE["sglang"] = {
                "phase": "decode",
                "running_req": int(m.group(1)),
                "token_usage": float(m.group(2)),
                "throughput": float(m.group(3)),
                "queue_req": int(m.group(4)),
            }
            ep = _epoch_from(line)
            if ep:
                STATE["phase"]["sglang_ts"] = ep
                STATE["phase"]["latest_ts"] = max(STATE["phase"]["latest_ts"], ep)
            continue
        m = PREFILL_RE.search(line)
        if m:
            s = STATE.get("sglang", {})
            s.update(prefill_newseq=int(m.group(1)),
                     prefill_newtok=int(m.group(2)),
                     prefill_cached=int(m.group(3)))
            STATE["sglang"] = s
            ep = _epoch_from(line)
            if ep:
                STATE["phase"]["sglang_ts"] = ep
                STATE["phase"]["latest_ts"] = max(STATE["phase"]["latest_ts"], ep)
            continue
        m = TIMER_RE.search(line)
        if m:
            ph = STATE["phase"]
            ep = _epoch_from(line)
            if ep:
                ph["latest_ts"] = max(ph["latest_ts"], ep)
            if m.group(2) == "start":
                ph["open"][m.group(1)] = ep
            else:
                ph["open"].pop(m.group(1), None)
                if m.group(3):
                    ph.setdefault("elapsed", {})[m.group(1)] = float(m.group(3))
            continue


def col(store, key):
    """Return (steps, values) sorted by step for a metric key."""
    steps = sorted(store)
    xs, ys = [], []
    for s in steps:
        v = store[s].get(key)
        if isinstance(v, (int, float)):
            xs.append(s)
            ys.append(round(float(v), 6))
    return xs, ys


def _rollouts_per_epoch():
    """Rollouts that make up ONE full pass over the training set. With
    CeilEpochRolloutDataSourceWithBuffer, num_rollout_per_epoch =
    ceil(train_set_size / rollout_batch_size). Read live from the train cmdline
    (--prompt-data line count / --rollout-batch-size); fallback 7 (=ceil(50/8)).
    Cached 300s (doesn't change during a run)."""
    c = STATE["epoch_cache"]
    if time.time() - c["ts"] < 300 and c["data"]:
        return c["data"]
    ep = 7
    try:
        out = subprocess.run(["pgrep", "-f", "train_async.py"],
                             capture_output=True, text=True, timeout=3)
        pid = out.stdout.split()[0] if out.stdout.strip() else None
        if pid:
            args = open("/proc/%s/cmdline" % pid, "rb").read().decode(
                "utf-8", "ignore").split("\0")
            bs = int(args[args.index("--rollout-batch-size") + 1]) if "--rollout-batch-size" in args else 8
            pd = args[args.index("--prompt-data") + 1] if "--prompt-data" in args else None
            if pd and os.path.exists(pd) and bs > 0:
                n = sum(1 for _ in open(pd))
                if n > 0:
                    ep = max(1, (n + bs - 1) // bs)   # integer ceil(n/bs)
    except Exception:
        pass
    STATE["epoch_cache"] = {"ts": time.time(), "data": ep}
    return ep


def _train_set_size():
    """Number of ops in the training set = --prompt-data jsonl line count. This is
    the coverage TARGET N for a coverage-based epoch (1 epoch = all N ops trained
    once). Cached 300s; 0 if unavailable (caller falls back to distinct-ops-seen)."""
    c = STATE["trainset_cache"]
    if time.time() - c["ts"] < 300 and c["data"]:
        return c["data"]
    n = 0
    try:
        out = subprocess.run(["pgrep", "-f", "train_async.py"],
                             capture_output=True, text=True, timeout=3)
        pid = out.stdout.split()[0] if out.stdout.strip() else None
        if pid:
            args = open("/proc/%s/cmdline" % pid, "rb").read().decode(
                "utf-8", "ignore").split("\0")
            pd = args[args.index("--prompt-data") + 1] if "--prompt-data" in args else None
            if pd and os.path.exists(pd):
                n = sum(1 for _ in open(pd))
    except Exception:
        pass
    STATE["trainset_cache"] = {"ts": time.time(), "data": n}
    return n


def rollout_block():
    r = STATE["rollouts"]
    keys = {
        "reward_mean": "polar/reward_mean",
        "reward_completed": "polar/reward_mean_completed",
        "reward_std": "polar/reward_std",
        "success_rate": "polar/rollout_success_rate",
        "resp_len_mean": "rollout/response_len/mean",
        "resp_len_max": "rollout/response_len/max",
        "rollout_time": "perf/rollout_time",
        "staleness": "polar/staleness/mean",
        "truncated": "rollout/truncated_ratio",
        "repetition": "rollout/repetition_frac",
        "run_ms": "polar/session_ms/run_mean",
    }
    out = {}
    for name, k in keys.items():
        xs, ys = col(r, k)
        out[name] = {"x": xs, "y": ys}
    # per-session lengths (from data.py rollout summary): the full prompt→end
    # trajectory, averaged over the round's sessions — what "序列长度" should mean
    d = STATE["data"]
    for name, k in {"total_len_mean": "rollout/total_lengths",
                    "sess_resp_len_mean": "rollout/response_lengths"}.items():
        xs, ys = col(d, k)
        out[name] = {"x": xs, "y": ys}
    return out


def train_block():
    t = STATE["train"]
    keys = ["train/grad_norm", "train/loss", "train/pg_loss", "train/ppo_kl",
            "train/pg_clipfrac", "train/kl_loss", "train/entropy_loss",
            "train/tis_masked", "train/tis_abs_masked", "train/logprob_abs_diff_masked"]
    out = {}
    for k in keys:
        xs, ys = col(t, k)
        out[k.split("/")[-1]] = {"x": xs, "y": ys}
    return out


def eval_block():
    """Held-out validation metrics parsed from 'eval <id>: {...}' log lines.

    Each eval line carries eval/<dataset> = mean reward over that val set plus
    eval/<dataset>/<...> and eval/<dataset>-<...> sub-metrics. We surface the
    per-dataset reward curve (x=rollout_id, y=reward) + latest sub-metrics so the
    page can show both a curve and a detection-status line.
    """
    e = STATE["eval"]
    if not e:
        return {"has": False, "datasets": {}, "latest_id": None}
    names = set()
    for d in e.values():
        for k in d:
            mk = re.match(r"^eval/([A-Za-z0-9_]+)$", k)  # bare eval/<name> = reward
            if mk:
                names.add(mk.group(1))
    latest_id = max(e)
    datasets = {}
    for name in sorted(names):
        xs, ys = col(e, "eval/" + name)
        sub = {k[len("eval/" + name):].lstrip("/-"): round(v, 4)
               for k, v in e[latest_id].items()
               if k.startswith("eval/" + name) and k != "eval/" + name
               and isinstance(v, (int, float))}
        datasets[name] = {"x": xs, "y": ys,
                          "latest": ys[-1] if ys else None,
                          "n_evals": len(ys), "sub": sub}
    return {"has": bool(datasets), "datasets": datasets, "latest_id": latest_id}


# --------------------------- npu-smi ------------------------------
def _npu_info_parse(out):
    cards = {}
    module_temp = {}   # npu 模组号 -> temp(行1 提供,给同模组两个 die 共用)
    # 行1(每模组一条;A3 第二 die 的 power 是 "-"):
    #   老910B: | 0  910B2C | OK | 104.8  54  0 / 0 |
    #   A3:     | 0  Ascend910 | OK | 191.7  36  0 / 0 |   /  | 0  Ascend910 | OK | -  36 ... |
    r1 = re.compile(r"\|\s*(\d+)\s+\S+\s*\|\s*\w+\s*\|\s*(?:[\d.]+|-)\s+(\d+)\s")
    # 行2(带 bus-id 的数据行):
    #   老910B 单 die: | 0 | 0000:5A:00.0 | 100  0 / 0  3774 / 65536 |
    #   A3 双 die:     | 0  0 | 0000:9D:00.0 | 0  0 / 0  40286 / 65536 |
    #   ↑ 首格是 "模组号 die号" 两个数;有 die 号时以它(Phy-ID 0-15)作为卡号 → 面板显示 16 卡。
    #   [2026-08-10 修复] 旧正则首格只允许一个数,在 A3 上一行都匹配不到 → 面板只显示 8 模组且 aicore/hbm 恒 0。
    r2 = re.compile(r"\|\s*(\d+)(?:\s+(\d+))?\s*\|\s*[\w:.]+\s*\|"
                    r"\s*([\d.]+)\s+[\d.]+\s*/\s*[\d.]+\s+(\d+)\s*/\s*(\d+)")
    last_npu_id = None   # 行1 刚给出的 NPU 号,留给紧随其后的行2 用
    for line in out.splitlines():
        m = r1.search(line)
        if m:
            last_npu_id = int(m.group(1))
            module_temp[last_npu_id] = int(m.group(2))
            continue
        m = r2.search(line)
        if m:
            if m.group(2) is not None:
                # A3 双 die:首格是 "模组号 die号",die 的 Phy-ID(0-15)即卡号,温度按模组取。
                card_id = int(m.group(2))
                temp = module_temp.get(card_id // 2, 0)
            else:
                # [2026-08-14 修复] 单 die(910B2C 等):行2 首格是 **Chip 号**,恒为 0
                #   —— 见 npu-smi 表头 "| Chip | Bus-Id | AICore(%) ... |",它不是卡号。
                #   旧代码拿它当卡号 → 16 张卡全写进 cards[0],面板只剩 1 条、
                #   且数值是最后一张卡的。卡号只在行1("| 0  910B2C | OK | ...")里,
                #   所以这里回退到 r1 刚解析出的 NPU 号。
                card_id = last_npu_id if last_npu_id is not None else int(m.group(1))
                temp = module_temp.get(card_id, 0)
            cards[card_id] = {"id": card_id, "power": 0.0,
                              "temp": temp,
                              "aicore": int(float(m.group(3))),
                              "hbm_used": int(m.group(4)),
                              "hbm_total": int(m.group(5))}
    return [cards[i] for i in sorted(cards)]


# [2026-08-11] npu-smi 在训练高负载下会变慢(实测 8s+,甚至分钟级),
# 原先同步调用 + 5s 超时会静默返回 [] → 面板 NPU 空白。
# 改成后台线程刷新 + 旧缓存兜底:请求永不阻塞,瞬时变慢显示缓存(标 stale)。
_NPU_CACHE = {"data": [], "ts": 0.0, "refreshing": False, "ever_ok": False}


def _npu_refresh():
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True,
                             text=True, timeout=30).stdout
        data = _npu_info_parse(out)
        if data:
            _NPU_CACHE.update(data=data, ts=time.time(), ever_ok=True)
    except Exception:
        pass
    finally:
        _NPU_CACHE["refreshing"] = False


def npu_info():
    now = time.time()
    if not _NPU_CACHE["refreshing"] and (now - _NPU_CACHE["ts"] > 10):
        _NPU_CACHE["refreshing"] = True
        threading.Thread(target=_npu_refresh, daemon=True).start()
    return _NPU_CACHE["data"]


# --------------------------- peer node (140) ------------------------------
# [2026-08-10] 跨机节点状态:140 跑 tools/node_exporter.py,这里定时拉取。
# 无免密 ssh,故走 HTTP。拉取失败用 ≤120s 的旧缓存兜底,绝不让面板刷新被拖死。
PEER_METRICS_URL = os.environ.get("PEER_METRICS_URL",
                                  "http://80.48.5.64:19100/node_metrics")
PEER = {"last_try": 0.0, "good": None, "good_ts": 0.0}

# 本机的 vLLM monitor 已经负责运行时 engine 发现、Prometheus 采集和吞吐
# delta 计算。dashboard 只读它的缓存 API，避免再逐个探测繁忙的 engine。
LOCAL_VLLM_METRICS_URL = os.environ.get(
    "LOCAL_VLLM_METRICS_URL", "http://127.0.0.1:5000/api/metrics")
LOCAL_VLLM = {"last_try": 0.0, "good": None, "good_ts": 0.0}


def peer_metrics():
    now = time.time()
    if now - PEER["last_try"] < 5:           # 5s 内不重复拉,直接用缓存
        if PEER["good"] is not None:
            out = dict(PEER["good"])
            out["age"] = round(now - PEER["good_ts"])
            return out
        return {"ok": False, "err": "pending"}
    PEER["last_try"] = now
    try:
        import urllib.request
        # 集群内网地址必须绕开 http_proxy 代理,否则被代理网关拦(504)。
        # 训练脚本靠 no_proxy 环境变量;面板进程环境不一定有 → 代码里硬性绕过。
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(PEER_METRICS_URL, timeout=2.5) as r:
            data = json.loads(r.read().decode())
        PEER["good"] = {"ok": True, "data": data}
        PEER["good_ts"] = now
        return {"ok": True, "age": 0, "data": data}
    except Exception as e:
        if PEER["good"] is not None and now - PEER["good_ts"] < 120:
            out = dict(PEER["good"])
            out.update(stale=True, age=round(now - PEER["good_ts"]))
            return out
        return {"ok": False, "err": type(e).__name__}


def local_vllm_metrics():
    """Read the local N-engine monitor cache without polling engines again."""
    now = time.time()
    if now - LOCAL_VLLM["last_try"] < 3:
        if LOCAL_VLLM["good"] is not None:
            out = dict(LOCAL_VLLM["good"])
            out["age"] = round(now - LOCAL_VLLM["good_ts"])
            return out
        return {"ok": False, "err": "pending"}
    LOCAL_VLLM["last_try"] = now
    try:
        import urllib.request
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(LOCAL_VLLM_METRICS_URL, timeout=1.0) as r:
            data = json.loads(r.read().decode())
        # monitor 的 300 点历史用于它自己的曲线页；6007 只展示当前快照，
        # 不把约百 KB 的 history 每 3 秒再转发给浏览器。
        data = {key: data.get(key) for key in (
            "last_update", "engines", "engines_total", "engines_alive",
            "combined", "avg_per_engine", "starved_pct",
        )}
        LOCAL_VLLM["good"] = {"ok": True, "data": data}
        LOCAL_VLLM["good_ts"] = now
        return {"ok": True, "age": 0, "data": data}
    except Exception as e:
        if LOCAL_VLLM["good"] is not None and now - LOCAL_VLLM["good_ts"] < 30:
            out = dict(LOCAL_VLLM["good"])
            out.update(stale=True, age=round(now - LOCAL_VLLM["good_ts"]))
            return out
        return {"ok": False, "err": type(e).__name__}


def host_mem():
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                p = line.split()
                if p and p[0].rstrip(":") in ("MemTotal", "MemAvailable"):
                    info[p[0].rstrip(":")] = int(p[1])  # kB
        total = info.get("MemTotal", 0) / 1024 / 1024
        avail = info.get("MemAvailable", 0) / 1024 / 1024
        used = total - avail
        peak = max(STATE.get("host_peak", 0.0), used)
        STATE["host_peak"] = peak
        thr = total * 0.995      # Ray RAY_memory_usage_threshold kill point
        # --- history (leak vs spike): sample sparsely, keep a bounded window ---
        now = time.time()
        hist = STATE["host_hist"]
        if not hist or now - hist[-1][0] >= HOST_HIST_EVERY:
            hist.append((now, round(used, 2)))
            if len(hist) > HOST_HIST_MAX:
                del hist[:len(hist) - HOST_HIST_MAX]
        t0 = hist[0][0] if hist else now
        return {"used_gb": round(used, 1), "total_gb": round(total, 1),
                "pct": round(100 * used / total, 1) if total else 0,
                "peak_gb": round(peak, 1), "thr_gb": round(thr, 1),
                "margin_gb": round(thr - used, 1),
                # x = minutes since first sample, y = used GB
                "hist_x": [round((t - t0) / 60.0, 2) for t, _ in hist],
                "hist_y": [u for _, u in hist]}
    except Exception:
        return {"used_gb": 0, "total_gb": 0, "pct": 0}


# ------------------------ operator status -------------------------
def operator_status():
    c = STATE["op_cache"]
    if time.time() - c["ts"] < 15 and c["data"]:
        return c["data"]
    # Match the CURRENT training run (by run_id from the live log), NOT the
    # newest polar dir by mtime — a just-started run has no sessions yet, and
    # mtime would wrongly pick the previous run's dir (stale completed/error).
    run_id = _current_run_id()
    data = {"polar_run": run_id,
            "total": 0, "by_status": {}, "current_op": None, "mean_turns": 0}
    if run_id:
        ses = [s for s in glob.glob(os.path.join(POLAR_RUNS, "polar_*",
               "rollout_results", "run_" + run_id, "**", "ses_*.json"),
               recursive=True) if "-eval-" not in s]   # training sessions only
        counts, turns, newest, newest_op = {}, [], 0.0, None
        st_re = re.compile(r'"status":\s*"(\w+)"')
        rc_re = re.compile(r'"record_count":\s*(\d+)')
        op_re = re.compile(r'"op_name":\s*"([^"]+)"')
        for s in ses:
            try:
                head = open(s, "rb").read(6000).decode("utf-8", "ignore")
            except OSError:
                continue
            m = st_re.search(head)
            status = m.group(1) if m else "UNKNOWN"
            counts[status] = counts.get(status, 0) + 1
            rc = rc_re.search(head)
            if rc:
                turns.append(int(rc.group(1)))
            mt = os.path.getmtime(s)
            if mt > newest:
                newest = mt
                om = op_re.search(head)
                newest_op = om.group(1) if om else None
        data.update(total=len(ses), by_status=counts, current_op=newest_op,
                    mean_turns=round(sum(turns) / len(turns), 1) if turns else 0)
    STATE["op_cache"] = {"ts": time.time(), "data": data}
    return data


# ------------------ per-session decode throughput -----------------
def session_throughput():
    """Per-session decode tok/s from Polar completion_metrics.jsonl.

    Mirrors CompletionMetricsAggregate: session tok/s = Σcompletion_tokens /
    (Σlatency_ms/1000) over that session's completion events. Denominator is the
    sum of per-request latencies (queue+prefill+decode), NOT session wall-clock
    (excludes tool/verify gaps). Returns the distribution + percentiles.
    """
    c = STATE["tps_cache"]
    if time.time() - c["ts"] < 20 and c["data"]:
        return c["data"]
    data = {"n": 0, "values": [], "p10": None, "p50": None, "p90": None,
            "mean": None, "min": None, "max": None, "capped": 0}
    # Match the CURRENT run (by run_id), consistent with the other Polar panels;
    # mtime would grab a previous run's completion_metrics when this run is new.
    run_id = _current_run_id()
    if run_id:
        files = glob.glob(os.path.join(POLAR_RUNS, "polar_*", "rollout_results",
                                       "run_" + run_id, "**",
                                       "completion_metrics.jsonl"),
                          recursive=True)
        files.sort(key=os.path.getmtime, reverse=True)
        CAP = 800                       # newest N sessions (bound refresh cost)
        capped = max(0, len(files) - CAP)
        vals = []
        for f in files[:CAP]:
            ct, lt = 0, 0.0
            try:
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                        except ValueError:
                            continue
                        ct += int(d.get("completion_tokens") or 0)
                        lat = d.get("latency_ms")
                        if lat:
                            lt += float(lat)
            except OSError:
                continue
            if ct > 0 and lt > 0:
                vals.append(ct / (lt / 1000.0))
        vals.sort()
        if vals:
            n = len(vals)
            pct = lambda p: round(vals[min(n - 1, int(p * n))], 1)
            data = {
                "n": n,
                "values": [round(v, 1) for v in vals],
                "p10": pct(0.10), "p50": pct(0.50), "p90": pct(0.90),
                "mean": round(sum(vals) / n, 1),
                "min": round(vals[0], 1), "max": round(vals[-1], 1),
                "capped": capped,
            }
    STATE["tps_cache"] = {"ts": time.time(), "data": data}
    return data


# ------------------------ training phase --------------------------
def num_rollout_total():
    """--num-rollout of the running train_async.py (cached once found)."""
    global _NUM_ROLLOUT
    if _NUM_ROLLOUT:
        return _NUM_ROLLOUT
    try:
        out = subprocess.run(["pgrep", "-af", "train_async.py"],
                             capture_output=True, text=True, timeout=3).stdout
        m = re.search(r"--num-rollout\s+(\d+)", out)
        if m:
            _NUM_ROLLOUT = int(m.group(1))
    except Exception:
        pass
    return _NUM_ROLLOUT or 0


# timers that mean "the trainer is doing a train step"
_TRAIN_TIMERS = ("train", "actor_train", "log_probs", "ref_log_probs",
                 "data_preprocess")


def phase_status():
    """What the run is doing NOW, inferred from open named-timers + recent
    sglang activity. Phases can overlap (async): e.g. training + generating."""
    p = STATE["phase"]
    now = p["latest_ts"]
    openset = p["open"]
    weight_update = "update_weights" in openset
    training = any(t in openset for t in _TRAIN_TIMERS)
    # generating: trainer explicitly waiting for rollout data, OR sglang decoded
    # within the last 25s (relative to the freshest log line = "now")
    gen_recent = now > 0 and p["sglang_ts"] > 0 and (now - p["sglang_ts"]) <= 25
    generating = ("train_wait" in openset) or gen_recent

    last = max(list(STATE["rollouts"]) + list(STATE["data"]) + [-1])
    cur = last
    # if only generating (past the last train step) we're producing the next one
    if generating and not training and not weight_update and last >= 0:
        cur = last + 1

    phases = []
    if weight_update:
        phases.append("权重更新")
    if training:
        phases.append("训练")
    if generating:
        phases.append("生成")

    # round cadence / ETA from consecutive data.py rollout timestamps
    rt = [t for _, t in p.get("round_ts", [])]
    mean_round = None
    if len(rt) >= 2:
        diffs = [rt[i + 1] - rt[i] for i in range(len(rt) - 1) if rt[i + 1] > rt[i]]
        if diffs:
            mean_round = sum(diffs) / len(diffs)
    total = num_rollout_total()
    since_round = (now - rt[-1]) if (rt and now >= rt[-1]) else None
    eta = max(0, (total - 1 - last)) * mean_round if (mean_round and total and last >= 0) else None
    el = p.get("elapsed", {})
    return {
        "rollout": cur if cur >= 0 else None,
        "last_done": last if last >= 0 else None,
        "total": total,
        "phases": phases,
        "weight_update": weight_update,
        "training": training,
        "generating": generating,
        "eval": eval_running(),
        "open_timers": sorted(openset),
        "elapsed": {k: round(el[k], 1) for k in el},
        "mean_round_sec": round(mean_round) if mean_round else None,
        "since_round_sec": round(since_round) if since_round is not None else None,
        "eta_sec": round(eta) if eta is not None else None,
    }


def _current_run_id():
    """RUN_ID from the live log filename train_<RUN_ID>.log."""
    lp = STATE.get("log_path")
    if lp:
        b = os.path.basename(lp)
        if b.startswith("train_") and b.endswith(".log"):
            return b[6:-4]
    return None


# ---------------------- operators run so far ----------------------
def operators_run():
    """Cumulative operators run for the current training run (1 operator = 8
    samples; 1 sample = 1 session). Counts distinct task groups (dir name with
    the -spNNN sample suffix stripped) across all polar dirs. Cached 20s."""
    c = STATE["opsrun_cache"]
    if time.time() - c["ts"] < 20 and c["data"]:
        return c["data"]
    run_id = _current_run_id()
    ops = samples = 0
    per_h = None
    if run_id:
        dirs = [d for d in glob.glob(os.path.join(POLAR_RUNS, "polar_*",
                "rollout_results", "run_" + run_id, "task_*"))
                if "-eval-" not in os.path.basename(d)]   # training only
        samples = len(dirs)
        ops = len({re.sub(r"-sp\d+$", "", os.path.basename(d)) for d in dirs})
        mm = re.search(r"_(\d{8})_(\d{6})$", run_id)
        if mm:
            try:
                st = time.mktime(time.strptime(mm.group(1) + mm.group(2),
                                               "%Y%m%d%H%M%S"))
                hrs = (time.time() - st) / 3600
                if hrs > 0:
                    per_h = round(ops / hrs, 1)
            except ValueError:
                pass
    data = {"operators": ops, "samples": samples, "per_hour": per_h}
    STATE["opsrun_cache"] = {"ts": time.time(), "data": data}
    return data


def solved_stats():
    """Sessions that FINALLY SOLVED the operator (trajectory.metadata.evaluation
    .success == true) vs total sessions, for the current run. status=COMPLETED
    only means the session finished; `success` is the pass/fail verdict. The
    evaluation block sits in the first ~2KB (before the huge traces array), so a
    bounded 8KB prefix read suffices. Cached 20s."""
    c = STATE["solved_cache"]
    if time.time() - c["ts"] < 20 and c["data"]:
        return c["data"]
    run_id = _current_run_id()
    solved = total = 0
    if run_id:
        files = [f for f in glob.glob(os.path.join(POLAR_RUNS, "polar_*",
                 "rollout_results", "run_" + run_id, "**", "ses_*.json"),
                 recursive=True) if "-eval-" not in f]   # training only
        ev_re = re.compile(r'"evaluation".*?"success":\s*(true|false)', re.S)
        for f in files:
            total += 1
            try:
                head = open(f, "rb").read(8192).decode("utf-8", "ignore")
            except OSError:
                continue
            mm = ev_re.search(head)
            if mm and mm.group(1) == "true":
                solved += 1
    data = {"solved": solved, "total": total,
            "rate": round(solved / total, 3) if total else None}
    STATE["solved_cache"] = {"ts": time.time(), "data": data}
    return data


def op_round_compare():
    """Per-operator solve rate BY ROLLOUT ROUND, for ops sampled in >=2 rounds.

    Round = rollout_step; a round's solve rate = solved sessions / sessions of
    that op in that round (group_size ~= 8). delta = latest-round rate minus
    first-round rate — the "前后两轮" lift that answers "did revisiting this
    operator after more training help?". Rows sorted by delta desc. An op whose
    latest round is still generating shows that round's partial solved/n.
    Cached 30s (same 8KB-prefix scan trick as solved_stats)."""
    c = STATE["oprounds_cache"]
    if time.time() - c["ts"] < 30 and c["data"]:
        return c["data"]
    run_id = _current_run_id()
    data = {"polar_run": run_id, "total_ops": 0, "repeated": 0,
            "improved": 0, "declined": 0, "flat": 0, "rows": []}
    if run_id:
        files = [f for f in glob.glob(os.path.join(POLAR_RUNS, "polar_*",
                 "rollout_results", "run_" + run_id, "**", "ses_*.json"),
                 recursive=True) if "-eval-" not in f]   # training only
        rs_re = re.compile(r'"rollout_step":\s*(\d+)')
        op_re = re.compile(r'"op_name":\s*"([^"]+)"')
        rw_re = re.compile(r'"reward":\s*([-\d.eE]+),\s*"success":\s*(true|false)')
        ops = {}   # op_name -> {rollout_step: [solved, total]}
        for f in files:
            try:
                head = open(f, "rb").read(8192).decode("utf-8", "ignore")
            except OSError:
                continue
            rs, op, rw = rs_re.search(head), op_re.search(head), rw_re.search(head)
            if not (rs and op and rw):
                continue
            cell = ops.setdefault(op.group(1), {}).setdefault(
                int(rs.group(1)), [0, 0])
            cell[1] += 1
            if rw.group(2) == "true":
                cell[0] += 1
        rows = []
        for op, steps in ops.items():
            rounds = [{"r": s, "solved": sv, "n": n, "rate": round(sv / n, 3)}
                      for s, (sv, n) in sorted(steps.items()) if n]
            if len(rounds) < 2:
                continue                      # only ops repeated across rounds
            # TREND over ALL rounds (not just first vs last): least-squares slope of
            # solve-rate vs appearance index. slope = per-revisit lift; trend =
            # slope*(k-1) = fitted first->last change using every point (robust to the
            # noisy single endpoints). mono = (#up - #down)/(k-1) monotonicity, -1..1.
            rates = [r["rate"] for r in rounds]
            k = len(rates)
            mx = (k - 1) / 2.0
            my = sum(rates) / k
            sxx = sum((i - mx) ** 2 for i in range(k))
            sxy = sum((i - mx) * (y - my) for i, y in enumerate(rates))
            slope = (sxy / sxx) if sxx else 0.0
            trend = slope * (k - 1)
            ups = sum(1 for a, b in zip(rates, rates[1:]) if b > a)
            downs = sum(1 for a, b in zip(rates, rates[1:]) if b < a)
            rows.append({"op": op.replace("cuda_llm_ops_1_2_simple_", ""),
                         "full": op, "rounds": rounds,
                         "delta": round(rates[-1] - rates[0], 3),   # raw first->last (kept for ref)
                         "slope": round(slope, 3),                  # per-revisit lift
                         "trend": round(trend, 3),                  # all-points fitted change
                         "mono": round((ups - downs) / (k - 1), 2)})
        rows.sort(key=lambda r: (-r["trend"], r["op"]))
        EPS = 0.05                            # deadband: |trend|<=5pp counts as flat
        data.update(total_ops=len(ops), repeated=len(rows), rows=rows,
                    improved=sum(1 for r in rows if r["trend"] > EPS),
                    declined=sum(1 for r in rows if r["trend"] < -EPS),
                    flat=sum(1 for r in rows if abs(r["trend"]) <= EPS))
    STATE["oprounds_cache"] = {"ts": time.time(), "data": data}
    return data


def holdout_eval_series():
    """holdout pass@1 vs step — the verdict curve. Reads the offline ckpt-eval
    results written by tools/polar_ckpt_eval_watch.py (eval_results.jsonl).
    Cached 60s."""
    c = STATE["holdout_cache"]
    if time.time() - c["ts"] < 60 and c["data"]:
        return c["data"]
    data = {}
    path = "/workspace/eval_ckpts/eval_results.jsonl"
    rows = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except Exception:
            rows = []
    if rows:
        rows.sort(key=lambda r: r.get("step", 0))
        data = {
            "pass_at_1": {"x": [r.get("step") for r in rows if r.get("pass_at_1") is not None],
                          "y": [r["pass_at_1"] for r in rows if r.get("pass_at_1") is not None]},
            "reward_mean": {"x": [r.get("step") for r in rows if r.get("reward_mean") is not None],
                            "y": [r["reward_mean"] for r in rows if r.get("reward_mean") is not None]},
            "latest": rows[-1],
            "n_points": len(rows),
        }
    STATE["holdout_cache"] = {"ts": time.time(), "data": data}
    return data


def health_metrics():
    """Signal-quality health of the CURRENT training run, computed from session
    heads (8KB prefix read, same trick as solved_stats). Panels:
      - zero-variance group fraction per rollout step: GRPO gives all-equal-reward
        groups zero advantage -> those sessions produced no gradient (pure waste).
      - failure taxonomy (error_type counts).
    Cached 60s."""
    c = STATE["health_cache"]
    if time.time() - c["ts"] < 60 and c["data"]:
        return c["data"]
    data = {}
    run_id = _current_run_id()
    if run_id:
        files = [f for f in glob.glob(os.path.join(POLAR_RUNS, "polar_*",
                 "rollout_results", "run_" + run_id, "**", "ses_*.json"),
                 recursive=True) if "-eval-" not in f]
        try:
            files.sort(key=lambda f: os.stat(f).st_mtime)   # chronological attempts
        except OSError:
            pass
        rs_re = re.compile(r'"rollout_step":\s*(\d+)')
        gr_re = re.compile(r'"group_index":\s*(\d+)')
        op_re = re.compile(r'"op_name":\s*"([^"]+)"')
        rw_re = re.compile(r'"reward":\s*([-\d.eE]+),\s*"success":\s*(true|false)')
        er_re = re.compile(r'"error_type":\s*"([^"]+)"')
        groups, errors, op_rw = {}, {}, {}
        n_sess = 0
        for f in files:
            try:
                head = open(f, "rb").read(8192).decode("utf-8", "ignore")
            except OSError:
                continue
            rs, gr, rw = rs_re.search(head), gr_re.search(head), rw_re.search(head)
            if not (rs and gr and rw):
                continue
            n_sess += 1
            step, gid = int(rs.group(1)), int(gr.group(1))
            reward = float(rw.group(1))
            groups.setdefault(step, {}).setdefault(gid, []).append(reward)
            op = op_re.search(head)
            if op:
                op_rw.setdefault(op.group(1), []).append((step, reward))
            er = er_re.search(head)
            if er:
                errors[er.group(1)] = errors.get(er.group(1), 0) + 1
        if n_sess:
            steps = sorted(groups)
            zx, zf, zn, zm = [], [], [], []
            for s in steps:
                gs = [v for v in groups[s].values() if len(v) >= 2]
                zx.append(s)
                zn.append(len(gs))
                zf.append(round(sum(1 for v in gs if max(v) - min(v) < 1e-9) / len(gs), 3) if gs else 0.0)
                allr = [r for v in groups[s].values() for r in v]
                zm.append(round(sum(allr) / len(allr), 3) if allr else 0.0)
            # op-composition-adjusted reward curve (noise-immune learning view):
            # subtract each OP's own baseline so the curve shows "did the policy beat
            # its own past on the ops it drew" instead of "which ops got sampled" (~70%
            # of the raw per-rollout wobble). Key on op_name: session group_index is a
            # unique-per-appearance counter (NOT a stable op id), so op_name is the only
            # correct key. MA over one full 50-op epoch (~7 rollouts) cancels the
            # residual composition noise. Empty if op_name absent from session heads.
            op_base = {op: sum(r for _, r in v) / len(v) for op, v in op_rw.items() if v}
            _allrw = [r for v in op_rw.values() for _, r in v]
            grand = (sum(_allrw) / len(_allrw)) if _allrw else 0.0
            step_raw, step_res = {}, {}
            for op, v in op_rw.items():
                for stp, r in v:
                    step_raw.setdefault(stp, []).append(r)
                    step_res.setdefault(stp, []).append(r - op_base[op])
            rax = sorted(step_res)
            raw_c = [round(sum(step_raw[s]) / len(step_raw[s]), 4) for s in rax]
            adj_c = [round(grand + sum(step_res[s]) / len(step_res[s]), 4) for s in rax]

            def _ma(x, w=7):
                out = []
                for i in range(len(x)):
                    seg = x[max(0, i - w + 1):i + 1]
                    out.append(round(sum(seg) / len(seg), 4))
                return out
            ma_c = _ma(adj_c, 7)
            # trend = smoothed(MA7) first->last, NOT raw adj endpoints: the last rollout
            # is often an in-progress half-batch whose point is noisy/inflated.
            reward_adj = {"x": rax, "raw": raw_c, "adj": adj_c, "ma": ma_c,
                          "grand": round(grand, 4),
                          "slope": round((ma_c[-1] - ma_c[0]), 4) if len(ma_c) >= 2 else None}
            # per-EPOCH reward, APPEARANCE-INDEXED: epoch k = the (k+1)-th time EACH op is
            # trained (epoch0 = every op's 1st appearance, epoch1 = 2nd, ...). Each op
            # appears EXACTLY ONCE per epoch, so every epoch has identical composition (all N
            # ops, one appearance each) -> no op-correction needed. An epoch is emitted only
            # when ALL N ops have that appearance (未满不画). N = --prompt-data line count.
            # One "appearance" = one rollout_step at which the op was sampled (its ~8 samples).
            N = _train_set_size() or len(op_rw)
            op_apps = {}   # op -> [(step, [rewards]), ...] in appearance (chronological) order
            for op, v in op_rw.items():
                bystep = {}
                for stp, r in v:
                    bystep.setdefault(stp, []).append(r)
                op_apps[op] = [(s, bystep[s]) for s in sorted(bystep)]
            maxk = max((len(a) for a in op_apps.values()), default=0)
            eex, eraw, espan = [], [], []
            for k in range(maxk):
                ops_k = [op for op, a in op_apps.items() if len(a) > k]
                if len(ops_k) < N:            # not all N ops have a (k+1)-th appearance yet
                    break                     # counts are monotonic in k -> stop
                rr = [r for op in ops_k for r in op_apps[op][k][1]]
                stps = [op_apps[op][k][0] for op in ops_k]
                eex.append(k)
                eraw.append(round(sum(rr) / len(rr), 4))
                espan.append([min(stps), max(stps)])
            epoch_reward_adj = {"x": eex, "raw": eraw, "span": espan, "N": N,
                                "slope": round((eraw[-1] - eraw[0]), 4) if len(eraw) >= 2 else None}
            data = {
                "n_sessions": n_sess,
                "zero_var": {"x": zx, "frac": zf, "n_groups": zn, "reward_mean": zm},
                "reward_adj": reward_adj,
                "epoch_reward_adj": epoch_reward_adj,
                "latest_zero_var_frac": zf[-1] if zf else None,
                "errors": sorted(errors.items(), key=lambda kv: -kv[1])[:5],
            }
    STATE["health_cache"] = {"ts": time.time(), "data": data}
    return data


def _eval_set_total():
    """Total operators in the held-out eval set (dynamic denominator). Reads the
    live train cmdline's --eval-config yaml, follows its dataset `path:` to the
    jsonl and counts rows. Cached long (doesn't change during a run); returns 0
    if unavailable (caller falls back to distinct-ops-seen)."""
    c = STATE["evalset_cache"]
    if time.time() - c["ts"] < 300 and c["data"]:
        return c["data"]
    total = 0
    try:
        out = subprocess.run(["pgrep", "-f", "train_async.py"],
                             capture_output=True, text=True, timeout=3)
        pid = out.stdout.split()[0] if out.stdout.strip() else None
        cfg = None
        if pid:
            args = open("/proc/%s/cmdline" % pid, "rb").read().decode(
                "utf-8", "ignore").split("\0")
            if "--eval-config" in args:
                cfg = args[args.index("--eval-config") + 1]
        if cfg and os.path.exists(cfg):
            for line in open(cfg):
                m = re.search(r"path:\s*(\S+\.jsonl)", line)
                if m and os.path.exists(m.group(1)):
                    total += sum(1 for _ in open(m.group(1)))   # sum over datasets
    except Exception:
        total = 0
    STATE["evalset_cache"] = {"ts": time.time(), "data": total}
    return total


def eval_operator_status():
    """Per-operator verification status of the held-out eval set, for the LATEST
    eval round only. Between eval rounds no new eval sessions land, so the
    snapshot naturally freezes until the next eval round starts producing them."""
    c = STATE["evalop_cache"]
    if time.time() - c["ts"] < 15 and c["data"]:
        return c["data"]
    run_id = _current_run_id()
    data = {"round": None, "ops": [], "n_ops": 0, "n_samples": 0,
            "samples": 0, "solved": 0}
    if run_id:
        evs = [f for f in glob.glob(os.path.join(POLAR_RUNS, "polar_*",
               "rollout_results", "run_" + run_id, "**", "ses_*.json"),
               recursive=True) if "-eval-" in f]
        rs_re = re.compile(r'"rollout_step":\s*(\d+)')
        op_re = re.compile(r'"op_name":\s*"([^"]+)"')
        rw_re = re.compile(r'"reward":\s*([-\d.eE]+),\s*"success":\s*(true|false)')
        rounds = {}    # eval round -> {op_name: [(reward, success), ...]}
        for f in evs:
            try:
                head = open(f, "rb").read(6000).decode("utf-8", "ignore")
            except OSError:
                continue
            rs, op = rs_re.search(head), op_re.search(head)
            if not rs or not op:
                continue
            rw = rw_re.search(head)
            rounds.setdefault(int(rs.group(1)), {}).setdefault(
                op.group(1), []).append(
                (float(rw.group(1)) if rw else None,
                 rw.group(2) == "true" if rw else None))
        if rounds:
            rnd = max(rounds)                       # latest eval round only
            ops, samples, solved = [], 0, 0
            for op, s in sorted(rounds[rnd].items()):
                rews = [r for r, _ in s if r is not None]
                sv = sum(1 for _, k in s if k)
                samples += len(s)
                solved += sv
                ops.append({"op": op.replace("cuda_llm_ops_1_2_simple_", ""),
                            "run": len(s), "solved": sv,
                            "reward": round(sum(rews) / len(rews), 3) if rews else None})
            data = {"round": rnd, "ops": ops, "n_ops": len(ops),
                    "total_ops": _eval_set_total() or len(ops),
                    "n_samples": max((o["run"] for o in ops), default=0),
                    "samples": samples, "solved": solved}
    STATE["evalop_cache"] = {"ts": time.time(), "data": data}
    return data


# --------------------------- payload ------------------------------
def build_metrics():
    update_log()
    r = STATE["rollouts"]
    cur_step = max(r) if r else None
    latest = r[cur_step] if cur_step is not None else {}
    return {
        "meta": {
            "log_file": os.path.basename(STATE["log_path"] or ""),
            "alive": train_alive(),
            "uptime": train_uptime(),
            "rollout_step": cur_step,
            "n_rollouts": len(r),
            "n_train_steps": len(STATE["train"]),
            "integrity_rejects": STATE.get("integrity_rejects", 0),
            "updated_at": time.strftime("%H:%M:%S"),
            "latest_reward": latest.get("polar/reward_mean"),
            "latest_success": latest.get("polar/rollout_success_rate"),
        },
        "rollouts": rollout_block(),
        "train": train_block(),
        "npu": npu_info(),
        "peer": peer_metrics(),
        "local_vllm": local_vllm_metrics(),
        "host": host_mem(),
        "sglang": STATE.get("sglang", {}),
        "operators": operator_status(),
        "ops_run": operators_run(),
        "solved_stats": solved_stats(),
        "op_rounds": op_round_compare(),
        "health": health_metrics(),
        "session_tps": session_throughput(),
        "phase": phase_status(),
    }


# ----------------------- raw training log -------------------------
def _decode(b):
    """Bytes -> display text: strip ANSI colour codes, tolerate bad utf-8."""
    return ANSI.sub("", b.decode("utf-8", errors="replace"))


def serve_log(qs):
    """Serve the live train_<RUN_ID>.log via line-aligned byte cursors.

    Modes (query args):
      tail=1            -> last LOG_TAIL_INIT bytes (initial paint)
      from=<offset>     -> new complete lines after <offset> (follow / tail -f)
      before=<offset>   -> one older page of lines ending at <offset> (scroll up)
    Always echoes `file` (basename); the client passes it back so a run switch
    (latest_log() changed) is reported as {rotated:true} instead of garbage.
    """
    path = latest_log()
    if path is None:
        return {"file": None, "text": "", "start": 0, "end": 0, "size": 0,
                "at_start": True}
    base = os.path.basename(path)
    try:
        sz = os.path.getsize(path)
    except OSError:
        return {"file": None, "text": "", "start": 0, "end": 0, "size": 0,
                "at_start": True}
    want = qs.get("file", [None])[0]

    # ---- follow: complete lines after `from` ----
    if "from" in qs:
        if want and want != base:
            return {"rotated": True, "file": base, "size": sz}
        try:
            frm = int(qs["from"][0])
        except ValueError:
            frm = 0
        if frm > sz:                      # truncated / rotated in place
            return {"rotated": True, "file": base, "size": sz}
        with open(path, "rb") as f:
            f.seek(frm)
            chunk = f.read(sz - frm)
        lnl = chunk.rfind(b"\n")
        if lnl == -1:                     # no complete new line yet
            return {"file": base, "text": "", "end": frm, "size": sz}
        return {"file": base, "text": _decode(chunk[:lnl + 1]),
                "end": frm + lnl + 1, "size": sz}

    # ---- scroll up: one older page ending at `before` ----
    if "before" in qs:
        if want and want != base:
            return {"rotated": True, "file": base, "size": sz}
        try:
            before = int(qs["before"][0])
        except ValueError:
            before = sz
        before = max(0, min(before, sz))
        bstart = max(0, before - LOG_PAGE)
        with open(path, "rb") as f:
            f.seek(bstart)
            chunk = f.read(before - bstart)
        if bstart > 0:                    # align to a line boundary
            fnl = chunk.find(b"\n")
            if fnl != -1:
                bstart += fnl + 1
                chunk = chunk[fnl + 1:]
        return {"file": base, "text": _decode(chunk),
                "start": bstart, "at_start": bstart == 0}

    # ---- default / tail=1: last LOG_TAIL_INIT bytes ----
    start = max(0, sz - LOG_TAIL_INIT)
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(sz - start)
    if start > 0:
        fnl = chunk.find(b"\n")
        if fnl != -1:
            start += fnl + 1
            chunk = chunk[fnl + 1:]
    lnl = chunk.rfind(b"\n")
    if lnl == -1:
        text, end = _decode(chunk), sz
    else:
        text, end = _decode(chunk[:lnl + 1]), start + lnl + 1
    return {"file": base, "text": text, "start": start, "end": end,
            "size": sz, "at_start": start == 0}


# ----------------------------- HTML -------------------------------
PAGE = r"""<!doctype html><html lang=zh><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>RL Training Monitor</title><style>
:root{--bg:#eef0f6;--panel:#ffffff;--line:rgba(0,0,0,.07);--text:#1d1d1f;--muted:#86868b;--grid:#e8e8ed;--gridline:#d0d0d8;
  --accent:#0071e3;--hair:rgba(0,0,0,.10);--chipbg:rgba(0,0,0,.05);
  --bggrad:radial-gradient(1200px 600px at 15% -10%,rgba(0,113,227,.08),transparent 60%),radial-gradient(900px 500px at 90% 0%,rgba(94,92,230,.07),transparent 55%),#eef0f6;
  --logbg:#fbfbfd;
  --shadow:0 1px 2px rgba(0,0,0,.04),0 6px 18px rgba(0,0,0,.05);
  --shadow-h:0 2px 4px rgba(0,0,0,.05),0 12px 28px rgba(0,0,0,.09)}
html[data-theme=dark]{--bg:#0b0d13;--panel:#17191f;--line:rgba(255,255,255,.09);--text:#f5f5f7;--muted:#9a9aa0;--grid:#26282f;--gridline:#3d4048;
  --accent:#2997ff;--hair:rgba(255,255,255,.14);--chipbg:rgba(255,255,255,.08);
  --bggrad:radial-gradient(1200px 600px at 15% -10%,rgba(41,151,255,.10),transparent 60%),radial-gradient(900px 500px at 90% 0%,rgba(191,90,242,.08),transparent 55%),#0b0d13;
  --logbg:#101216;
  --shadow:0 1px 2px rgba(0,0,0,.5),0 6px 18px rgba(0,0,0,.3);
  --shadow-h:0 2px 4px rgba(0,0,0,.5),0 14px 34px rgba(0,0,0,.45)}
*{box-sizing:border-box}
::selection{background:color-mix(in srgb,var(--accent) 25%,transparent)}
body{margin:0;-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;
  font:13px/1.47 -apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text","Helvetica Neue","PingFang SC","Microsoft YaHei",sans-serif;
  background:var(--bggrad);background-attachment:fixed;color:var(--text);height:100vh;display:flex;flex-direction:column;overflow:hidden;transition:background .35s ease,color .35s ease}
#main{flex:1;display:flex;min-height:0}
.left{flex:1;overflow-y:auto;min-width:0;scroll-behavior:smooth;padding-bottom:28px}
.right{display:flex;flex-direction:column;min-height:0;width:42%;max-width:860px;min-width:360px;border-left:.5px solid var(--line);background:var(--panel);box-shadow:-12px 0 32px rgba(0,0,0,.04)}
html[data-theme=dark] .right{box-shadow:-12px 0 32px rgba(0,0,0,.3)}
.loghead{flex:0 0 auto;display:flex;align-items:center;gap:10px;padding:10px 16px;border-bottom:.5px solid var(--line);font-size:12px;color:var(--muted);letter-spacing:-.01em;font-weight:500}
.loghead button,#theme{cursor:pointer;background:var(--chipbg);color:var(--text);border:.5px solid var(--hair);border-radius:980px;padding:5px 13px;font-size:12px;line-height:1;font-weight:500;transition:background .2s,color .2s,transform .1s,box-shadow .2s}
.loghead button:hover,#theme:hover{background:var(--accent);color:#fff;border-color:transparent;box-shadow:0 4px 12px color-mix(in srgb,var(--accent) 35%,transparent)}
.loghead button:active,#theme:active{transform:scale(.96)}
.loghead button:focus-visible,#theme:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
#logbox{flex:1;overflow-y:auto;margin:0;padding:12px 16px;background:var(--logbg);font:11.5px/1.65 ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,"DejaVu Sans Mono",monospace;white-space:pre-wrap;word-break:break-word;color:var(--text)}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-thumb{background:color-mix(in srgb,var(--muted) 55%,transparent);border-radius:980px;border:2px solid transparent;background-clip:padding-box}
::-webkit-scrollbar-thumb:hover{background:var(--muted);background-clip:padding-box}
::-webkit-scrollbar-track{background:transparent}
#logbox .logp{color:#ff3b30}html[data-theme=dark] #logbox .logp{color:#ff453a}
#logbox .logt{color:#248a3d}html[data-theme=dark] #logbox .logt{color:#30d158}
#logbox .logsrc-vllm{color:#0a84ff}html[data-theme=dark] #logbox .logsrc-vllm{color:#64d2ff}
#logbox .logsrc-rollout{color:#af52de}html[data-theme=dark] #logbox .logsrc-rollout{color:#bf5af2}
#logbox .logsrc-megatron{color:#e08600}html[data-theme=dark] #logbox .logsrc-megatron{color:#ff9f0a}
#logbox .logsrc-ray{color:#8e8e93}html[data-theme=dark] #logbox .logsrc-ray{color:#98989d}
header{display:flex;align-items:center;gap:14px;padding:9px 22px;flex:0 0 auto;z-index:8;flex-wrap:wrap;border-bottom:.5px solid var(--line);
  box-shadow:0 1px 12px rgba(0,0,0,.04);
  background:var(--panel);background:color-mix(in srgb,var(--panel) 72%,transparent);backdrop-filter:saturate(180%) blur(20px);-webkit-backdrop-filter:saturate(180%) blur(20px)}
header h1{font-size:16px;margin:0;font-weight:700;letter-spacing:-.021em;
  background:linear-gradient(120deg,var(--text) 30%,var(--accent) 85%);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.secnav{display:flex;gap:4px;margin-left:6px;padding:3px;border:.5px solid var(--line);border-radius:980px;background:color-mix(in srgb,var(--panel) 60%,transparent)}
.secnav a{padding:4px 12px;border-radius:980px;font-size:11.5px;font-weight:600;color:var(--muted);text-decoration:none;letter-spacing:-.003em;transition:background .18s,color .18s}
.secnav a:hover{background:var(--accent);color:#fff}
@media (max-width:1180px){.secnav{display:none}}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:6px;vertical-align:middle}
.on{background:#34c759;box-shadow:0 0 0 3px rgba(52,199,89,.18);animation:pulse 2s ease-in-out infinite}
.off{background:#ff3b30;box-shadow:0 0 0 3px rgba(255,59,48,.18)}
@keyframes pulse{0%,100%{box-shadow:0 0 0 3px rgba(52,199,89,.18)}50%{box-shadow:0 0 0 6px rgba(52,199,89,.06)}}
.muted{color:var(--muted)}
/* ---- sticky KPI 总览条:滚动时核心指标常驻 ---- */
.kpis{position:sticky;top:0;z-index:6;display:flex;gap:10px;flex-wrap:wrap;padding:12px 22px;
  background:color-mix(in srgb,var(--bg) 84%,transparent);backdrop-filter:blur(18px) saturate(160%);-webkit-backdrop-filter:blur(18px) saturate(160%);
  box-shadow:0 1px 0 var(--line)}
.kpi{background:var(--panel);border:.5px solid var(--line);border-radius:14px;padding:9px 14px;min-width:112px;box-shadow:var(--shadow);position:relative;overflow:hidden;
  transition:box-shadow .25s ease,transform .25s ease,border-color .25s ease}
.kpi::before{content:"";position:absolute;inset:0 0 auto 0;height:2.5px;background:linear-gradient(90deg,var(--accent),#5e5ce6);opacity:.85}
.kpi:hover{box-shadow:var(--shadow-h);transform:translateY(-1px);border-color:color-mix(in srgb,var(--accent) 25%,var(--line))}
.kpi .v{font-size:20px;font-weight:700;letter-spacing:-.022em;line-height:1.15;font-variant-numeric:tabular-nums}
.kpi .l{font-size:10.5px;color:var(--muted);margin-top:2px;letter-spacing:-.004em;font-weight:500}
/* ---- 语义分区:每组一个色标 ---- */
.sec{padding:0 22px;margin-top:20px;scroll-margin-top:84px}
.sechead{display:flex;align-items:center;gap:10px;margin:0 2px 10px}
.sechead::after{content:"";flex:1;height:.5px;background:var(--line)}
.secbadge{display:inline-flex;align-items:center;gap:7px;font-size:12px;font-weight:700;letter-spacing:.02em;color:var(--text)}
.secbadge::before{content:"";width:8px;height:8px;border-radius:3px;background:var(--sa,var(--accent));box-shadow:0 0 0 3px color-mix(in srgb,var(--sa,var(--accent)) 18%,transparent)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(400px,1fr));gap:14px}
.card{background:var(--panel);border:.5px solid var(--line);border-radius:14px;padding:14px 16px;box-shadow:var(--shadow);
  transition:box-shadow .25s ease,transform .25s ease,border-color .25s ease,background .35s ease;animation:rise .45s ease both}
.card:hover{box-shadow:var(--shadow-h);transform:translateY(-1px);border-color:color-mix(in srgb,var(--sa,var(--accent)) 30%,var(--line))}
.card h2{font-size:12px;margin:0 0 10px;color:var(--muted);font-weight:600;letter-spacing:.01em;display:flex;align-items:center;gap:6px}
.card h2::after{content:"";flex:1;height:.5px;background:var(--line);margin-left:6px}
@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
canvas{width:100%;height:248px;display:block;border-radius:8px}
.bar{height:8px;background:var(--grid);border-radius:980px;overflow:hidden;margin:5px 0}
.bar>span{display:block;height:100%;border-radius:980px;transition:width .4s ease;box-shadow:inset 0 1px 0 rgba(255,255,255,.25)}
.row{display:flex;justify-content:space-between;align-items:center;font-size:12px;margin:3px 0;font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:12px}td{padding:2px 4px}
.tag{display:inline-block;padding:2px 10px;border-radius:980px;font-size:11px;font-weight:600;letter-spacing:-.003em}
th{font-size:11px;color:var(--muted);font-weight:600;text-align:left;padding:2px 4px;border-bottom:.5px solid var(--line)}
.pgbtn{cursor:pointer;background:var(--chipbg);color:var(--text);border:.5px solid var(--hair);border-radius:980px;padding:3px 10px;font-size:11px;line-height:1;font-weight:500;transition:background .2s,color .2s}
.pgbtn:hover:not([disabled]){background:var(--accent);color:#fff;border-color:transparent}
.pgbtn[disabled]{opacity:.35;cursor:default}
</style></head><body>
<header>
  <h1>🧪 RL Training Monitor</h1>
  <span id=alive></span>
  <span class=muted id=meta></span>
  <span id=uptime title="脚本从运行到现在的耗时(train_async.py 进程存活时长)"></span>
  <nav class=secnav>
    <a href="#sec-signal">🎯 信号</a><a href="#sec-train">📉 训练</a><a href="#sec-rollout">🌀 生成</a><a href="#sec-sys">🖥️ 系统</a>
  </nav>
  <span class=muted style="margin-left:auto" id=upd></span>
  <button id=theme title="切换 白天/黑夜">🌙</button>
</header>
<div id=main>
<div class=left>
<div class=kpis id=kpis></div>
<section class=sec style="--sa:#8e8e93">
  <div class=card><h2>🔁 重复算子 · 前后轮解题率对比 (≥2轮)</h2><div id=ops></div></div>
</section>
<section class=sec id=sec-signal style="--sa:#34c759">
  <div class=sechead><span class=secbadge>信号质量</span></div>
  <div class=grid>
    <div class=card><h2>reward (per rollout)</h2><canvas id=c_reward></canvas></div>
    <div class=card><h2>🧭 去噪 reward 曲线 (扣除 op 组成噪声 · MA7=1个epoch)</h2><canvas id=c_reward_adj></canvas><div id=reward_adj_stat class=muted style="margin-top:6px;font-size:11px;word-break:break-all"></div></div>
    <div class=card><h2>📚 每 epoch 平均 reward (epoch k = 每题的第 k 次出现 · 每题各一次 · 未满不画)</h2><canvas id=c_epoch_reward></canvas><div id=epoch_reward_stat class=muted style="margin-top:6px;font-size:11px;word-break:break-all"></div></div>
    <div class=card><h2>🔬 CoT 放开监视 · 训推一致 + 塌缩 + 护栏</h2><canvas id=c_cot></canvas><div id=cot_stat class=muted style="margin-top:6px;font-size:11px;word-break:break-all"></div></div>
  </div>
</section>
<section class=sec id=sec-train style="--sa:#0a84ff">
  <div class=sechead><span class=secbadge>训练健康 · 每 train step</span></div>
  <div class=grid>
    <div class=card><h2>loss / pg_loss</h2><canvas id=c_loss></canvas></div>
    <div class=card><h2>grad_norm (per train step)</h2><canvas id=c_grad></canvas></div>
    <div class=card><h2>entropy(每 train step · 防策略收窄)</h2><canvas id=c_entropy></canvas></div>
    <div class=card><h2>🎯 零方差组占比 (组内 reward 全等 → 零梯度 · 目标&lt;10%)</h2><canvas id=c_zerovar></canvas><div id=zerovar_stat class=muted style="margin-top:6px;font-size:11px;word-break:break-all"></div></div>
  </div>
</section>
<section class=sec id=sec-rollout style="--sa:#bf5af2">
  <div class=sechead><span class=secbadge>生成 / Rollout</span></div>
  <div class=grid>
    <div class=card><h2>序列长度 · 每 session 均值 · K tokens(整段 prompt→结束 / 仅生成)</h2><canvas id=c_len></canvas></div>
    <div class=card><h2>rollout_time (h) / staleness</h2><canvas id=c_time></canvas></div>
    <div class=card><h2>per-session decode tok/s 分布(单会话:Σtok / Σ请求延迟)</h2><canvas id=c_tps></canvas><div id=tps_stat class=muted style="margin-top:6px;font-size:11px;word-break:break-all"></div></div>
  </div>
</section>
<section class=sec id=sec-sys style="--sa:#ff9f0a">
  <div class=sechead><span class=secbadge>系统资源</span></div>
  <div class=grid>
    <div class=card><h2>🖥️ NPU load (算力 AICore% / 显存 HBM%) · Actor</h2><div id=npu></div></div>
    <div class=card><h2>🖥️ NPU load (算力 AICore% / 显存 HBM%) · Rollout</h2><div id=npu_peer></div></div>
    <div class=card><h2>🚀 本机 vLLM 推理引擎</h2><div id=local_vllm></div></div>
    <div class=card><h2>🧠 Host CPU memory · 训练阶段</h2><div id=host></div>
        <canvas id=c_hostmem style="height:140px;margin-top:8px" title="波谷逐轮抬升=泄漏;回到同一基线=尖峰"></canvas>
        <div id=hostmem_stat class=muted style="font-size:11px;margin-top:2px"></div>
        <h2 style="margin-top:12px">⚡ sglang prefill / decode</h2><div id=sglang></div></div>
  </div>
</section>
</div><!--left-->
<div class=right>
  <div class=loghead><span>📜 训练日志</span><span id=logfile class=muted>—</span><span id=logstat style="margin-left:auto"></span><button id=logbottom title="回到最新输出">⤓ 最新</button></div>
  <pre id=logbox>加载中…</pre>
</div>
</div><!--main-->
<script>
const $=id=>document.getElementById(id);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim()||'#888';
const theme=$('theme');
function setTheme(t){document.documentElement.setAttribute('data-theme',t);theme.textContent=t==='light'?'☀️':'🌙';localStorage.setItem('rl_theme',t);}
theme.onclick=()=>{setTheme(document.documentElement.getAttribute('data-theme')==='light'?'dark':'light');tick();};
setTheme(localStorage.getItem('rl_theme')||'light');
const CHARTS={},HOVER={};
let opPage=0,opPageRun=null;   // 重复算子表格:当前页 + 所属 run(换 run 归零)
const tv=v=>v==null?'—':(Math.abs(v)>=1?(+v).toFixed(3):(+v).toPrecision(3));
function draw(id,series,opt={}){
 const cv=$(id),dpr=devicePixelRatio||1;const W=cv.clientWidth,H=cv.clientHeight;
 cv.width=W*dpr;cv.height=H*dpr;const g=cv.getContext('2d');g.scale(dpr,dpr);
 g.clearRect(0,0,W,H);const pad={l:44,r:10,t:16,b:28};
 let all=[];series.forEach(s=>all=all.concat(s.y));
 if(!all.length){g.fillStyle='#484f58';g.fillText('waiting for data…',W/2-40,H/2);return;}
 let mn=Math.min(...all),mx=Math.max(...all);if(opt.min!=null)mn=Math.min(mn,opt.min);
 if(mn===mx){mx+=1;mn-=1;}const pd=(mx-mn)*0.08;mn-=pd;mx+=pd;
 let xs=[];series.forEach(s=>xs=xs.concat(s.x));const xmn=Math.min(...xs),xmx=Math.max(...xs);
 const px=x=>pad.l+(xmx===xmn?0.5:(x-xmn)/(xmx-xmn))*(W-pad.l-pad.r);
 const py=y=>pad.t+(1-(y-mn)/(mx-mn))*(H-pad.t-pad.b);
 // opt.plain: 只要横线的老样式(--grid 原色, 无竖线)
 const axy=H-pad.b,CG=css(opt.plain?'--grid':'--gridline'),CM=css('--muted');
 // x 轴刻度:先算出来,横竖网格线一起画
 const uxs=[...new Set(xs)].sort((a,b)=>a-b),xstep=Math.max(1,Math.ceil(uxs.length/6)),marks=[];
 for(let i=0;i<uxs.length;i+=xstep)marks.push(uxs[i]);
 if(marks[marks.length-1]!==uxs[uxs.length-1])marks.push(uxs[uxs.length-1]);
 // 网格:横线 +(非 plain 时)竖线
 g.strokeStyle=CG;g.fillStyle=CM;g.font='12px sans-serif';g.lineWidth=1;
 for(let i=0;i<=3;i++){const y=pad.t+i/3*(H-pad.t-pad.b);g.beginPath();g.moveTo(pad.l,y);g.lineTo(W-pad.r,y);g.stroke();
   g.fillText((mx-(mx-mn)*i/3).toPrecision(3),2,y+3);}
 if(!opt.plain)marks.forEach(x=>{const X=px(x);g.beginPath();g.moveTo(X,pad.t);g.lineTo(X,axy);g.stroke();});
 CHARTS[id]={series,opt,pad,W,xmn,xmx,uxs};
 if(!cv.__hk){cv.__hk=1;
   cv.onmousemove=e=>{const c=CHARTS[id];if(!c||!c.uxs.length)return;const r=cv.getBoundingClientRect();
     const dx=c.xmn+(e.clientX-r.left-c.pad.l)/(c.W-c.pad.l-c.pad.r)*(c.xmx-c.xmn);
     const nx=c.uxs.reduce((a,b)=>Math.abs(b-dx)<Math.abs(a-dx)?b:a,c.uxs[0]);
     if(HOVER[id]!==nx){HOVER[id]=nx;draw(id,c.series,c.opt);}};
   cv.onmouseleave=()=>{if(HOVER[id]!=null){HOVER[id]=null;draw(id,CHARTS[id].series,CHARTS[id].opt);}};}
 g.textAlign='center';g.fillStyle=CM;
 marks.forEach(x=>{g.fillText(x,px(x),axy+13);
   if(opt.plain){g.strokeStyle=CG;g.beginPath();g.moveTo(px(x),axy);g.lineTo(px(x),axy+3);g.stroke();}});
 g.fillStyle=CM;g.globalAlpha=.6;g.fillText(opt.xl||'rollout',pad.l+(W-pad.l-pad.r)/2,H-2);g.globalAlpha=1;g.textAlign='left';
 series.forEach(s=>{if(!s.y.length)return;g.strokeStyle=s.c;g.lineWidth=s.w||1.8;g.beginPath();
   s.x.forEach((x,i)=>{const X=px(x),Y=py(s.y[i]);i?g.lineTo(X,Y):g.moveTo(X,Y);});g.stroke();
   const lx=s.x[s.x.length-1],ly=s.y[s.y.length-1];g.fillStyle=s.c;g.beginPath();g.arc(px(lx),py(ly),2.5,0,7);g.fill();});
 // legend(顶部)
 let lgx=pad.l+2;series.forEach(s=>{if(!s.name)return;g.fillStyle=s.c;g.fillRect(lgx,5,8,3);
   g.fillStyle=CM;g.fillText(s.name,lgx+11,9);lgx+=g.measureText(s.name).width+26;});
 // 悬停:十字线 + tooltip(显示 rollout 号 + 各序列在该轮的值)
 const hx=HOVER[id];
 if(hx!=null){const X=px(hx);
   g.strokeStyle=CM;g.globalAlpha=.5;g.setLineDash([4,3]);g.beginPath();g.moveTo(X,pad.t);g.lineTo(X,axy);g.stroke();g.setLineDash([]);g.globalAlpha=1;
   const rows=[{c:CM,t:(opt.xl||'rollout')+' '+hx}];
   series.forEach(s=>{const i=s.x.indexOf(hx);if(i<0)return;const Y=py(s.y[i]);
     g.fillStyle=s.c;g.beginPath();g.arc(X,Y,4,0,7);g.fill();g.strokeStyle=css('--bg');g.lineWidth=1.5;g.stroke();
     rows.push({c:s.c,t:(s.name?s.name+': ':'')+tv(s.y[i])+(opt.unit||'')});});
   g.font='12px sans-serif';g.textAlign='left';
   const tw=Math.max(...rows.map(r=>g.measureText(r.t).width))+22,th=rows.length*17+8;
   let bx=X+12;if(bx+tw>W-2)bx=X-tw-12;if(bx<2)bx=2;const by=pad.t+2;
   g.fillStyle=css('--panel');g.globalAlpha=.97;g.fillRect(bx,by,tw,th);g.globalAlpha=1;
   g.strokeStyle=CG;g.lineWidth=1;g.strokeRect(bx,by,tw,th);
   rows.forEach((r,k)=>{const y=by+16+k*17;if(k>0){g.fillStyle=r.c;g.fillRect(bx+7,y-8,9,3);}g.fillStyle=r.c;g.fillText(r.t,bx+(k>0?20:8),y);});}
}
const fmt=(v,d=3)=>v==null?'—':(+v).toFixed(d);
const fmtDur=s=>{s=Math.max(0,Math.floor(s));const d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60),ss=s%60;
  return d?`${d}d ${h}h ${m}m`:h?`${h}h ${m}m ${ss}s`:m?`${m}m ${ss}s`:`${ss}s`;};
const barColor=p=>p>90?'#ff3b30':p>70?'#ff9f0a':'#34c759';
// per-session tok/s 直方图(柱状图 + 中位垂直线)
function drawHist(id,vals){
 const cv=$(id),dpr=devicePixelRatio||1;const W=cv.clientWidth,H=cv.clientHeight;
 cv.width=W*dpr;cv.height=H*dpr;const g=cv.getContext('2d');g.scale(dpr,dpr);
 g.clearRect(0,0,W,H);const pad={l:36,r:10,t:12,b:26};
 if(!vals||!vals.length){g.fillStyle='#484f58';g.font='12px sans-serif';g.fillText('waiting for data…',W/2-40,H/2);return;}
 const mn=Math.min(...vals),mx=Math.max(...vals),lo=mn,hi=mx>mn?mx:mn+1;
 const NB=Math.min(24,Math.max(6,Math.ceil(Math.sqrt(vals.length)))),bw=(hi-lo)/NB;
 const bins=new Array(NB).fill(0);
 vals.forEach(v=>{let b=Math.floor((v-lo)/bw);if(b<0)b=0;if(b>=NB)b=NB-1;bins[b]++;});
 const bmax=Math.max(...bins,1),CG=css('--grid'),CM=css('--muted');
 g.strokeStyle=CG;g.fillStyle=CM;g.font='11px sans-serif';g.lineWidth=1;
 for(let i=0;i<=3;i++){const y=pad.t+i/3*(H-pad.t-pad.b);g.beginPath();g.moveTo(pad.l,y);g.lineTo(W-pad.r,y);g.stroke();
   g.fillText(Math.round(bmax-bmax*i/3),2,y+3);}
 const plotW=W-pad.l-pad.r,plotH=H-pad.t-pad.b,axy=H-pad.b;
 for(let i=0;i<NB;i++){const x=pad.l+i/NB*plotW,w=plotW/NB-1,h=bins[i]/bmax*plotH;
   g.fillStyle='#0a84ff';g.fillRect(x,axy-h,Math.max(1,w),h);}
 const med=vals[Math.floor(vals.length/2)],mx_=pad.l+(hi>lo?(med-lo)/(hi-lo):0.5)*plotW;
 g.strokeStyle='#ff9500';g.setLineDash([4,3]);g.beginPath();g.moveTo(mx_,pad.t);g.lineTo(mx_,axy);g.stroke();g.setLineDash([]);
 g.fillStyle='#ff9500';g.font='11px sans-serif';g.fillText('中位 '+med,Math.min(mx_+3,W-pad.r-46),pad.t+10);
 g.fillStyle=CM;g.textAlign='center';[0,0.5,1].forEach(f=>g.fillText((lo+f*(hi-lo)).toFixed(0),pad.l+f*plotW,axy+13));
 g.textAlign='left';g.globalAlpha=.6;g.fillText('tok/s',pad.l+plotW/2-12,H-2);g.globalAlpha=1;
}
// reward 迷你趋势线(host 面板内,无轴)
function drawSpark(id,vals){
 const cv=$(id);if(!cv)return;const dpr=devicePixelRatio||1,W=cv.clientWidth,H=cv.clientHeight;
 if(!W||!H)return;cv.width=W*dpr;cv.height=H*dpr;const g=cv.getContext('2d');g.scale(dpr,dpr);g.clearRect(0,0,W,H);
 if(!vals||vals.length<2){g.fillStyle='#98989d';g.font='11px sans-serif';g.fillText('reward 趋势 · 待数据',6,H/2+3);return;}
 const mn=Math.min(...vals),mx=Math.max(...vals),pd=6,x=i=>pd+i/(vals.length-1)*(W-2*pd),y=v=>H-pd-((v-mn)/((mx-mn)||1))*(H-2*pd);
 g.beginPath();g.moveTo(x(0),H);vals.forEach((v,i)=>g.lineTo(x(i),y(v)));g.lineTo(x(vals.length-1),H);g.closePath();
 const gr=g.createLinearGradient(0,0,0,H);gr.addColorStop(0,'rgba(10,132,255,.22)');gr.addColorStop(1,'rgba(10,132,255,0)');g.fillStyle=gr;g.fill();
 g.strokeStyle='#0a84ff';g.lineWidth=1.8;g.beginPath();vals.forEach((v,i)=>{const X=x(i),Y=y(v);i?g.lineTo(X,Y):g.moveTo(X,Y);});g.stroke();
 g.fillStyle='#0a84ff';g.beginPath();g.arc(x(vals.length-1),y(vals[vals.length-1]),2.5,0,7);g.fill();
 g.fillStyle='#98989d';g.font='10px sans-serif';g.fillText('reward 趋势',6,11);
}
async function tick(){
 let m;try{m=await (await fetch('/api/metrics')).json();}catch(e){$('upd').textContent='fetch error';return;}
 const meta=m.meta;
 $('alive').innerHTML=`<span class="dot ${meta.alive?'on':'off'}"></span>${meta.alive?'running':'stopped'}`;
 $('meta').textContent=`${meta.log_file} · rollout ${meta.rollout_step??'—'} · ${meta.n_rollouts} rollouts / ${meta.n_train_steps} train steps`;
 $('uptime').innerHTML = meta.uptime!=null
   ? `⏱ 已运行 <b>${fmtDur(meta.uptime)}</b>` : '';
 $('upd').textContent='updated '+meta.updated_at;
 const R=m.rollouts,T=m.train;
 $('kpis').innerHTML=[
   ['reward_mean',fmt(meta.latest_reward)],['success',fmt(meta.latest_success,2)],
   ['解对/session',`${(m.solved_stats||{}).solved||0}/${(m.solved_stats||{}).total||0}`],
   ['grad_norm',fmt(last(T.grad_norm),2)],['loss',fmt(last(T.loss),4)],
   ['整段seq(K)',kK(last(R.total_len_mean))],['生成resp(K)',kK(last(R.sess_resp_len_mean))],['rollout_time(h)',fmt(last(R.rollout_time)&&last(R.rollout_time)/3600,2)],
   // ── vllm 推理性能(140 exporter → 引擎 /metrics 聚合)──
   ...(()=>{const eng=(((m.peer||{}).data||{}).engines)||[];
     const agg=(key,keep)=>{let sv=0,sn=0;for(const e of eng){const s=e[key];if(s&&s.n>0&&(!keep||keep(s))){sv+=s.mean*s.n;sn+=s.n;}}return sn?sv/sn:null;};
     const ttft=agg('ttft'), tpot=agg('tpot',s=>s.mean>0), qt=agg('queue_t');
     const run=eng.reduce((a,e)=>a+(e.running||0),0), wait=eng.reduce((a,e)=>a+(e.waiting||0),0);
     const fmtS=v=>v==null?'—':(v<1?(v*1000).toFixed(0)+'ms':v.toFixed(1)+'s');
     if(!eng.length)return [['推理延迟','离线']];
     return [['TTFT',fmtS(ttft)],['TPOT',fmtS(tpot)],['排队耗时',fmtS(qt)],['eng运行/排队',`${run}/${wait}`]];})(),
   ['turns(mean)',m.operators.mean_turns],
 ].map(([l,v])=>`<div class=kpi><div class=v>${v}</div><div class=l>${l}</div></div>`).join('');
 draw('c_reward',[S(R.reward_std,'#8b949e','std'),S(R.reward_mean,'#0a84ff','mean'),S(R.reward_completed,'#34c759','completed'),MA(R.reward_mean,5,'#ff9500','MA'+MAW)]);
 // 每 epoch 平均 reward(只画完整 epoch)
 const ER=((m.health||{}).epoch_reward_adj)||{x:[],raw:[],span:[],N:0};
 draw('c_epoch_reward',[S({x:ER.x,y:ER.raw},'#0a84ff','epoch平均reward')],{xl:'epoch(覆盖全训练集一遍)'});
 const _sp=(ER.span&&ER.span.length)?ER.span[ER.span.length-1]:null;
 $('epoch_reward_stat').innerHTML=(ER.x&&ER.x.length)?
   `已完成 <b>${ER.x.length}</b> 个 epoch(每题各出现1次 · 全 <b>${ER.N}</b> 题)· 最新 <b>${fmt(ER.raw[ER.raw.length-1],3)}</b> · 全程Δ <b style="color:${(ER.slope||0)>=0?'#34c759':'#ff3b30'}">${(ER.slope>=0?'+':'')+fmt(ER.slope,3)}</b>${_sp?` · 第${ER.x[ER.x.length-1]}次出现分布在 rollout ${_sp[0]}-${_sp[1]}`:''}`
   :`尚无完整 epoch(需先把全 ${ER.N||50} 题各跑过一遍)`;
 draw('c_entropy',[S(T.entropy_loss,'#bf5af2','entropy'),MA(T.entropy_loss,5,'#ff9500','MA5')],{xl:'train step'});
 // CoT 放开监视:训推一致(|Δlogprob|_masked / |TIS-1|_masked)曲线 + 塌缩/护栏状态行
 draw('c_cot',[S(T.logprob_abs_diff_masked,'#ff2d55','|Δlogprob|_masked'),S(T.tis_abs_masked,'#ff9f0a','|TIS-1|_masked'),MA(T.logprob_abs_diff_masked,5,'#0a84ff','MA5')],{xl:'train step',min:0});
 {const ld=last(T.logprob_abs_diff_masked),ir=meta.integrity_rejects||0,en=last(T.entropy_loss),rp=last(R.repetition);
  const cD=ld==null?'#86868b':(ld>0.06?'#ff3b30':ld>0.04?'#ff9f0a':'#34c759');
  const cI=ir>0?'#ff3b30':'#34c759';
  const cE=en==null?'#86868b':(en<0.05?'#ff3b30':en<0.06?'#ff9f0a':'#34c759');
  $('cot_stat').innerHTML=`训推 |Δlogprob|_masked <b style="color:${cD}">${fmt(ld,4)}</b> <span class=muted>(基线≈0.019 · 放开CoT跳高=推理段训推不一致)</span> · `
   +`护栏 INTEGRITY-REJECT <b style="color:${cI}">${ir}</b> <span class=muted>(&gt;0=坏logprob进训练)</span> · `
   +`entropy <b style="color:${cE}">${fmt(en,3)}</b> <span class=muted>(跌破~0.05预警塌缩)</span> · `
   +`repetition <b style="color:${(rp||0)>0.02?'#ff9f0a':'#34c759'}">${fmt(rp,3)}</b>`;}
 draw('c_grad',[S(T.grad_norm,'#ff9f0a')],{xl:'train step'});
 draw('c_loss',[S(T.loss,'#0a84ff','loss'),S(T.pg_loss,'#bf5af2','pg_loss')],{xl:'train step'});
 draw('c_len',[SK(R.total_len_mean,'#ff9500','整段 total(prompt→结束)'),SK(R.sess_resp_len_mean,'#0a84ff','仅生成 response')],{unit:'K'});
 draw('c_time',[SH(R.rollout_time,'#bf5af2','time(h)'),S(R.staleness,'#34c759','staleness')]);
 // 信号质量健康面板:零方差组占比
 const H=m.health||{},zv=H.zero_var||{x:[],frac:[],reward_mean:[]};
 draw('c_zerovar',[S({x:zv.x,y:zv.frac},'#ff2d55','零方差组占比'),MA({x:zv.x,y:zv.frac},5,'#ff9500','MA'+MAW),S({x:zv.x,y:zv.reward_mean},'#8b949e','reward均值')],{min:0,xl:'rollout'});
 // 去噪 reward 曲线:raw(灰) vs op组成校正(蓝) vs MA7趋势(绿)
 const RA=H.reward_adj||{x:[],raw:[],adj:[],ma:[]};
 draw('c_reward_adj',[S({x:RA.x,y:RA.adj},'#0a84ff','op校正(session口径)'),S({x:RA.x,y:RA.ma},'#34c759','MA7 趋势')],{xl:'rollout'});
 $('reward_adj_stat').innerHTML=(RA.x&&RA.x.length)?
   `最新 MA7 <b>${fmt(RA.ma[RA.ma.length-1],3)}</b> · 全程Δ <b style="color:${(RA.slope||0)>=0?'#34c759':'#ff3b30'}">${(RA.slope>=0?'+':'')+fmt(RA.slope,3)}</b> · 已扣除~70%抽样噪声;仍为训练集/固定输入趋势,非泛化`
   :'暂无 session 数据';
 $('zerovar_stat').innerHTML=H.n_sessions?
   `sessions <b>${H.n_sessions}</b> · 最新占比 <b style="color:${(H.latest_zero_var_frac||0)>0.1?'#ff3b30':'#34c759'}">${fmt(H.latest_zero_var_frac,3)}</b> (目标&lt;0.10) · 组数 ${zv.n_groups?zv.n_groups[zv.n_groups.length-1]:'—'}`
   +` · top错误 ${(H.errors||[]).map(e=>e[0]+':'+e[1]).join(' | ')||'—'}`
   :'暂无 session 数据';
 // per-session decode tok/s 分布 + sglang 对比
 const tp=m.session_tps||{};
 drawHist('c_tps',tp.values||[]);
 $('tps_stat').innerHTML = tp.n ?
   `n=${tp.n} sessions · p10 <b>${tp.p10}</b> · 中位 <b>${tp.p50}</b> · p90 <b>${tp.p90}</b> · mean ${tp.mean} · [${tp.min}, ${tp.max}] tok/s`
   +(tp.capped?` · <span style="color:#ff9f0a">仅最近${tp.n}个(略${tp.capped}个)</span>`:'')
   +`　│　对比 sglang decode 聚合 ≈ <b>${fmt(m.sglang&&m.sglang.throughput,0)}</b> tok/s`
   : '<span>暂无 completion_metrics 数据</span>';
 // NPU
 $('npu').innerHTML=m.npu.map(n=>{const hp=n.hbm_total?100*n.hbm_used/n.hbm_total:0,ap=n.aicore||0;
   return `<div class=row><b>NPU ${n.id}</b><span class=muted>AICore ${ap}% · HBM ${hp.toFixed(0)}% (${(n.hbm_used/1024).toFixed(1)}/${(n.hbm_total/1024).toFixed(0)}GB) · ${n.temp}°C</span></div>
   <div class=bar title="HBM 显存占比 ${hp.toFixed(0)}%"><span style="width:${hp}%;background:${barColor(hp)}"></span></div>`;}).join('')||'<span class=muted>npu-smi 无数据</span>';
 // peer node (140,经 tools/node_exporter.py)
 const pr=m.peer||{};
 $('npu_peer').innerHTML=(()=>{
   if(!pr.ok)return `<span class=muted>140 exporter 离线(${pr.err||'未配置'}) · 在 140 起:python3 tools/node_exporter.py</span>`;
   const d=pr.data||{},mem=d.mem||{},cpu=d.cpu_pct==null?'—':d.cpu_pct+'%';
   const head=`<div class=row><b>host ${d.host||'140'}</b><span class=muted>CPU ${cpu} · 内存 ${mem.used_gb??'—'}/${mem.total_gb??'—'}GB${pr.stale?` · <span style="color:#ff9f0a">数据 ${pr.age}s 前</span>`:''}</span></div>`;
   const rows=(d.npu||[]).map(n=>{const hp=n.hbm_total?100*n.hbm_used/n.hbm_total:0,ap=n.aicore||0;
     return `<div class=row><b>NPU ${n.id}</b><span class=muted>AICore ${ap}% · HBM ${hp.toFixed(0)}% (${(n.hbm_used/1024).toFixed(1)}/${(n.hbm_total/1024).toFixed(0)}GB) · ${n.temp}°C</span></div>
     <div class=bar><span style="width:${hp}%;background:${barColor(hp)}"></span></div>`;}).join('');
   // vllm 推理性能(来自 140 exporter 采的 /metrics)
   const eng=(d.engines||[]);
   const engRows = eng.length ? eng.map(e=>{
     const t=e.ttft||{}, o=e.tpot||{}, q=e.queue_t||{}, ee=e.e2e||{};
     const tpotTxt = o.mean>0 ? ` · TPOT ${o.mean}s` : '';
     const qTxt = q.mean!=null ? ` · 排队 ${q.mean}s` : '';
     const e2eTxt = ee.mean!=null ? ` · e2e ${ee.mean}s` : '';
     const kv = e.kv_usage!=null ? ` · KV ${(e.kv_usage*100).toFixed(0)}%` : '';
     const phit = e.prefix_hit!=null ? ` · 命中 ${(e.prefix_hit*100).toFixed(0)}%` : '';
     const w = (e.waiting||0) > 0 ? ` · <span style="color:#ff9f0a">等待 ${e.waiting}</span>` : '';
     return `<div class=row><b>eng :${e.port}</b><span class=muted>TTFT ${t.mean??'—'}s(p90 ${t.p90??'—'})${tpotTxt}${qTxt}${e2eTxt}${kv}${phit} · 运行 ${e.running??0}${w}</span></div>`;
   }).join('') : '';
   const engSec = engRows ? `<div class=row style="margin-top:6px"><span class=muted>── vllm 推理性能(引擎 /metrics)──</span></div>` + engRows : '';
   return head+rows+engSec||'<span class=muted>140 无 npu 数据</span>';
 })();
 // 本机 vLLM：复用 :5000 N-engine monitor 的缓存，不直接轮询 engine。
 const lv=m.local_vllm||{};
 $('local_vllm').innerHTML=(()=>{
   if(!lv.ok)return `<span class=muted>本机指标聚合器离线(${lv.err||'未配置'}) · 启动 scripts/vllm_metrics_monitor_v2.py</span>`;
   const d=lv.data||{}, c=d.combined||{}, engines=d.engines||[];
   const stale=lv.stale?` · <span style="color:#ff9f0a">数据 ${lv.age}s 前</span>`:'';
   const head=`<div class=row><span><span class="dot ${(d.engines_alive||0)>0?'on':'off'}"></span><b>${d.engines_alive||0}/${d.engines_total||0} engines</b></span><span class=muted>更新 ${d.last_update||'—'}${stale}</span></div>
     <div class=row><span>总吞吐 <b>${fmt(c.throughput,1)} tok/s</b></span><span class=muted>平均 ${fmt(d.avg_per_engine,1)}/engine · 运行/排队 ${c.running||0}/${c.waiting||0}</span></div>`;
   const rows=engines.map(e=>{
     const port=(e.key||'').split(':').pop()||'—';
     const wait=(e.waiting||0)>0?`<span style="color:#ff9f0a">${e.waiting}</span>`:'0';
     return `<div class=row><span><span class="dot ${e.alive?'on':'off'}"></span>eng :${port}</span><span class=muted>${fmt(e.throughput,1)} tok/s · 运行/排队 ${e.running||0}/${wait}</span></div>`;
   }).join('');
   return head+(rows||'<div class=muted style="margin-top:6px">尚未发现本机 engine</div>');
 })();
 // host
 const ph=m.phase||{};
 const pbadge=(on,txt,col)=>on?`<span class=tag style="background:${col}22;color:${col};margin-right:4px">${txt}</span>`:'';
 const rlab=(ph.rollout!=null?`rollout <b>${ph.rollout}</b>${ph.total?' / '+ph.total:''}`:'rollout —')+(ph.eval?' <span class=tag style="background:#bf5af233;color:#bf5af2;margin-left:5px;font-weight:600">🔬 验证中</span>':'');
 const pcell=(ph.phases&&ph.phases.length)?(pbadge(ph.weight_update,'权重更新','#ff3b30')+pbadge(ph.training,'训练','#34c759')+pbadge(ph.generating,'生成','#0a84ff')):'<span class=muted>idle</span>';
 const fmtT=s=>s==null?'—':(s>=3600?(s/3600).toFixed(1)+'h':s>=60?Math.round(s/60)+'m':Math.round(s)+'s');
 const pc=v=>v==null?'—':(v*100).toFixed(v>0&&v<0.1?1:0)+'%';
 const el=ph.elapsed||{};
 const rd=(R.reward_mean&&R.reward_mean.y)||[];const rl=rd.length?rd[rd.length-1]:null,rp=rd.length>1?rd[rd.length-2]:null;
 const arr=(rl!=null&&rp!=null)?(rl>rp?'<span style="color:#34c759">▲</span>':rl<rp?'<span style="color:#ff3b30">▼</span>':'<span class=muted>→</span>'):'';
 const std=last(R.reward_std),comp=last(R.reward_completed),succ=last(R.success_rate);
 const ent=last(T.entropy_loss),kl=last(T.kl_loss),pk=last(T.ppo_kl),clip=last(T.pg_clipfrac),gn=last(T.grad_norm);
 const trunc=last(R.truncated),rep=last(R.repetition),stale=last(R.staleness),tot=last(R.total_len_mean);
 const stdw=(std!=null&&std<0.005)?' style="color:#ff9f0a" title="接近零方差:GRPO 无学习信号"':'';
 const h=m.host;const thrPct=h.total_gb?100*h.thr_gb/h.total_gb:99.5;
 $('host').innerHTML=`<div class=row style="margin-bottom:4px" title="open timers: ${(ph.open_timers||[]).join(', ')||'—'}"><span>${rlab} &nbsp; ${pcell}</span><span class=muted>本轮 ${fmtT(ph.since_round_sec)} · 均 ${fmtT(ph.mean_round_sec)} · ETA ${fmtT(ph.eta_sec)}</span></div>
   <canvas id=c_rspark style="height:44px;margin:2px 0 6px"></canvas>
   <div class=row style="font-size:11px"><span class=muted>reward</span><span>${fmt(rl)} ${arr} · std <b${stdw}>${fmt(std,3)}</b> · 完成 ${fmt(comp,3)} · 成功 ${pc(succ)}</span></div>
   <div class=row style="font-size:11px"><span class=muted>策略更新</span><span>ent ${fmt(ent,3)} · kl ${fmt(kl,3)} · ppo_kl ${fmt(pk,3)} · clip ${pc(clip)} · grad ${fmt(gn,2)}</span></div>
   <div class=row style="font-size:11px"><span class=muted>生成质量</span><span>截断 ${pc(trunc)} · 重复 ${pc(rep)} · staleness ${fmt(stale,2)} · 整段 ${kK(tot)}</span></div>
   <div class=row style="font-size:11px;margin-bottom:6px"><span class=muted>各阶段(上轮)</span><span>生成 ${fmtT(el.train_wait)} · 训练 ${fmtT(el.train)} · 权重 ${fmtT(el.update_weights)}</span></div>
   <div class=row><span><span class=tag style="background:#0a84ff22;color:#0a84ff;margin-right:4px">141</span>${h.used_gb} / ${h.total_gb} GB</span><span class=muted>${h.pct}% · 峰值 ${h.peak_gb} · 余量 ${h.margin_gb}G</span></div>
   <div class=bar style="position:relative"><span style="width:${h.pct}%;background:${barColor(h.pct)}"></span><i style="position:absolute;top:0;bottom:0;left:${thrPct}%;width:2px;background:#ff3b30;opacity:.7" title="Ray OOM 阈值 ${h.thr_gb}G"></i></div>
   ${(()=>{const q=m.peer||{};if(!q.ok)return `<div class=row style="margin-top:4px"><span class=tag style="background:#bf5af222;color:#bf5af2;margin-right:4px">140</span><span class=muted>exporter 离线(${q.err||''})</span></div>`;
     const mm=(q.data||{}).mem||{};const p2=mm.total_gb?100*mm.used_gb/mm.total_gb:0;
     return `<div class=row style="margin-top:4px"><span><span class=tag style="background:#bf5af222;color:#bf5af2;margin-right:4px">140</span>${mm.used_gb} / ${mm.total_gb} GB</span><span class=muted>${p2.toFixed(1)}%${q.stale?` · ${q.age}s前`:''}</span></div>
     <div class=bar><span style="width:${p2}%;background:#bf5af2"></span></div>`;})()}`;
 drawSpark('c_rspark',rd);
 // Host RAM 历史:泄漏看波谷是否抬升(尖峰会回到同一基线)
 if(h.hist_x&&h.hist_x.length>1){
   draw('c_hostmem',[{x:h.hist_x,y:h.hist_y,c:'#0a84ff',name:'used GB'}],{xl:'分钟',plain:true});
   const n=h.hist_y.length,mid=Math.floor(n/2);
   const lo1=Math.min(...h.hist_y.slice(0,mid)),lo2=Math.min(...h.hist_y.slice(mid));
   const d=lo2-lo1,span=h.hist_x[n-1]-h.hist_x[0];
   const tag=span<10?'<span class=muted>样本不足,继续观察</span>'
     :d>1?`<span style="color:#ff9f0a">波谷 +${d.toFixed(1)}G ↑ 疑似泄漏</span>`
     :d<-1?`<span style="color:#34c759">波谷 ${d.toFixed(1)}G ↓ 在回收</span>`
     :`<span style="color:#34c759">波谷持平 ${d>=0?'+':''}${d.toFixed(1)}G · 无泄漏迹象</span>`;
   $('hostmem_stat').innerHTML=`近 ${span.toFixed(0)} 分钟 · 波谷 ${lo1.toFixed(1)}G → ${lo2.toFixed(1)}G · ${tag}`;
 }else $('hostmem_stat').textContent='采样中…(每 15s 一点)';
 // sglang
 const s=m.sglang;$('sglang').innerHTML=s.running_req!=null?
   `<div class=row><span>running-req</span><b>${s.running_req}</b></div>
    <div class=row><span>token usage</span><b>${(s.token_usage*100||0).toFixed(0)}%</b></div>
    <div class=row><span>throughput</span><b>${fmt(s.throughput,1)} tok/s</b></div>
    <div class=row><span>queue-req</span><b>${s.queue_req??0}</b></div>
    ${s.prefill_newtok!=null?`<div class=row><span>prefill new/cached</span><b>${s.prefill_newtok}/${s.prefill_cached}</b></div>`:''}`
   :'<span class=muted>暂无 decode 记录</span>';
 // 重复算子(≥2轮): 表头色块=提升/下降/持平数, 行=各轮解题率+首轮→最新轮Δ, 分页(块高恒定)
 const OR=m.op_rounds||{},orows=OR.rows||[];
 const PS=10,np=Math.max(1,Math.ceil(orows.length/PS));
 if(OR.polar_run!==opPageRun){opPageRun=OR.polar_run;opPage=0;}
 opPage=Math.min(opPage,np-1);
 const rateC=v=>v>=0.75?'#34c759':v>=0.4?'#ff9f0a':'#ff3b30';
 const seg=r=>`r${r.r} <b style="color:${rateC(r.rate)}">${Math.round(r.rate*100)}%</b><span class=muted style="font-size:10px"> ${r.solved}/${r.n}</span>`;
 const segT=r=>`r${r.r} ${Math.round(r.rate*100)}%(${r.solved}/${r.n})`;
 const tb=orows.slice(opPage*PS,opPage*PS+PS).map(o=>{
   const d=o.trend,dc=d>0.05?'#34c759':d<-0.05?'#ff3b30':'#86868b';
   const dt=(d>0.05?'+':d<-0.05?'-':'±')+Math.round(Math.abs(d)*100)+'pp';
   const tip=`趋势(全轮最小二乘)${dt} · 斜率${(o.slope>=0?'+':'')}${Math.round(o.slope*100)}pp/次复现 · 原始首末Δ${(o.delta>=0?'+':'')}${Math.round(o.delta*100)}pp · 单调性${o.mono}(+1一路升/-1一路降)`;
   return `<tr><td style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="${o.full}">${o.op}</td>`
    +`<td style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="${o.rounds.map(segT).join(' → ')}">${o.rounds.map(seg).join('<span class=muted> → </span>')}</td>`
    +`<td style="text-align:right;font-weight:700;color:${dc};white-space:nowrap" title="${tip}">${dt}</td></tr>`;});
 if(!orows.length)tb.push(`<tr><td colspan=3 class=muted style="padding:6px 4px">当前 run 尚无被跑满两轮的算子</td></tr>`);
 while(tb.length<PS)tb.push('<tr><td>&nbsp;</td><td></td><td></td></tr>');   // 填充行:块高不随页变化
 $('ops').innerHTML=`<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:12px;margin-bottom:8px">
   <span class=tag style="background:#34c75922;color:#34c759">提升 <b>${OR.improved||0}</b></span>
   <span class=tag style="background:#ff3b3022;color:#ff3b30">下降 <b>${OR.declined||0}</b></span>
   <span class=tag style="background:#86868b22;color:#86868b">持平 <b>${OR.flat||0}</b></span>
   <span class=muted style="font-size:11px" title="同一算子在≥2个 rollout 轮被重复采样;趋势=对该题各轮解题率按出现次序做最小二乘拟合的首→末变化(用上所有中间轮,比只看首尾两点稳);±5pp 内算持平。鼠标悬停单元格看斜率/原始Δ/单调性">重复 <b style="color:var(--text)">${OR.repeated||0}</b>/${OR.total_ops||0} 算子 · 趋势=全轮最小二乘(±5pp死区)</span>
   <span style="margin-left:auto;display:flex;gap:6px;align-items:center">
     <button class=pgbtn id=opPrev ${opPage<=0?'disabled':''}>‹ 上一页</button>
     <span class=muted style="font-size:11px">${orows.length?opPage+1:0}/${np} 页</span>
     <button class=pgbtn id=opNext ${opPage>=np-1?'disabled':''}>下一页 ›</button></span>
 </div>
 <table style="table-layout:fixed"><thead><tr>
   <th style="width:215px">算子</th><th>各轮解题率 (r=rollout 轮)</th><th style="width:72px;text-align:right" title="全轮最小二乘趋势">趋势</th>
 </tr></thead><tbody>${tb.join('')}</tbody></table>`;
 $('opPrev').onclick=()=>{if(opPage>0){opPage--;tick();}};
 $('opNext').onclick=()=>{if(opPage<np-1){opPage++;tick();}};
}
const last=s=>s&&s.y&&s.y.length?s.y[s.y.length-1]:null;
const S=(s,c,name)=>({x:(s&&s.x)||[],y:(s&&s.y)||[],c,name});
const SK=(s,c,name)=>({x:(s&&s.x)||[],y:((s&&s.y)||[]).map(v=>v/1000),c,name}); // →K
const SH=(s,c,name)=>({x:(s&&s.x)||[],y:((s&&s.y)||[]).map(v=>v/3600),c,name}); // →小时
const kK=v=>v==null?'—':(v/1000).toFixed(1)+'K';
const MAW=5;  // 滑动平均窗口
const MA=(s,w,c,name)=>{const y=(s&&s.y)||[],x=(s&&s.x)||[],o=[];let q=[],sum=0;
  for(let i=0;i<y.length;i++){q.push(y[i]);sum+=y[i];if(q.length>w)sum-=q.shift();o.push(sum/q.length);}
  return{x,y:o,c,name,w:2.6};};  // 更粗、亮橙,压噪看趋势
// ------------------------ 训练日志(右栏)------------------------
const logbox=$('logbox');
// 逐行着色:最左 (Actor …) 按来源着色 —— VLLMEngine 蓝 / RolloutManager 紫 /
// MegatronTrainRayActor 橙 / raylet 灰 / 其他带 pid 的红;时间戳 [YYYY-MM-DD HH:MM:SS …] 绿。
// 用 span+textContent 构造(不拼 innerHTML),日志里的 < { 等字符安全。
// 兼容三种形态:(Name pid=N)、(Name pid=N, ip=…)、(raylet)。
const LOG_PID=/^\(([A-Za-z_]\w*)[^)]*\)/, LOG_TS=/^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[^\]]*\]/;
const LOG_SRC={VLLMEngine:'logsrc-vllm',RolloutManager:'logsrc-rollout',MegatronTrainRayActor:'logsrc-megatron',raylet:'logsrc-ray'};
const logSrcCls=n=>LOG_SRC[n]||(n&&n.indexOf('ray')===0?'logsrc-ray':'logp');
const LS=(cls,txt)=>{const e=document.createElement('span');e.className=cls;e.textContent=txt;return e;};
function addLogLine(frag,line){
  let s=line;const mp=s.match(LOG_PID);
  if(mp){frag.appendChild(LS(logSrcCls(mp[1]),mp[0]));s=s.slice(mp[0].length);}
  const lead=(s.match(/^\s*/)||[''])[0],mt=s.slice(lead.length).match(LOG_TS);
  if(mt){if(lead)frag.appendChild(document.createTextNode(lead));
    frag.appendChild(LS('logt',mt[0]));s=s.slice(lead.length+mt[0].length);}
  if(s)frag.appendChild(document.createTextNode(s));
}
function logFrag(text){const f=document.createDocumentFragment(),ls=text.split('\n');
  for(let i=0;i<ls.length;i++){addLogLine(f,ls[i]);if(i<ls.length-1)f.appendChild(document.createTextNode('\n'));}
  return f;}
let logFile=null,logTop=0,logEnd=0,logSize=0,logAtStart=false,logLoading=false,logFollow=true;
const atBottom=()=>logbox.scrollHeight-logbox.scrollTop-logbox.clientHeight<30;
function logStat(){
  $('logstat').textContent=(logFollow?'🟢 跟随最新':'⏸ 已暂停 · 下拉到底恢复')+(logSize?' · '+(logSize/1024).toFixed(0)+'KB':'');
}
async function logGet(q){try{return await(await fetch('/api/log'+q)).json();}catch(e){return null;}}
async function loadTail(){
  const d=await logGet('?tail=1');if(!d)return;
  logFile=d.file;logTop=d.start||0;logEnd=d.end||0;logSize=d.size||0;logAtStart=!!d.at_start;
  $('logfile').textContent=d.file||'(无日志)';
  logbox.textContent='';
  if(d.text)logbox.appendChild(logFrag(d.text));else logbox.textContent='(暂无日志输出)';
  logbox.scrollTop=logbox.scrollHeight;logFollow=true;logStat();
}
// [2026-08-31] 日志 DOM 上限:pollLog 只增不删,长跑 run 挂几小时就是几十万节点,
// pre-wrap 的 <pre> 每次追加+scrollHeight 读取都全量排版,标签页内存/CPU 无界膨胀
// 把整机拖卡。到上限就重拉一次 tail(loadTail 清空重装,跟随态视觉无感);
// 硬上限兜底覆盖"滚上去读历史挂机"的场景(会跳回底部,牺牲极少数情况保内存)。
const LOG_MAX_NODES=20000, LOG_HARD_MAX_NODES=40000;   // ~20000节点≈6000+行≈1MB文本
async function pollLog(){
  if(logFile===null){await loadTail();return;}
  const d=await logGet('?from='+logEnd+'&file='+encodeURIComponent(logFile));if(!d)return;
  if(d.rotated){await loadTail();return;}      // 新一轮 run 换了日志文件
  logSize=d.size||logSize;
  if(d.text){const stick=atBottom();
    logbox.appendChild(logFrag(d.text));logEnd=d.end;
    if(stick)logbox.scrollTop=logbox.scrollHeight;   // 到底则跟随最新
    if(logbox.childNodes.length>LOG_HARD_MAX_NODES||(stick&&logbox.childNodes.length>LOG_MAX_NODES)){
      await loadTail();return;}}   // DOM 裁剪:清回 256KB tail 规模
  logFollow=atBottom();logStat();
}
async function loadOlder(){
  if(logLoading||logAtStart||logFile===null||logTop<=0)return;
  logLoading=true;
  const d=await logGet('?before='+logTop+'&file='+encodeURIComponent(logFile));
  if(!d){logLoading=false;return;}
  if(d.rotated){logLoading=false;await loadTail();return;}
  if(d.text){const pH=logbox.scrollHeight,pT=logbox.scrollTop;
    logbox.insertBefore(logFrag(d.text),logbox.firstChild);
    logbox.scrollTop=pT+(logbox.scrollHeight-pH);}    // 保持视口不跳
  logTop=d.start||0;logAtStart=!!d.at_start;logLoading=false;logStat();
}
logbox.addEventListener('scroll',()=>{if(logbox.scrollTop<48)loadOlder();logFollow=atBottom();logStat();});
$('logbottom').onclick=()=>{logbox.scrollTop=logbox.scrollHeight;logFollow=true;logStat();};
loadTail();setInterval(pollLog,2000);
tick();setInterval(tick,3000);addEventListener('resize',tick);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/metrics":
            body = json.dumps(build_metrics()).encode()
            ctype = "application/json"
        elif parsed.path == "/api/log":
            body = json.dumps(serve_log(parse_qs(parsed.query))).encode()
            ctype = "application/json"
        else:
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # [2026-08-31] 轮询响应永不落盘缓存(页面每3s重取也无妨,量级几十KB)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    global LOG_POINTER, LOG_DIRS, POLAR_RUNS, PORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=":".join(LOG_DIRS), help="train_*.log dirs, colon-separated")
    ap.add_argument(
        "--log-pointer",
        default=LOG_POINTER,
        help="file containing the exact locally launched train log; empty disables it",
    )
    ap.add_argument("--polar", default=POLAR_RUNS, help="polar runs dir")
    ap.add_argument("--port", type=int, default=PORT)
    a = ap.parse_args()
    LOG_POINTER = a.log_pointer.strip()
    LOG_DIRS, POLAR_RUNS, PORT = a.log.split(":"), a.polar, a.port
    print(f"[rl_dashboard] log ptr : {LOG_POINTER or '(disabled)'}")
    print(f"[rl_dashboard] log dirs: {LOG_DIRS}")
    print(f"[rl_dashboard] latest  : {latest_log()}")
    print(f"[rl_dashboard] polar   : {latest_polar_run()}")
    print(f"[rl_dashboard] serving : http://0.0.0.0:{PORT}/  (Ctrl-C to stop)")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()


if __name__ == "__main__":
    main()
