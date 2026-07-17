#!/usr/bin/env python3
"""vLLM 内置 metrics 采集 + PD 配比测算(在真实 polar rollout 跑时用)。

用法:
    python3 vllm_metrics_probe.py <metrics_url> [dt_seconds] [--prefill-batch N]
例:
    python3 vllm_metrics_probe.py http://localhost:15000/metrics 15
    # 多引擎:对每个引擎端口各跑一次,QPS/token速率相加、TTFT/TPOT/长度取加权平均

产出:base 指标 + decode 单例 QPS + prefill 单例 QPS(两版:原式 & token 式)+ N_p:N_d 配比。
两版 prefill QPS 差多少 = 判断"prefill batch/TTFT 工作点干不干净"的信号(见 handoff 评估方法节)。
"""
import sys, time, urllib.request

def scrape(url):
    m = {}
    for ln in urllib.request.urlopen(url, timeout=5).read().decode().splitlines():
        if ln.startswith("vllm:") and " " in ln and not ln.startswith("#"):
            k, v = ln.rsplit(" ", 1)
            try:
                m[k] = m.get(k, 0.0) + float(v)   # 同名多标签(多 DP rank)累加
            except ValueError:
                pass
    return m

def sc(m, base):                       # histogram: (_count, _sum)
    return m.get(base + "_count", 0.0), m.get(base + "_sum", 0.0)

def total(m, base):                    # counter: 匹配 base 前缀的 _total 求和
    return sum(v for k, v in m.items() if k.startswith(base))

def main():
    url = sys.argv[1]
    dt = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else 15.0
    prefill_batch = None
    if "--prefill-batch" in sys.argv:
        prefill_batch = float(sys.argv[sys.argv.index("--prefill-batch") + 1])

    a = scrape(url); time.sleep(dt); b = scrape(url)

    # base 指标(histogram 均值)
    c_ttft, s_ttft = sc(b, "vllm:time_to_first_token_seconds")
    c_tpot, s_tpot = sc(b, "vllm:request_time_per_output_token_seconds")
    c_in,  s_in    = sc(b, "vllm:request_prompt_tokens")
    c_out, s_out   = sc(b, "vllm:request_generation_tokens")
    TTFT   = s_ttft / c_ttft if c_ttft else float("nan")     # s
    TPOT   = s_tpot / c_tpot if c_tpot else float("nan")     # s/token
    IN_LEN = s_in / c_in if c_in else float("nan")
    OUT_LEN= s_out / c_out if c_out else float("nan")
    RUN    = total(b, "vllm:num_requests_running")           # 瞬时 batch(gauge)
    WAIT   = total(b, "vllm:num_requests_waiting")

    # 速率(counter 做差)
    req_qps      = (total(b, "vllm:request_success") - total(a, "vllm:request_success")) / dt
    prefill_toks = (total(b, "vllm:prompt_tokens")   - total(a, "vllm:prompt_tokens"))   / dt
    decode_toks  = (total(b, "vllm:generation_tokens")- total(a, "vllm:generation_tokens"))/ dt

    # ── PD 单例 QPS ──
    #  Decode 单例 QPS = DecodeBatchSize / (TPOT × 输出长度)   [Little's law]
    qps_d = RUN / (TPOT * OUT_LEN) if (TPOT and OUT_LEN) else float("nan")
    #  Prefill 单例 QPS 原式 = BatchSize / TTFT  (BatchSize 用 --prefill-batch,缺省用 RUN 作粗代理)
    pbatch = prefill_batch if prefill_batch is not None else RUN
    qps_p_orig = pbatch / TTFT if TTFT else float("nan")
    #  Prefill 单例 QPS token 式 = prefill_token吞吐 / 平均输入长度   (稳,不用猜 prefill batch)
    qps_p_tok = prefill_toks / IN_LEN if IN_LEN else float("nan")

    def ratio(qp): return (qps_d / qp) if qp else float("nan")   # N_p:N_d = qps_d:qps_p → 归一到 N_d=1

    print(f"[窗口 {dt:.0f}s]  Running(batch)={RUN:.1f}  Waiting={WAIT:.1f}")
    print(f"  TTFT      = {TTFT*1000:8.1f} ms")
    print(f"  TPOT      = {TPOT*1000:8.1f} ms")
    print(f"  输入长度  = {IN_LEN:8.0f} tok")
    print(f"  输出长度  = {OUT_LEN:8.0f} tok")
    print(f"  请求 QPS  = {req_qps:8.3f} req/s   prefill={prefill_toks:.0f} tok/s  decode={decode_toks:.0f} tok/s")
    print(f"  ── 单例 QPS ──")
    print(f"  Decode 单例 QPS                  = {qps_d:.4f} req/s")
    print(f"  Prefill 单例 QPS [原式 B/TTFT]   = {qps_p_orig:.4f} req/s  (B={pbatch:.1f}{' 供' if prefill_batch else ' =Running粗代理'})")
    print(f"  Prefill 单例 QPS [token 式]      = {qps_p_tok:.4f} req/s")
    print(f"  ── N_p : N_d 配比(归一 N_d=1) ──")
    print(f"  用原式    prefill QPS:  N_p:N_d = {ratio(qps_p_orig):.2f} : 1")
    print(f"  用 token 式:            N_p:N_d = {ratio(qps_p_tok):.2f} : 1")
    if qps_p_orig and qps_p_tok:
        print(f"  两版 prefill QPS 差异 = {abs(qps_p_orig-qps_p_tok)/qps_p_tok*100:.0f}%  (差大→信 token 式,见 handoff)")

if __name__ == "__main__":
    main()
