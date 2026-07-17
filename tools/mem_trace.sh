#!/bin/bash
# mem_trace.sh —— 每 N 秒按进程类别聚合 PSS(正确处理 /dev/shm 共享内存),
# 输出总量 + 各类别 + top-N 单进程。边跑边抓,峰值一目了然。
# 用法: bash tools/mem_trace.sh [interval_sec] >> mem_trace.log
INTERVAL=${1:-3}
echo "# ts total_used shm | train vllm ray driver rollout other (PSS, GB) | top procs"
while true; do
  python3 - <<'PY'
import glob, time
buckets = {'train':0,'vllm':0,'ray':0,'driver':0,'rollout':0,'other':0}
procs = []
for p in glob.glob('/proc/[0-9]*'):
    try:
        cmd = open(f'{p}/cmdline','rb').read().replace(b'\0',b' ').decode('utf8','ignore')
        if not cmd.strip():
            continue
        pss = 0
        for line in open(f'{p}/smaps_rollup'):
            if line.startswith('Pss:'):
                pss = int(line.split()[1]); break   # kB
    except Exception:
        continue
    if 'MegatronTrainRayActor' in cmd: k='train'
    elif any(x in cmd for x in ('Worker_TP','VLLM::','EngineCore','VLLMEngine')): k='vllm'
    elif any(x in cmd for x in ('raylet','gcs_server','plasma','dashboard','log_monitor','runtime_env','monitor.py')): k='ray'
    elif 'train_async' in cmd: k='driver'
    elif 'RolloutManager' in cmd or 'rollout_function' in cmd: k='rollout'
    else: k='other'
    buckets[k]+=pss
    procs.append((pss, p.split('/')[-1], cmd[:38]))
import os
# 总量 & shm
mt=mf=0
for line in open('/proc/meminfo'):
    if line.startswith('MemTotal:'): mt=int(line.split()[1])
    if line.startswith('MemAvailable:'): mf=int(line.split()[1])
used=(mt-mf)/1048576
sumpss=sum(buckets.values())/1048576
ts=time.strftime('%H:%M:%S')
cat=' '.join(f'{k}={v/1048576:.0f}' for k,v in buckets.items())
top=' ; '.join(f'{pid}:{pss/1048576:.0f}G:{c.split()[0] if c.split() else c}' for pss,pid,c in sorted(procs,reverse=True)[:6])
print(f'[{ts}] used={used:.0f}G sumPss={sumpss:.0f}G | {cat} | {top}')
PY
  sleep "$INTERVAL"
done
