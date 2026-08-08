#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# 在 140 上一键落地:①写出 HCCL all-reduce 测试文件  ②改好 rollout 卡位(4-15)
# 用法(在 140 的 /workspace/vime 下):
#   cd /workspace/vime && bash setup_on_140.sh
# 跑完即可做不对称测试:
#   140:  ROLE=worker bash scripts/run_allreduce_asym.sh
#   141:  ROLE=head   bash scripts/run_allreduce_asym.sh   ← 141 先起
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "[*] 工作目录: $(pwd)"
mkdir -p scripts

# ─── 文件1: test_allreduce.py(asym / 对称测试共用的测试主体)───
cat > test_allreduce.py <<'PY_EOF'
import torch
import torch_npu  # noqa: F401  注册 npu/hccl 后端,必须 import
import os, time, socket, datetime

rank       = int(os.getenv('RANK', 0))
local_rank = int(os.getenv('LOCAL_RANK', 0))
world      = int(os.getenv('WORLD_SIZE', 1))
host       = socket.gethostname()
torch.npu.set_device(local_rank)
dev = f'npu:{local_rank}'

# 建 HCCL 域。timeout 设短(默认吃 HCCL_CONNECT_TIMEOUT),数据面不通时【快速失败】,
# 不要像权重同步那次干等到 watchdog 10 分钟。
t0 = time.time()
torch.distributed.init_process_group(
    backend='hccl', init_method='env://',
    timeout=datetime.timedelta(seconds=int(os.getenv('PG_INIT_TIMEOUT', 180))))
if rank == 0:
    print(f'[init] world={world} process group ready in {time.time()-t0:.1f}s', flush=True)
torch.distributed.barrier()

# ① 正确性:每 rank 贡献自身 rank,SUM 应 = 0+1+..+(world-1)
a = torch.full((1024,), float(rank), device=dev)
torch.distributed.all_reduce(a)
torch.npu.synchronize()
exp = world * (world - 1) / 2.0
got = a[0].item()
print(f'[rank {rank:>2} {host} {dev}] got={got:.0f} expect={exp:.0f} '
      f'{"OK" if abs(got-exp) < 1e-3 else "MISMATCH"}', flush=True)

# ② 带宽:256MB all-reduce ×5(近似权重广播),看真实跨机 RoCE 吞吐
torch.distributed.barrier()
n = int(os.getenv('BW_MB', 256)) * 1024 * 1024
buf = torch.ones(n // 4, device=dev)
for _ in range(2):
    torch.distributed.all_reduce(buf)
torch.npu.synchronize(); torch.distributed.barrier()
t0 = time.time()
for _ in range(5):
    torch.distributed.all_reduce(buf)
torch.npu.synchronize()
dt = (time.time() - t0) / 5
if rank == 0:
    algbw = n / dt / 1e9
    print(f'[bw] {int(os.getenv("BW_MB",256))}MB x5: {dt*1000:.1f} ms/iter  '
          f'algbw={algbw:.1f} GB/s  busbw={algbw*2*(world-1)/world:.1f} GB/s', flush=True)

torch.distributed.barrier()
if rank == 0:
    print(f'[done] {world} 卡 all-reduce 全通 → 跨机 HCCL 数据面 OK', flush=True)
torch.distributed.destroy_process_group()
PY_EOF
echo "[ok] 写出 test_allreduce.py"

# ─── 文件2: scripts/run_allreduce_asym.sh(不对称 1+12 复刻测试)───
cat > scripts/run_allreduce_asym.sh <<'ASYM_EOF'
#!/usr/bin/env bash
# 不对称 HCCL all-reduce 复刻:141=1 rank + 140=N rank → world=1+N。
# 手动 spawn + env://(不用 torchrun rendezvous —— c10d 会解析本机 hostname
#   node-25-14x,容器里不可解析 → getaddrinfo gaierror)。
#   141:  ROLE=head   bash scripts/run_allreduce_asym.sh
#   140:  ROLE=worker bash scripts/run_allreduce_asym.sh   ← 141 先起
set -uo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

ROLE=${ROLE:?必须指定 ROLE=head(141) 或 ROLE=worker(140)}
PYBIN=${PYBIN:-/usr/local/python3.11.15/bin/python3}
SOCKET_IFNAME=${SOCKET_IFNAME:-enp48s3u1u1}
MASTER_ADDR=${MASTER_ADDR:-80.5.25.141}
MASTER_PORT=${MASTER_PORT:-29500}
HEAD_NPROC=${HEAD_NPROC:-1}
WORKER_NPROC=${WORKER_NPROC:-12}
WORLD=$((HEAD_NPROC + WORKER_NPROC))

"$PYBIN" -c "import torch_npu" 2>/dev/null || { echo "[FATAL] ${PYBIN} 无 torch_npu" >&2; exit 2; }

if [ "$ROLE" = "head" ]; then
   NPROC=$HEAD_NPROC; BASE_RANK=0
   export ASCEND_RT_VISIBLE_DEVICES=${HEAD_VIS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
else
   NPROC=$WORKER_NPROC; BASE_RANK=$HEAD_NPROC
   export ASCEND_RT_VISIBLE_DEVICES=${WORKER_VIS:-4,5,6,7,8,9,10,11,12,13,14,15}
fi
VIS_COUNT=$(awk -F, '{print NF}' <<<"${ASCEND_RT_VISIBLE_DEVICES}")
[ "$VIS_COUNT" -ge "$NPROC" ] || { echo "[FATAL] 可见卡 ${VIS_COUNT} < NPROC ${NPROC}" >&2; exit 2; }

export HCCL_SOCKET_IFNAME="${SOCKET_IFNAME}"
export GLOO_SOCKET_IFNAME="${SOCKET_IFNAME}"
export TP_SOCKET_IFNAME="${SOCKET_IFNAME}"
export HCCL_SOCKET_FAMILY=${HCCL_SOCKET_FAMILY:-AF_INET}
export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-120}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-120}
export PG_INIT_TIMEOUT=${PG_INIT_TIMEOUT:-180}
# 【必须】单机多进程:每进程 NPU 绑不同端口,否则全撞默认 16666 → EI0020 / error code 7。
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export MASTER_ADDR MASTER_PORT
export WORLD_SIZE=$WORLD
export HCCL_IF_IP=${HCCL_IF_IP:-$("$PYBIN" -c "
import socket,fcntl,struct,sys
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
try: print(socket.inet_ntoa(fcntl.ioctl(s.fileno(),0x8915,struct.pack('256s','${SOCKET_IFNAME}'.encode()[:15]))[20:24]))
except OSError: sys.exit(0)" 2>/dev/null)}

echo "[cfg] role=${ROLE} nproc=${NPROC} base_rank=${BASE_RANK} world=${WORLD} master=${MASTER_ADDR}:${MASTER_PORT} if=${SOCKET_IFNAME} IF_IP=${HCCL_IF_IP} vis=${ASCEND_RT_VISIBLE_DEVICES}"

pids=()
for i in $(seq 0 $((NPROC-1))); do
   RANK=$((BASE_RANK + i)) LOCAL_RANK=$i "$PYBIN" test_allreduce.py &
   pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ "$rc" = 0 ] && echo "[asym] 本节点所有 rank 正常退出" || echo "[asym] 有 rank 失败(见上)"
exit $rc
ASYM_EOF
echo "[ok] 写出 scripts/run_allreduce_asym.sh"

# ─── 改动1: start_pd_worker.sh 的卡位(8-15 → 4-15,NPUS 8 → 12)───
W=scripts/start_pd_worker.sh
if [ -f "$W" ]; then
   cp -f "$W" "${W}.bak.$(date +%Y%m%d_%H%M%S)"
   sed -i \
     -e 's/^export ASCEND_RT_VISIBLE_DEVICES=.*/export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7,8,9,10,11,12,13,14,15/' \
     -e 's/^export NPUS_PER_NODE=.*/export NPUS_PER_NODE=12/' \
     "$W"
   echo "[ok] 已改 ${W}(已备份): CVD=$(grep -oE 'ASCEND_RT_VISIBLE_DEVICES=[0-9,]+' "$W")  $(grep -oE 'NPUS_PER_NODE=[0-9]+' "$W")"
else
   echo "[warn] 未找到 ${W},跳过(140 若不跑 worker 脚本可忽略)"
fi

# ─── 改动2: layout 卡位(rollout 4-15 / polar 0-3 / dp_size 3)───
L=scripts/resource_layout.dual56train57infer_pd.yaml
if [ -f "$L" ]; then
   cp -f "$L" "${L}.bak.$(date +%Y%m%d_%H%M%S)"
   "${PYBIN:-python3}" - "$L" <<'PYFIX'
import re, sys
p = sys.argv[1]
s = open(p).read()
# rollout 那台 140 的 devices → 4-15
s = re.sub(r'(rollout:\s*\n\s*-\s*\{node:\s*80\.5\.25\.140,\s*devices:\s*")[^"]*(")', r'\g<1>4-15\g<2>', s)
# polar_reserved 140 → 0-3
s = re.sub(r'(polar_reserved:\s*\n\s*-\s*\{node:\s*80\.5\.25\.140,\s*devices:\s*")[^"]*(")', r'\g<1>0-3\g<2>', s)
# vllm_dp_size → 3
s = re.sub(r'(vllm_dp_size:\s*)\d+', r'\g<1>3', s)
open(p, 'w').write(s)
print("[ok] 已改 layout:")
for ln in s.splitlines():
    if any(k in ln for k in ('actor','rollout','polar','devices','vllm_dp_size','node:')):
        print("     " + ln.strip())
PYFIX
else
   echo "[warn] 未找到 ${L},跳过"
fi

# ─── 自检:卡位一致性 ───
echo
echo "===== 卡位一致性自检 ====="
grep -oE 'ASCEND_RT_VISIBLE_DEVICES=[0-9,]+' scripts/start_pd_worker.sh 2>/dev/null || true
grep -E 'devices:|vllm_dp_size:' scripts/resource_layout.dual56train57infer_pd.yaml 2>/dev/null | sed 's/^/  /' || true
echo
echo "[done] 140 落地完成。下一步做不对称测试:"
echo "   141 先起:  ROLE=head   bash scripts/run_allreduce_asym.sh"
echo "   140 再起:  ROLE=worker bash scripts/run_allreduce_asym.sh"

