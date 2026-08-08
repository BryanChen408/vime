#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 不对称 HCCL all-reduce 复刻:精确模拟权重同步组的形状
#   141(训练节点)= 1 个 rank(1 张卡)   +   140(rollout 节点)= N 个 rank(N 张卡)
#   → world_size = 1 + N。验证"1 张孤卡 vs 一堆跨双网段卡"这个拓扑能否 all_reduce。
#
# 【手动 spawn + env://】不用 torchrun rendezvous —— c10d 会去解析本机 hostname
#   (node-25-14x)判断谁是 host,容器里该名不在 DNS/hosts → getaddrinfo gaierror。
#   改成每进程显式给 RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT,直接用 IP,
#   与真实权重同步(手动指定 rank/world/master)也更一致。
#
# 用法(rank0 在 141,故 141 先起):
#   141:  ROLE=head   bash scripts/run_allreduce_asym.sh
#   140:  ROLE=worker bash scripts/run_allreduce_asym.sh
#
# 可调:HEAD_NPROC(默认1)、WORKER_NPROC(默认12)、MASTER_ADDR(默认80.5.25.141)、
#       MASTER_PORT(默认29500)、WORKER_VIS(140物理卡,默认4-15)、HEAD_VIS。
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

ROLE=${ROLE:?必须指定 ROLE=head(141) 或 ROLE=worker(140)}
PYBIN=${PYBIN:-/usr/local/python3.11.15/bin/python3}
SOCKET_IFNAME=${SOCKET_IFNAME:-enp48s3u1u1}
MASTER_ADDR=${MASTER_ADDR:-80.5.25.141}       # rank0 在 141
MASTER_PORT=${MASTER_PORT:-29500}
HEAD_NPROC=${HEAD_NPROC:-1}
WORKER_NPROC=${WORKER_NPROC:-12}
WORLD=$((HEAD_NPROC + WORKER_NPROC))

"$PYBIN" -c "import torch_npu" 2>/dev/null || { echo "[FATAL] ${PYBIN} 无 torch_npu" >&2; exit 2; }

if [ "$ROLE" = "head" ]; then
   NPROC=$HEAD_NPROC; BASE_RANK=0
   export ASCEND_RT_VISIBLE_DEVICES=${HEAD_VIS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
else
   NPROC=$WORKER_NPROC; BASE_RANK=$HEAD_NPROC   # 140 的 rank 从 HEAD_NPROC 起(=1)
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
# 【必须】单机多进程:每进程 NPU 要绑不同端口,否则全撞默认 16666 → EI0020 / error code 7。
# 与训练脚本一致(start_pd.sh)。范围要 >= 本机进程数。
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
