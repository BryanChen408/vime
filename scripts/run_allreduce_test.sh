#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 32 卡跨机 HCCL all-reduce 自测启动器(复现权重同步 HcclAllReduce timeout)
#
# 两台各跑一次(master 定在 140,故 140 = node_rank 0):
#   140:  NODE_RANK=0 bash scripts/run_allreduce_test.sh
#   141:  NODE_RANK=1 bash scripts/run_allreduce_test.sh
#
# 可调:NODE_RANK(必填,0/1)、MASTER_ADDR(默认 140)、SOCKET_IFNAME(默认按机器)、
#       NNODES(默认 2)、NPROC(默认 16)。
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

NODE_RANK=${NODE_RANK:?必须指定 NODE_RANK(140 上=0,141 上=1)}
MASTER_ADDR=${MASTER_ADDR:-80.5.25.140}
MASTER_PORT=${MASTER_PORT:-20022}
NNODES=${NNODES:-2}
NPROC=${NPROC:-16}
SOCKET_IFNAME=${SOCKET_IFNAME:-enp48s3u1u1}   # 承载 80.5.25.x 的 host 网卡;换机改这里
# 必须用带 torch_npu 的解释器!裸 python3/torchrun 可能命中 /usr/bin/python3(无 torch_npu)。
PYBIN=${PYBIN:-/usr/local/python3.11.15/bin/python3}

# ── 前置自检:解释器能否 import torch_npu(否则不是通信问题,是解释器选错)──
if ! "$PYBIN" -c "import torch_npu" 2>/dev/null; then
   echo "[FATAL] ${PYBIN} 无法 import torch_npu。" >&2
   echo "        用带 torch_npu 的解释器,例如:PYBIN=/usr/local/python3.11.15/bin/python3 $0" >&2
   echo "        (本机确认可用: /usr/local/python3.11.15/bin/python3)" >&2
   exit 2
fi

# 16 卡全暴露(A3:0-15)。Ascend 要求升序。
# 【强制覆盖】不用 :- —— 否则会继承 shell 里 start_pd_worker 残留的 "8..15"(只 8 卡可见),
# 而每 rank set_device(local_rank=0..15) 会在 local_rank>=8 时 Invalid device ID(107001)。
# 要自定义就显式传 VIS=... 。
export ASCEND_RT_VISIBLE_DEVICES=${VIS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

# ── 前置自检:可见卡数必须 >= 每节点进程数,否则高 local_rank 的 rank 起不来 → root 凑不齐 ──
VIS_COUNT=$(awk -F, '{print NF}' <<<"${ASCEND_RT_VISIBLE_DEVICES}")
if [ "${VIS_COUNT}" -lt "${NPROC}" ]; then
   echo "[FATAL] 可见卡数 ${VIS_COUNT} (ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}) < NPROC=${NPROC}。" >&2
   echo "        每 rank set_device(local_rank 0..$((NPROC-1))) 会越界报 107001。" >&2
   echo "        解决:别继承旧的可见卡设置(本脚本已强制 0-15),或把 NPROC 调到 ${VIS_COUNT}。" >&2
   exit 2
fi

# ── 跨机 HCCL 必需(与训练脚本对齐;缺则跨机 rendezvous 卡死)──
export HCCL_SOCKET_IFNAME="${SOCKET_IFNAME}"
export GLOO_SOCKET_IFNAME="${SOCKET_IFNAME}"
export TP_SOCKET_IFNAME="${SOCKET_IFNAME}"
export HCCL_SOCKET_FAMILY=${HCCL_SOCKET_FAMILY:-AF_INET}   # 强制 IPv4,避免 IPv6 socket family mismatch
export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1} # 禁 IP 白名单(否则跨机连接被拒→卡死)
# 注意:本容器【没装 ip 命令】,故用 python ioctl 从网卡取 IPv4(勿改回 `ip addr`)
export HCCL_IF_IP=${HCCL_IF_IP:-$(python3 -c "
import socket,fcntl,struct,sys
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
try: print(socket.inet_ntoa(fcntl.ioctl(s.fileno(),0x8915,struct.pack('256s','${SOCKET_IFNAME}'.encode()[:15]))[20:24]))
except OSError: sys.exit(0)" 2>/dev/null)}
export HCCL_IF_IP=${HCCL_IF_IP:-$(hostname -I | awk '{print $1}')}

# ── 关键:让它【失败快】(不通时秒级/百秒级报,而不是干等 watchdog 10 分钟)──
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-120}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-120}
export PG_INIT_TIMEOUT=${PG_INIT_TIMEOUT:-180}

# 端口范围(与训练脚本一致,便于同款防火墙放行验证)
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}

echo "[cfg] node_rank=${NODE_RANK}/${NNODES}  master=${MASTER_ADDR}:${MASTER_PORT}  if=${SOCKET_IFNAME}  HCCL_IF_IP=${HCCL_IF_IP}"
echo "[cfg] world=$((NNODES*NPROC))  connect_timeout=${HCCL_CONNECT_TIMEOUT}s  exec_timeout=${HCCL_EXEC_TIMEOUT}s"

# 用 $PYBIN -m torch.distributed.run(绑定解释器),不用裸 torchrun
exec "${PYBIN}" -m torch.distributed.run \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC}" \
  test_allreduce.py
