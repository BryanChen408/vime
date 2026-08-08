#!/usr/bin/env python3
# 权重同步卡死的【忠实复刻】:不走 torch.distributed,直接用真实路径
#   StatelessProcessGroup(TCPStore) + PyHcclCommunicator(hcclCommInitRank + warmup all_reduce)
# 与 vllm-ascend hccl_engine.py:_stateless_init_process_group 逐字同路径。
#
# 用法(env 驱动,两台分别跑):
#   141(rank0, device 0):
#     RANK=0 LOCAL_RANK=0 WORLD_SIZE=9 MASTER_ADDR=80.5.25.141 MASTER_PORT=39999 \
#       ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
#       /usr/local/python3.11.15/bin/python3 test_wsync_repro.py
#   140(rank1-8, 8 进程, logical device 0-7 → 物理 8-15):
#     for i in 0..7: RANK=$((i+1)) LOCAL_RANK=$i WORLD_SIZE=9 MASTER_ADDR=80.5.25.141 MASTER_PORT=39999 \
#       ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 python3 test_wsync_repro.py &
#   (或用 scripts/run_wsync_repro.sh 包一层)
#
# 判读:全部打印 "[done]" = 真实路径也通 → 问题在 vime/vllm 进程环境;
#       卡住/600s 后 HCCL timeout = 100% 复刻训练故障 → 可用 env 逐个二分。
import os
import time

import torch
import torch_npu  # noqa: F401

from vllm.distributed.utils import StatelessProcessGroup
from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
host = os.environ.get("MASTER_ADDR", "80.5.25.141")
port = int(os.environ.get("MASTER_PORT", "39999"))

torch.npu.set_device(local_rank)
device = torch.device(f"npu:{local_rank}")

t0 = time.time()
print(f"[rank {rank}] StatelessProcessGroup.create host={host}:{port} world={world} ...", flush=True)
pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world)
print(f"[rank {rank}] TCPStore pg ready in {time.time()-t0:.1f}s → 建 HCCL comm(hcclCommInitRank + warmup all_reduce)...", flush=True)

t1 = time.time()
comm = PyHcclCommunicator(pg, device=device)   # ← 真实卡死点:内部 hcclCommInitRank + warmup all_reduce
print(f"[rank {rank}] HCCL comm ready in {time.time()-t1:.1f}s", flush=True)

# 再主动做一次 all_reduce 确认数据面
x = torch.full((1024,), float(rank), device=device, dtype=torch.float32)
comm.all_reduce(x)
torch.npu.synchronize()
exp = world * (world - 1) / 2.0
got = x[0].item()
print(f"[rank {rank}] all_reduce got={got:.0f} expect={exp:.0f} {'OK' if abs(got-exp)<1e-3 else 'MISMATCH'}", flush=True)
print(f"[rank {rank}] [done]", flush=True)
