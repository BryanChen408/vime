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
