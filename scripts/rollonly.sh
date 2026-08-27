unset https_proxy
unset http_proxy
nic_name="ens1f3" # ifconfig 查看，选和本机 ip 相同的网卡
local_ip=80.48.5.56

export HCCL_IF_IP=$local_ip         # 指定HCCL通信库使用的网卡 IP 地址
export GLOO_SOCKET_IFNAME=$nic_name # 指定使用 Gloo通信库时指定网络接口名称 
export TP_SOCKET_IFNAME=$nic_name   # 指定 TensorParallel使用的网络接口名称
export HCCL_SOCKET_IFNAME=$nic_name # 指定 HCCL 通信库使用的网络接口名称
export OMP_PROC_BIND=false          # 允许操作系统调度线程在多个核心之间迁移
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1          # 在支持 OpenMP 的程序中，最多使用 100 个 CPU 线程进行并行计算
export HCCL_BUFFSIZE=2048           # 每个通信操作的缓冲区大小为 1024 Bytes
# echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# sysctl -w vm.swappiness=0
# sysctl -w kernel.numa_balancing=0
# sysctl kernel.sched_migration_cost_ns=50000
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH:-}"
# NNAL/ATB: vllm_ascend worker 会调用 _register_atb_extensions()，需要 libatb.so 在 LD_LIBRARY_PATH 上
source /usr/local/Ascend/nnal/atb/set_env.sh

export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
# export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export DYNAMIC_EPLB="false"
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=500

# --quantization ascend \

# --enforce-eager \
# --async-scheduling \
# --prefill-context-parallel-size 2 \
# --decode_context_parallel_size 8 \
# --data-parallel-size 2 \
# --data-parallel-size-local 2 \

#   --host $local_ip \
#   --data-parallel-address $local_ip \
#   --data-parallel-rpc-port 13389 \
# "ascend_scheduler_config":{"enabled":false}, "torchair_graph_config":{"enabled":false,"enable_multistream_moe":false,"use_cached_graph":false},
# /mnt/nfs/weight/Qwen3-Coder-480B-A35B-Instruct-W8A8  /mnt/nfs/weights/DeepSeek-R1-0528_w8a8_mix_mtp /mnt/weight/DeepSeek-V3.1-Terminus-w8a8-QuaRot-lfs
# --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16], "cudagraph_mode": "FULL_DECODE_ONLY"}' \
# --max-model-len 1050000  \ /mnt/share/d00933242/DeepSeek-V2-Lite-w8a8
# --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
# --enable-chunked-prefill \
# --profiler_config '{"profiler":"torch", "torch_profiler_dir":"/home/g00955623/profiling/cpp_16k", "torch_profiler_with_stack":false, "torch_profiler_with_memory":false, "torch_profiler_record_shapes":true}' \

vllm serve /home/docker/Qwen3.6-35B-A3B \
  --host 0.0.0.0 \
  --port 8003 \
  --served-model-name model \
  --data-parallel-size 1 \
  --tensor-parallel-size 2 \
  --pipeline_parallel_size 1 \
  --prefill-context_parallel-size 1 \
  --decode-context_parallel-size 1 \
  --enable-expert-parallel \
  --max-num-seqs 16 \
  --max-model-len 131072  \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --trust-remote-code \
  --additional-config '{
    "enable_cpu_binding":false
    }' \
  --compilation-config '{"cudagraph_capture_sizes":[4,8,12,16,24,32], "cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --speculative_config ' {"method": "mtp", "num_speculative_tokens": 3, "enforce_eager": true} ' \
  2>&1 | tee test2.log