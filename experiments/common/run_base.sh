#!/bin/bash
# 多机 NPU 训练的公共底座：环境变量、机器拓扑、模型/rollout/性能/优化器/vLLM 参数、Ray 启动。
# 算法脚本只设置算法相关的东西，然后在最后一行 source 本文件。
#
# 契约：底座不设默认值。算法脚本必须提供下面这批变量，缺一个就 unbound variable 退出。
#   ALGORITHM_TAG                 算法名，进日志/权重/wandb group 路径
#   ALGO_ARGS                     算法参数数组（advantage estimator、clip、KL、critic 等）
#   ROLLOUT_BATCH_SIZE            每轮取几个 prompt
#   N_SAMPLES_PER_PROMPT          每个 prompt 采几条（group 系算法必须 > 1）
#   ROLLOUT_MAX_ACTIVE_SESSIONS   polar 并发 session 上限，要随 N_SAMPLES_PER_PROMPT 放大
# 其余参数（模型、拓扑、并行度、长度、环境变量）在本文件里直接赋值，单一来源，不留覆盖口子。
# 某个参数确实要按算法区分时，把它从下面的常量区挪到这里，改两行。
#
# 用法
#   ALGORITHM_TAG=gspo
#   ROLLOUT_BATCH_SIZE=8
#   N_SAMPLES_PER_PROMPT=8
#   ROLLOUT_MAX_ACTIVE_SESSIONS=64
#   ALGO_ARGS=(--advantage-estimator gspo --eps-clip 1e-4 --eps-clip-high 2e-4)
#   source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/../common/run_base.sh"

# 配置写错当场退出，
set -eux
: "${ALGORITHM_TAG}" "${ROLLOUT_BATCH_SIZE}" "${N_SAMPLES_PER_PROMPT}" "${ROLLOUT_MAX_ACTIVE_SESSIONS}" "${ALGO_ARGS[@]}"

# 清理残留进程：VLLM:: worker 占着显存，lb_proxy 占着 router 端口。
pkill -9 -f 'VLL[M]::' 2>/dev/null || true
pkill -9 -f 'vime.ray.lb_proxy' 2>/dev/null || true
sleep 2

BASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${BASE_DIR}/../.." &>/dev/null && pwd)"
cd "${VIME_ROOT}"

# 这两个第三方 set_env.sh 不属于本文件，nnal/atb 里直接引用了未定义的 $ZSH_VERSION，
# 在 -u 下会 unbound variable 退出。只对它们临时关掉 nounset，其余代码保持 -u。
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

# ─────────────────────────────────────────────────────────────────────
# 全算法共用的常量（要改就在这里改，改了所有算法一起变）
# ─────────────────────────────────────────────────────────────────────

# 训练 & 推理规模。ROLLOUT_BATCH_SIZE / N_SAMPLES_PER_PROMPT 由算法脚本提供。
NUM_ROLLOUT=50
NUM_STEPS_PER_ROLLOUT=1
# gbs 是推出来的，不给算法脚本手写的机会——手写就要和 arguments.py:1977 的断言对齐，
# 对不上会在跑起来之后才 assert。
GLOBAL_BATCH_SIZE=$(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / NUM_STEPS_PER_ROLLOUT ))
VLLM_MAX_NUM_SEQS=96
VLLM_MAX_NUM_BATCHED_TOKENS=8192
ROLLOUT_MAX_ASYNC_LEVEL=5

ROLLOUT_MAX_RESPONSE_LEN=24576
ROLLOUT_MAX_CONTEXT_LEN=262144
VLLM_MAX_MODEL_LEN=262144
SEQ_LENGTH=262144
MAX_TOKENS_PER_GPU=32768
# SEQ_LENGTH 是 megatron 训练侧支持的最大序列长度（prompt + response，单个 sample/trace）
# MAX_TOKENS_PER_GPU 每个 micro-batch 中每张 CP 卡处理的最大 token 数，训练时按它动态拆 micro-batch

# 并行度。约束：TP * CP * PP 要能整除 ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE，
# EP 要能整除同一个数。加机器后主要调 CP 和 EP。
TP=2
PP=1
CP=8
EP=8

(( MAX_TOKENS_PER_GPU * CP >= SEQ_LENGTH )) || { echo "MAX_TOKENS_PER_GPU * CP must be >= SEQ_LENGTH" >&2; exit 1; }

# 数据
DATASET_TAG=ascendc
OPERATOR_TASK_JSONL=/home/docker/op_tasks/op_assets_kernelbench_level1/operator_tasks.ascendc.jsonl
OPERATOR_TASKS_DIR=/home/docker/op_tasks/op_assets_kernelbench_level1/op_tasks

# 权重
HF_CKPT=/home/docker/Qwen3.6-35B-A3B
REF_LOAD=/home/docker/Qwen3.6-35B-A3B_fused_torch_dist

MODEL_TAG=qwen36_35b
LR=1e-6
SAVE_INTERVAL=5

# ─────────────────────────────────────────────────────────────────────
# 机器拓扑
# ─────────────────────────────────────────────────────────────────────

# 本机配置（两台机器分别维护 experiments/common/node_config.sh）
source "${BASE_DIR}/node_config.sh"

NNODES=2
NODE1_IP=80.48.5.57
NODE2_IP=80.48.5.66
MASTER_ADDR="${NODE1_IP}"

# NODE1 跑 actor/critic（0-15，CPU offload 换手），NODE2 跑 rollout（4-15，0-3 留给 polar）。
# NODE1 是 Ray head 和训练节点，NODE2 上的 polar 服务要先起在 :8080。
ACTOR_NUM_NODES=1
ACTOR_NUM_GPUS_PER_NODE=16
ROLLOUT_NUM_GPUS=12
ROLLOUT_NUM_GPUS_PER_ENGINE=4
POLAR_URL="http://${NODE2_IP}:8080"

RAY_PORT=6460
RAY_DASHBOARD_PORT=8290
RUN_ID=version__$(date +%Y%m%d-%H%M)
RAY_TEMP_DIR=/tmp/ray_${MODEL_TAG}_${ALGORITHM_TAG}_${DATASET_TAG}
POLAR_OUTPUT_DIR=output/${MODEL_TAG}_${ALGORITHM_TAG}_${DATASET_TAG}/${RUN_ID}
SAVE_DIR=/workspace/${MODEL_TAG}_${ALGORITHM_TAG}_${DATASET_TAG}/${RUN_ID}
LOG_FILE=/home/docker/logs/${MODEL_TAG}_${ALGORITHM_TAG}_${DATASET_TAG}_${RUN_ID}.log
mkdir -p logs "${POLAR_OUTPUT_DIR}" /home/docker/logs

# 布局表
RESOURCE_LAYOUT="${POLAR_OUTPUT_DIR}/resource_layout_${RUN_ID}.yaml"
cat > "${RESOURCE_LAYOUT}" <<EOF
roles:
  actor:
    - {node: ${NODE1_IP}, devices: "0-15"}
  rollout:
    - {node: ${NODE2_IP}, devices: "4-15"}
rollout:
  num_gpus_per_engine: ${ROLLOUT_NUM_GPUS_PER_ENGINE}
EOF

# ─────────────────────────────────────────────────────────────────────
# 下面一般不用动
# ─────────────────────────────────────────────────────────────────────

export WANDB_BASE_URL="http://80.48.5.56:8088"
export WANDB_API_KEY="local-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"
WANDB_ARGS=(
   --use-wandb
   --wandb-mode online
   --wandb-project zhh
   --wandb-group "${MODEL_TAG}_${ALGORITHM_TAG}_${DATASET_TAG}_${RUN_ID}"
   --disable-wandb-random-suffix
   --wandb-dir /home/docker/wandb
)

export PYTHONUNBUFFERED=1
export PYTHONPATH="/workspace/vllm:/workspace/vllm-ascend:/workspace/Megatron-LM:${VIME_ROOT}:${PYTHONPATH:-}"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2    # 高性能内存回收，防止 CPU OOM
# Ascend 自定义 MoE 训练算子库（moe_grouped_matmul / grouped_matmul_swiglu）
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH:-}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TASK_QUEUE_ENABLE=0          # =1 会让 GDN/ring-attn 训练出 NaN
export TORCHDYNAMO_DISABLE=1        # 昇腾 inductor get_gpu_type() 断言 → eager
export CPU_AFFINITY_CONF=1
export VLLM_ASCEND_ENABLE_NZ=0      # MoE + RL 下 NZ 格式冲突，必须 0
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=1
export RAY_memory_usage_threshold=0.98
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HCCL_CONNECT_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=2400
export HCCL_BUFFSIZE=512
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_SOCKET_FAMILY=AF_INET
export HCCL_WHITELIST_DISABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export POLAR_KEEP_SESSION_DIR=1
export POLAR_ANTHROPIC_DEFAULT_MAX_TOKENS=12288

source "${VIME_ROOT}/scripts/models/qwen3.5-35B-A3B.sh"

# qwen3.5-35B-A3B.sh 里有 --moe-permute-fusion，NPU 走 MindSpeed fallback，去掉避免警告。
_MODEL_ARGS_KEPT=()
for _arg in "${MODEL_ARGS[@]}"; do
   [ "$_arg" = "--moe-permute-fusion" ] || _MODEL_ARGS_KEPT+=("$_arg")
done
MODEL_ARGS=("${_MODEL_ARGS_KEPT[@]}")
unset _MODEL_ARGS_KEPT _arg

CURRENT_IP=$(ip -o -4 addr show "${SOCKET_IFNAME}" | awk '{print $4}' | cut -d/ -f1 | head -1)
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},${CURRENT_IP}${no_proxy:+,${no_proxy}},80.48.5.56,80.48.5.57,80.48.5.58,80.48.5.59,80.48.5.64,80.48.5.66"
export NO_PROXY="${no_proxy}"
export HCCL_SOCKET_IFNAME="${SOCKET_IFNAME}"
export GLOO_SOCKET_IFNAME="${SOCKET_IFNAME}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_DIR}/"
   --save-interval "${SAVE_INTERVAL}"
   --no-save-optim
   --megatron-to-hf-mode raw
)

TOPO_ARGS=(
   --actor-num-nodes ${ACTOR_NUM_NODES}
   --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE}
   --rollout-num-gpus ${ROLLOUT_NUM_GPUS}
   --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
   --resource-layout "${RESOURCE_LAYOUT}"
)

ROLLOUT_ARGS=(
   --rollout-function-path vime_bridge.rollout.generate_rollout_polar_async
   --prompt-data "${OPERATOR_TASK_JSONL}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --reward-key score
   --custom-reward-post-process-path vime_bridge.reward_post_process.post_process_rewards
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN}
   --rollout-max-context-len ${ROLLOUT_MAX_CONTEXT_LEN}
   --rollout-temperature 0.7
   --rollout-seed 42
   # 每步落盘，供事后分析。train 侧每个 rank 一份。
   --save-debug-rollout-data "${POLAR_OUTPUT_DIR}/debug/rollout_{rollout_id}.pt"
   --save-debug-train-data "${POLAR_OUTPUT_DIR}/debug/train_{rollout_id}_rank{rank}.pt"
)

POLAR_ARGS=(
   --polar-url "${POLAR_URL}"
   --polar-run-id "${RUN_ID}"
   --polar-reward-key score
   --polar-task-id-template "{args.polar_run_id}-polar-op-{rollout_id}-{sample.group_index}"
   --operator-tasks-dir "${OPERATOR_TASKS_DIR}"
   --rollout-max-async-level ${ROLLOUT_MAX_ASYNC_LEVEL}
   --rollout-request-timeout 8000
   --rollout-scheduler-mode session_pool
   --rollout-max-active-sessions ${ROLLOUT_MAX_ACTIVE_SESSIONS}
   --rollout-release-on-postrun
   --rollout-min-complete-accept-fraction 0.8
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP}
   --pipeline-model-parallel-size ${PP}
   --context-parallel-size ${CP}
   --expert-model-parallel-size ${EP}
   --expert-tensor-parallel-size 1
   --sequence-parallel
   --chunked-lm-head
   # actor 和 critic 轮流用同一批卡，所以参数 buffer 也要让出去，不只是梯度。
   --offload-release-param-buffer
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
   --log-probs-chunk-size 64
   --seq-length ${SEQ_LENGTH}
   --train-env-vars '{"PYTORCH_NPU_ALLOC_CONF":"expandable_segments:True","VIME_MOE_STATS":"1","VIME_MOE_STATS_FILE":"/workspace/vime/'"${POLAR_OUTPUT_DIR}"'/debug/moe_stats.jsonl"}'
)

OPTIMIZER_ARGS=(
   --optimizer adam
   # actor 与 critic 共用。给 critic 单独 lr 要走 --megatron-config-path。
   --lr "${LR}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --optimizer-offload-fraction 1.0
   --use-precision-aware-optimizer
   # m/v 用 bf16 存储，节省约 495G host 内存（35B MoE 峰值接近 host 上限）。
   --exp-avg-dtype bf16
   --exp-avg-sq-dtype bf16
)

VLLM_ARGS=(
   # polar 发的是带 tool 的请求，引擎必须能吐结构化 tool call，否则 400。
   --vllm-tool-call-parser qwen3_coder
   --vllm-enable-auto-tool-choice
   --vllm-reasoning-parser qwen3
   # MoE 专有：指定 GDN backend 和 vLLM 模型架构
   --qwen-gdn-backend npu
   --model-name qwen3_5moeforconditionalgeneration
   --vllm-hf-overrides '{"architectures":["Qwen3_5MoeForConditionalGeneration"]}'
   --vllm-router-port 8001
   --no-vllm-weight-sync-packed
   --vllm-gpu-memory-utilization 0.8
   --vllm-max-num-seqs ${VLLM_MAX_NUM_SEQS}
   --vllm-max-num-batched-tokens ${VLLM_MAX_NUM_BATCHED_TOKENS}
   --vllm-max-model-len ${VLLM_MAX_MODEL_LEN}
   --vllm-enable-sleep-mode
   --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
   --vllm-enable-prefix-caching
   --vllm-enable-chunked-prefill
   --no-offload-rollout
   --vllm-weight-sync-mode native
   --vllm-additional-config '{"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":true}}'
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-flash-attn
   --moe-token-dispatcher-type alltoall
   --no-gradient-accumulation-fusion
   --seed 1234
)

ray stop --force
rm -rf "${RAY_TEMP_DIR}"

if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
   # head 节点：启动 Ray head，等所有 worker 加入后再起 train
   ray start --head --port "${RAY_PORT}" --dashboard-host=0.0.0.0 \
      --node-ip-address="${CURRENT_IP}" --dashboard-port="${RAY_DASHBOARD_PORT}" \
      --num-gpus="${NPUS_PER_NODE}" --resources='{"NPU": '"${NPUS_PER_NODE}"'}' \
      --temp-dir="${RAY_TEMP_DIR}" --object-store-memory=50000000000 --disable-usage-stats

   while true; do
      active_node_count=$(ray status | awk '
         /^Active:/ {in_active=1; next}
         /^Pending:/ {in_active=0}
         in_active && $1 == "1" && $2 ~ /^node_/ {count++}
         END {print count + 0}')
      echo "[stage] wait Ray nodes active=${active_node_count}/${NNODES}"
      [ "${active_node_count}" = "${NNODES}" ] && break
      sleep 5
   done

   # Ray 起来之后要取消这几个，否则 worker 会继承 head 的可见设备限制。
   unset ASCEND_RT_VISIBLE_DEVICES HCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME
   python3 train_async.py \
      --rollout-lb-proxy \
      "${TOPO_ARGS[@]}" "${MODEL_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${POLAR_ARGS[@]}" \
      "${OPTIMIZER_ARGS[@]}" "${ALGO_ARGS[@]}" "${PERF_ARGS[@]}" \
      "${VLLM_ARGS[@]}" "${MISC_ARGS[@]}" "${CKPT_ARGS[@]}" "${WANDB_ARGS[@]}" \
      2>&1 | tee "${LOG_FILE}"
else
   # worker：挂上去就结束，训练进程只在 head 上。
   while true; do
      ray start --address="${MASTER_ADDR}:${RAY_PORT}" \
         --node-ip-address="${CURRENT_IP}" \
         --num-gpus="${NPUS_PER_NODE}" --resources='{"NPU": '"${NPUS_PER_NODE}"'}' \
         --temp-dir="${RAY_TEMP_DIR}" --disable-usage-stats && break
      sleep 5
   done
   echo "[stage] worker ${CURRENT_IP} joined ${MASTER_ADDR}:${RAY_PORT}"
fi
