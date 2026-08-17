#!/bin/bash
# Qwen3.5-35B-A3B 中间 checkpoint 单机验证。
# 只启动 vLLM 和 Polar rollout，不创建训练 actor/critic。
#
# Ray、vLLM proxy 和 Polar 均使用 LOCAL_IP；Polar 服务需先在本机 :8080 启动。

# ray stop 不回收这些：VLLM:: worker 占着显存，lb_proxy 占着 router 端口。
# 只动 vime 自己的进程，不碰 :8080 上的 polar。
pkill -9 -f 'VLL[M]::' 2>/dev/null || true
pkill -9 -f 'vime.ray.lb_proxy' 2>/dev/null || true
sleep 2

set -exo pipefail

VIME_ROOT=/workspace/vime
cd "${VIME_ROOT}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

# ─────────────────────────────────────────────────────────────────────
# 常用实验参数（需要调整时只改这里）
# ─────────────────────────────────────────────────────────────────────

# 本机配置
NPUS_PER_NODE=12
SOCKET_IFNAME="enp91s0f3"
LOCAL_IP="80.48.5.66"
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7,8,9,10,11,12,13,14,15

ROLLOUT_NUM_GPUS=12
POLAR_URL="http://80.48.5.66:8080"

# 验证规模
NUM_ROLLOUT=1
ROLLOUT_BATCH_SIZE=16
N_SAMPLES_PER_PROMPT=1
ROLLOUT_MAX_ACTIVE_SESSIONS=16

# 数据
OPERATOR_TASK_JSONL=/home/docker/rl_ops1_2_simple/holdout16_hard.jsonl
OPERATOR_TASKS_DIR=/home/docker/rl_ops1_2_simple/op_tasks

# 权重
HF_CKPT=/home/docker/vime_ppo35b_eval_iter49/hf_checkpoint

# ─────────────────────────────────────────────────────────────────────
# 下面一般不用动
# ─────────────────────────────────────────────────────────────────────

RUN_ID=qwen36_35b_ppo_$(date +%Y%m%d-%H%M%S)
RAY_PORT=6460
RAY_DASHBOARD_PORT=8290
RAY_TEMP_DIR=/home/docker/vime_ppo35b_eval_iter49/ray
POLAR_OUTPUT_DIR=/home/docker/vime_ppo35b_eval_iter49/output

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

CURRENT_IP="${LOCAL_IP}"
export no_proxy="127.0.0.1,localhost,${CURRENT_IP}${no_proxy:+,${no_proxy}}"
export NO_PROXY="${no_proxy}"
export HCCL_SOCKET_IFNAME="${SOCKET_IFNAME}"
export GLOO_SOCKET_IFNAME="${SOCKET_IFNAME}"

LOG_FILE=/home/docker/vime_ppo35b_eval_iter49/logs/eval_${RUN_ID}.log
mkdir -p "${POLAR_OUTPUT_DIR}" /home/docker/vime_ppo35b_eval_iter49/logs

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
)

TOPO_ARGS=(
   --rollout-num-gpus ${ROLLOUT_NUM_GPUS}
   --rollout-num-gpus-per-engine 4
   --num-gpus-per-node "${NPUS_PER_NODE}"
)

ROLLOUT_ARGS=(
   --rollout-function-path vime_bridge.rollout.generate_rollout_polar_async
   --prompt-data "${OPERATOR_TASK_JSONL}"
   --eval-prompt-data holdout "${OPERATOR_TASK_JSONL}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --reward-key score
   --custom-reward-post-process-path vime_bridge.reward_post_process.post_process_rewards
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len 32768
   --rollout-max-context-len 131072
   --rollout-temperature 0.7
   --rollout-seed 42
   # 每步落盘，供事后分析。
   --save-debug-rollout-data "${POLAR_OUTPUT_DIR}/${RUN_ID}/debug/{rollout_id}.pt"
)

POLAR_ARGS=(
   --polar-url "${POLAR_URL}"
   --polar-run-id "${RUN_ID}"
   --polar-reward-key score
   --polar-task-id-template "{args.polar_run_id}-polar-op-{rollout_id}-{sample.group_index}"
   --operator-tasks-dir "${OPERATOR_TASKS_DIR}"
   --rollout-max-async-level 1
   --rollout-request-timeout 8000
   --rollout-scheduler-mode session_pool
   --rollout-max-active-sessions ${ROLLOUT_MAX_ACTIVE_SESSIONS}
   --rollout-release-on-postrun
   --rollout-min-complete-accept-fraction 0.8
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
   --vllm-gpu-memory-utilization 0.8
   --vllm-max-num-seqs 96
   --vllm-max-model-len 131072
   --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
   --vllm-enable-prefix-caching
   --vllm-enable-chunked-prefill
   --vllm-additional-config '{"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":true}}'
)

ray stop --force
rm -rf "${RAY_TEMP_DIR}"

echo "[stage] local_ip=${CURRENT_IP} polar=${POLAR_URL} vllm=http://${CURRENT_IP}:8001"
ray start --head --port "${RAY_PORT}" --dashboard-host=0.0.0.0 \
   --node-ip-address="${CURRENT_IP}" --dashboard-port="${RAY_DASHBOARD_PORT}" \
   --num-gpus="${NPUS_PER_NODE}" --resources='{"NPU": '"${NPUS_PER_NODE}"'}' \
   --temp-dir="${RAY_TEMP_DIR}" --object-store-memory=50000000000 --disable-usage-stats

# Polar topology 中的 inference.base_url 也必须是 http://${CURRENT_IP}:8001。
unset ASCEND_RT_VISIBLE_DEVICES HCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME
export RAY_ADDRESS="${CURRENT_IP}:${RAY_PORT}"
python3 experiments/evaluation/evaluation.py \
   --debug-rollout-only \
   --rollout-lb-proxy \
   ${TOPO_ARGS[@]} ${ROLLOUT_ARGS[@]} ${POLAR_ARGS[@]} \
   ${VLLM_ARGS[@]} ${CKPT_ARGS[@]} \
   2>&1 | tee "${LOG_FILE}"
