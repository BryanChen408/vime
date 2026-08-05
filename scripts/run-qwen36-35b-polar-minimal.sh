#!/bin/bash
# vime + polar 算子 RL 启动(qwen3.6-35B-A3B / NPU)。
# 前置:宿主机先用 profile.vime.yaml 起 polar,其推理端点指向 vime vllm 的 :${VLLM_ROUTER_PORT}。
# 卡位:polar agent 占 0-3;vime rollout(4-7)+ train(8-15)用 4-15。
set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
cd "${VIME_ROOT}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# ─── 运行标识 / 多节点 ───
RUN_ID=${RUN_ID:-qwen36_polar_$(date +%Y%m%d-%H%M%S)}
MASTER_ADDR=${MASTER_ADDR:-80.48.5.88}
CURRENT_IP=${CURRENT_IP:-}
SOCKET_IFNAME=${SOCKET_IFNAME:-}
NNODES=${NNODES:-1}
NPUS_PER_NODE=${NPUS_PER_NODE:-12}            # 单机:rollout 4 + train 8 = 卡 4-15
RAY_PORT=${RAY_PORT:-6460}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8290}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/ray_qwen36_vime_polar}

# ─── 拓扑(gpu 计数;或设 RESOURCE_LAYOUT 走显式钉位)───
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-4}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}

# ─── polar 数据 / 端点 ───
POLAR_OUTPUT_DIR=${POLAR_OUTPUT_DIR:-output/polar_bridge}
OPERATOR_DATA_ROOT=${OPERATOR_DATA_ROOT:-/mnt/share/c00937190/datasets/op_tasks/op_assets_cudallm_filtered189}
OPERATOR_TASK_JSONL=${OPERATOR_TASK_JSONL:-${OPERATOR_DATA_ROOT}/operator_tasks.jsonl}
OPERATOR_TASKS_DIR=${OPERATOR_TASKS_DIR:-${OPERATOR_DATA_ROOT}/op_tasks}
VLLM_ROUTER_PORT=${VLLM_ROUTER_PORT:-8001}    # profile.vime.yaml 推理端点指向它

# ─── 环境 ───
export PYTHONBUFFERED=16
export PYTHONPATH="/workspace/vllm:/workspace/vllm-ascend:/workspace/Megatron-LM:${VIME_ROOT}:${PYTHONPATH:-}"
# [复核-D 保留 2026-07-14] Ascend 自定义 MoE 训练算子库(moe_grouped_matmul/grouped_matmul_swiglu/swiglu)。
#   slime 在同名脚本 run-qwen36-35b-polar-minimal.sh:52 设的就是这条 identical 路径 → 参考正确,保留。
#   --moe-grouped-gemm 用它;此路径在本容器不存在时被 ld 直接跳过 → 回退原生实现(不崩,仅性能),故当前 inert。
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH:-}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TASK_QUEUE_ENABLE=0                     # 必须 0:=1 会让 GDN/ring-attn 训练出 NaN
export TORCHDYNAMO_DISABLE=1                   # 昇腾 inductor get_gpu_type() 断言 → 走 eager
export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}   # NPU 邻近 NUMA 绑核,降延迟抖动
export QWEN36_CP_MODE=ulysses
export QWEN36_CAUSAL_CONV1D_IMPL=triton
export QWEN36_CHUNK_LMHEAD=${QWEN36_CHUNK_LMHEAD:-0}   # =1 chunked LM-head logprob,长序列免 OOM
export VLLM_ASCEND_ENABLE_NZ=0                         # 必须 0:vllm-ascend wake_up 对 NZ+RL 硬 raise、weight-sync 每步换权重与 NZ 格式冲突(精度崩);推理加速收益 RL 下无法安全兑现
export VLLM_TOOL_CALL_PARSER=qwen3_coder
export VLLM_REASONING_PARSER=qwen3
# 本机 vllm = v0.21.0 tag + 2 commits → 自报 0.21.1.dev2,vllm-ascend 的 vllm_version_is("0.21.0")
# 会误判 → 走 vllm-main 才有的 expert_map_manager 导入 → ModuleNotFound。钉成 0.21.0。
export VLLM_VERSION=${VLLM_VERSION:-0.21.0}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=1
# HCCL(节点内 HCCS + 跨机 socket;长跑 EI0013 容错 + 35B 权重广播大 buffer)
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-2400}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-512}
#export HCCL_INTRA_ROCE_ENABLE=${HCCL_INTRA_ROCE_ENABLE:-1}
#export HCCL_INTRA_PCIE_ENABLE=${HCCL_INTRA_PCIE_ENABLE:-0}
# 跨机 HCCL 必需(对齐 slime;缺则双机权重同步 world>N 卡死在 rendezvous):
export HCCL_SOCKET_FAMILY=${HCCL_SOCKET_FAMILY:-AF_INET}       # 强制 IPv4(网卡带 IPv6 地址会 socket family mismatch)
export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}     # 禁 IP 白名单(否则跨机对端 IP 不在白名单→连接被拒→卡死)
# Ascend 要求 ASCEND_RT_VISIBLE_DEVICES 升序(乱序 → torch_npu 见 0 卡)
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7,8,9,10,11,12,13,14,15}
export POLAR_KEEP_SESSION_DIR=${POLAR_KEEP_SESSION_DIR:-1}
export POLAR_TRAJECTORY_PG_STRICT=${POLAR_TRAJECTORY_PG_STRICT:-1}
export POLAR_ANTHROPIC_DEFAULT_MAX_TOKENS=${POLAR_ANTHROPIC_DEFAULT_MAX_TOKENS:-12288}

source "${VIME_ROOT}/scripts/models/qwen3.5-35B-A3B.sh"     # → MODEL_ARGS

# [双机修复 2026-07-14] CURRENT_IP 优先从 SOCKET_IFNAME 指定的网卡取(对齐 slime polar-minimal:56),
#   避免 hostname -I 首个 IP 命中 docker/bridge/别的网卡 → 跨机 HCCL ranktable 检测拿错 IP → EI0015。
if [ -n "${SOCKET_IFNAME}" ]; then
   CURRENT_IP=${CURRENT_IP:-$(ip -o -4 addr show "${SOCKET_IFNAME}" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)}
   CURRENT_IP=${CURRENT_IP:-$(ifconfig "${SOCKET_IFNAME}" 2>/dev/null | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')}
fi
CURRENT_IP=${CURRENT_IP:-$(hostname -I | awk '{print $1}')}
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},${CURRENT_IP}${no_proxy:+,${no_proxy}}"
export NO_PROXY="${no_proxy}"
if [ -n "${SOCKET_IFNAME}" ]; then
   export HCCL_SOCKET_IFNAME="${SOCKET_IFNAME}"
   export GLOO_SOCKET_IFNAME="${SOCKET_IFNAME}"
fi

POLAR_ROLLOUT_URL=${POLAR_ROLLOUT_URL:-http://${MASTER_ADDR}:8080}
LOG_FILE=${LOG_FILE:-/mnt/share/c00937190/logs/train_${RUN_ID}.log}
mkdir -p logs "${POLAR_OUTPUT_DIR}" /mnt/share/c00937190/logs

# ─── 参数分组 ───
CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT:-/mnt/weight/Qwen3.6-35B-A3B}
   --ref-load ${REF_LOAD:-/mnt/weight/Qwen3.6-35B-A3B_torch_dist}
   --save ${SAVE:-/workspace/Qwen3.6-35B-A3B_vime_polar}/
   --save-interval 100
   --no-save-optim
   --megatron-to-hf-mode raw
   # FEAT_OPT2=1 → --optimization-level 2(对齐 slime 默认,激活 MindSpeed level-2 fusion,含 moe-permute
   #   的 dummy-TE 桩 + NPU 融合算子);默认 0 = 现 proven-healthy 态。opt2 是 blast-radius 项,须验 token-faith+GDN。
   --optimization-level "$([ "${FEAT_OPT2:-0}" = "1" ] && echo 2 || echo 0)"
)

TOPO_ARGS=(
   --actor-num-nodes ${ACTOR_NUM_NODES}
   --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE}
   --rollout-num-gpus ${ROLLOUT_NUM_GPUS}
   --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
)

ROLLOUT_ARGS=(
   --rollout-function-path vime_bridge.rollout.generate_rollout_polar_async
   --eval-function-path vime_bridge.rollout.generate_rollout_polar_async
   --prompt-data "${OPERATOR_TASK_JSONL}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --reward-key score
   --custom-reward-post-process-path vime_bridge.reward_post_process.post_process_rewards
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT:-1}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-4}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-32768}"
   --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-131072}"
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE:-32}"
   --save-debug-rollout-data "${POLAR_OUTPUT_DIR}/vime_debug_rollout_${RUN_ID}_{rollout_id}.pt"
   --save-debug-train-data "${POLAR_OUTPUT_DIR}/vime_debug_train_${RUN_ID}_rollout_{rollout_id}_{rank}.pt"
   --use-dynamic-global-batch-size
   --rollout-seed "${ROLLOUT_SEED:-42}"
)

POLAR_ARGS=(
   --polar-url "${POLAR_ROLLOUT_URL}"
   --polar-run-id "${RUN_ID}"
   --polar-reward-key score
   --polar-task-id-template "{args.polar_run_id}-polar-op-{rollout_id}-{sample.group_index}"
   --operator-tasks-dir "${OPERATOR_TASKS_DIR}"
   --rollout-max-async-level "${POLAR_MAX_ASYNC_LEVEL:-1}"
   --rollout-request-timeout "${POLAR_ROLLOUT_REQUEST_TIMEOUT:-8000}"
   --rollout-scheduler-mode session_pool
   --rollout-max-active-sessions "${POLAR_MAX_ACTIVE_SESSIONS:-16}"
   --rollout-release-on-postrun
   --rollout-min-complete-accept-fraction "${POLAR_MIN_COMPLETE_ACCEPT_FRACTION:-0.8}"
)

# [FLOOR] 轨迹内 trace 保底权重(vime/ray/rollout.py 的 rollout_mask_sums 分母)。
# 不设则不传该参,默认纯 token 加权,行为与之前完全一致。
[ -n "${POLAR_TRAJECTORY_PG_FLOOR:-}" ] && \
   POLAR_ARGS+=(--polar-trajectory-pg-floor "${POLAR_TRAJECTORY_PG_FLOOR}")

PERF_ARGS=(
   --tensor-model-parallel-size "${TP:-2}"
   --pipeline-model-parallel-size "${PP:-1}"
   --context-parallel-size "${CP:-8}"
   --expert-model-parallel-size "${EP:-8}"
   --expert-tensor-parallel-size 1
   --sequence-parallel
   --chunked-lm-head
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-512}"
   --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-64}"
   --seq-length "${SEQ_LENGTH:-131072}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   #--use-kl-loss
   #--kl-loss-coef 0.001
   #--kl-loss-type low_var_kl
   --entropy-coef 0.001
   --eps-clip 0.2
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

VLLM_ARGS=(
   --rollout-backend vllm
   --qwen-gdn-backend npu
   --model-name qwen3_5moeforconditionalgeneration
   --vllm-hf-overrides '{"architectures":["Qwen3_5MoeForConditionalGeneration"]}'
   --vllm-router-port "${VLLM_ROUTER_PORT}"
   --vllm-weight-sync-mode native
   --no-vllm-weight-sync-packed
   --vllm-gpu-memory-utilization "${VLLM_GPU_MEM_UTIL:-0.8}"
   --vllm-max-num-seqs "${VLLM_MAX_NUM_SEQS:-96}"
   --vllm-max-model-len "${VLLM_MAX_MODEL_LEN:-131072}"
   --vllm-enable-sleep-mode
   --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
   --no-offload-train
   --no-offload-rollout
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-flash-attn
   --moe-token-dispatcher-type alltoall
   # [复核-B 回退 2026-07-14] slime 在**所有** NPU 脚本都用 --no-gradient-accumulation-fusion
   #   (grad-fusion 依赖 CUDA-only 的 fused_weight_gradient_mlp_cuda,NPU 无)。
   #   之前注释里"slime 开着跑通"是错的,已回退对齐 slime。
   --no-gradient-accumulation-fusion
   --seed "${SEED:-1234}"
)

# ─── 特性开关(默认全 OFF = baseline 逐位不变)───
[ "${FEAT_ASYNC_SCHED:-0}" = "1" ] && VLLM_ARGS+=(--vllm-async-scheduling)
EP_ON=0
# FEAT_CROSS_DP_EP=1:跨 DP EP —— DP(external-LB)+ EP 同开,EP world=dp×tp。vLLM 自动 flatten
#   ep_size=dp×tp(config.py:1179/1202),FusedMoE loader 按 dp×tp 分片(复用已验 c04b1dea4,无需新补丁)。
#   硬需 FEAT_DP_EXTERNAL_LB=1(否则只是引擎内 EP);建议同开 FEAT_BALANCE_SCHED(防跨 DP EP batch 拖尾,§20)。
if [ "${FEAT_ROLLOUT_EP:-0}" = "1" ] || [ "${FEAT_FLASHCOMM1:-0}" = "1" ] || [ "${FEAT_CROSS_DP_EP:-0}" = "1" ]; then
   VLLM_ARGS+=(--vllm-enable-expert-parallel); EP_ON=1     # FlashComm1 硬需 rollout EP;CROSS_DP_EP 也需
fi
if [ "${FEAT_CROSS_DP_EP:-0}" = "1" ]; then
   [ "${FEAT_DP_EXTERNAL_LB:-0}" = "1" ] || { echo "[cross-dp-ep][FATAL] 需同开 FEAT_DP_EXTERNAL_LB=1(EP world=dp×tp 依赖 DP 先到位)" >&2; exit 1; }
   [ "${FEAT_BALANCE_SCHED:-0}" = "1" ] || echo "[cross-dp-ep][WARN] 建议同开 FEAT_BALANCE_SCHED=1(防跨 DP EP batch 不均拖尾,§20)" >&2
fi
[ "${FEAT_FLASHCOMM1:-0}" = "1" ] && export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
[ "${FEAT_PREFIX_CACHE:-0}" = "1" ] && VLLM_ARGS+=(--vllm-enable-prefix-caching --vllm-enable-chunked-prefill)   # align 模式硬依赖 chunked-prefill
# FEAT_DP_EXTERNAL_LB=1:vLLM 原生 external-LB 分布式 DP —— 每引擎 = 一个 DP rank,各自 API server + 前置 LB。
#   **DP 组大小由 layout 的 rollout.vllm_dp_size 唯一决定**(arguments.py 消费 + 校验 == 引擎数);脚本只置模式
#   开关。--data-parallel-external-lb 经 _forward_vllm_cli_args 自动带到 vllm serve;per-rank
#   --data-parallel-rank/-address/-rpc-port 由 vime 运行时分配(rollout.py #3)。默认 OFF = baseline 逐位不变。
if [ "${FEAT_DP_EXTERNAL_LB:-0}" = "1" ]; then
   if [ "${FEAT_LB_PROXY:-0}" != "1" ]; then
      echo "[dp-extlb][FATAL] external-LB DP 需前置 LB 分发各 rank API server,请同开 FEAT_LB_PROXY=1" >&2; exit 1
   fi
   VLLM_ARGS+=(--vllm-data-parallel-external-lb)
fi
# 多特性各自贡献 additional-config 顶层键 → 合并成单个 JSON(否则重复 flag 后者覆盖前者)
ADDCFG_PARTS=()
[ "${FEAT_MULTISTREAM_SHARED_EXPERT:-0}" = "1" ] && ADDCFG_PARTS+=('"multistream_overlap_shared_expert":true')
[ "${FEAT_STATIC_KERNEL:-0}" = "1" ] && ADDCFG_PARTS+=('"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":true}')
# FEAT_BALANCE_SCHED=1:跨 DP rank 均衡调度(§20)——每 engine step 后 all_gather 各 rank 运行请求数,
#   最忙副本打满时本副本停接 WAITING,防跨 DP EP batch 不均拖尾。DP-only:dp_size=1 时是无害 no-op;
#   与 PD 分离(kv_producer/consumer)互斥(vllm-ascend 会 raise),纯 DP(kv_transfer_config=None)✅。
#   纯调度层、理论不破 token-faith(§20.4)。建议随 FEAT_DP_EXTERNAL_LB 一起开。
[ "${FEAT_BALANCE_SCHED:-0}" = "1" ] && ADDCFG_PARTS+=('"enable_balance_scheduling":true')
[ "${#ADDCFG_PARTS[@]}" -gt 0 ] && VLLM_ARGS+=(--vllm-additional-config "{$(IFS=,; echo "${ADDCFG_PARTS[*]}")}")
[ "${FEAT_HCCL_AIV:-0}" = "1" ] && export HCCL_OP_EXPANSION_MODE=AIV
[ "${REPRO_DETERMINISTIC:-0}" = "1" ] && VLLM_ARGS+=(--vllm-enable-deterministic-inference)
# [vime 2026-07-15 EXPERIMENTAL] FEAT_TRAIN_EXPANDABLE=1:把 expandable_segments forward 进训练 actor
#   (ray actor 不继承父 env,必须经 --train-env-vars)。这是 OOM 所在的 88 训练侧,不碰 vLLM/CaMem/assert。
#   默认 0 = 不变。与 vLLM 侧的 VIME_VLLM_KEEP_EXPANDABLE 独立。
[ "${FEAT_TRAIN_EXPANDABLE:-0}" = "1" ] && PERF_ARGS+=(--train-env-vars "{\"PYTORCH_NPU_ALLOC_CONF\":\"${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}\"}")
# [vime 2026-07-15 B] FEAT_OPT2=1:对齐 slime 开 opt-level 2 + moe-permute-fusion(NPU 融合算子省 unfused
#   MoE permute 的中间张量 + workspace)。opt-level 值在上面数组处按 FEAT_OPT2 切;此处补 --moe-permute-fusion。
#   前置已验:torch_npu 2.10.0 有 npu_moe_token_permute_with_routing_map。须验:model init 不崩 + token-faith + GDN。
[ "${FEAT_OPT2:-0}" = "1" ] && PERF_ARGS+=(--moe-permute-fusion)
echo "[feat] async=${FEAT_ASYNC_SCHED:-0} flashcomm1=${FEAT_FLASHCOMM1:-0} ep=${EP_ON} prefix_cache=${FEAT_PREFIX_CACHE:-0} multistream=${FEAT_MULTISTREAM_SHARED_EXPERT:-0} static_kernel=${FEAT_STATIC_KERNEL:-0} hccl_aiv=${FEAT_HCCL_AIV:-0} lb_proxy=${FEAT_LB_PROXY:-0} dp_external_lb=${FEAT_DP_EXTERNAL_LB:-0} balance_sched=${FEAT_BALANCE_SCHED:-0} train_expandable=${FEAT_TRAIN_EXPANDABLE:-0} vllm_keep_expandable=${VIME_VLLM_KEEP_EXPANDABLE:-0} opt2=${FEAT_OPT2:-0} cross_dp_ep=${FEAT_CROSS_DP_EP:-0}"

# ─── 清本节点 rollout 卡残留(只清 $ASCEND_RT_VISIBLE_DEVICES 钉的卡)───
# 上个 run 异常结束后,vllm 栈(ray::VLLMEngine / vllm serve / EngineCore / Worker)
# 可能变成孤儿进程:占着 HBM(实测 55GB/卡)和 15000 端口,ray stop 清不干净。
# 判定 = 进程 env 的 ASCEND_RT_VISIBLE_DEVICES 与本机 rollout 卡集求交;
# Ascend 运行时不持 /dev/davinciN 句柄,fuser/fd 扫描不可靠,勿用。
# 模式里禁止裸写 ray:::head 节点的 ray::IDLE 预启动 worker 池(约 200 个,env 全卡)
# 会被扫进来 —— 它们不占 HBM/端口,杀之无益还会误伤共享集群的 worker 池。
# 仅在 ray stop 之后、ray start 之前调用:此时卡上的 vllm 按定义都是残留,
# 函数不区分残留与活 run,禁止单独手动执行。
cleanup_rollout_residue() {
   local devs="${ASCEND_RT_VISIBLE_DEVICES//[[:space:]]/}"
   [ -n "$devs" ] || { echo "[cleanup] ASCEND_RT_VISIBLE_DEVICES 为空,跳过"; return 0; }
   local pid env_devs overlap pass hit
   for pass in TERM KILL; do
      hit=0
      for pid in $(pgrep -fi "vllm" 2>/dev/null); do
         env_devs=$(tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null | sed -n 's/^ASCEND_RT_VISIBLE_DEVICES=//p' | head -1)
         [ -n "$env_devs" ] || continue
         overlap=$(printf '%s\n%s\n' "$devs" "$env_devs" | tr ',' '\n' | grep -E '^[0-9]+$' | sort -n | uniq -d | head -1)
         [ -n "$overlap" ] || continue
         echo "[cleanup] kill -$pass pid=$pid cards=[$env_devs] $(ps -o comm= -p "$pid" 2>/dev/null)"
         kill -"$pass" "$pid" 2>/dev/null || true
         hit=1
      done
      [ "$hit" = 0 ] && break
      [ "$pass" = TERM ] && sleep 8
   done
   return 0
}

# ─── Ray(单机 NNODES=1 走 head 分支)+ 启动 ───
if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
   ray stop --force
   rm -rf "${RAY_TEMP_DIR}"
   cleanup_rollout_residue
   ray start --head --port "${RAY_PORT}" --dashboard-host=0.0.0.0 --node-ip-address="${CURRENT_IP}" --dashboard-port="${RAY_DASHBOARD_PORT}" --num-gpus="${NPUS_PER_NODE}" --resources='{"NPU": '"${NPUS_PER_NODE}"'}' --temp-dir="${RAY_TEMP_DIR}" --disable-usage-stats

   while true; do
      active_node_count=$(ray status | awk '
         /^Active:/ {in_active=1; next}
         /^Pending:/ {in_active=0}
         in_active && $1 == "1" && $2 ~ /^node_/ {count++}
         END {print count + 0}')
      echo "[stage] wait Ray nodes active=${active_node_count}/${NNODES}"
      if [ "$active_node_count" -eq "$NNODES" ]; then
         # layout 路径:清全局可见卡,交给 Ray 按 actor 钉卡
         unset ASCEND_RT_VISIBLE_DEVICES HCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME
         EXTRA_ARGS=()
         [ -n "${RESOURCE_LAYOUT:-}" ] && EXTRA_ARGS+=(--resource-layout "${RESOURCE_LAYOUT}")
         # FEAT_LB_PROXY=1:Python 透传 LB proxy 替 Rust router(保 return_token_ids + 会话亲和);
         #   需把 polar 推理端点指向 :${VLLM_ROUTER_PORT}。见 docs/design/router_return_token_ids_passthrough.md §10。
         [ "${FEAT_LB_PROXY:-0}" = "1" ] && EXTRA_ARGS+=(--rollout-lb-proxy)
         python3 train_async.py \
            ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
            ${TOPO_ARGS[@]} ${MODEL_ARGS[@]} ${ROLLOUT_ARGS[@]} ${POLAR_ARGS[@]} \
            ${OPTIMIZER_ARGS[@]} ${GRPO_ARGS[@]} ${PERF_ARGS[@]} ${VLLM_ARGS[@]} \
            ${MISC_ARGS[@]} ${CKPT_ARGS[@]} \
            2>&1 | tee "${LOG_FILE}"
         break
      fi
      sleep 5
   done
else
   ray stop --force
   rm -rf "${RAY_TEMP_DIR}"
   cleanup_rollout_residue
   while true; do
      ray start --address="${MASTER_ADDR}:${RAY_PORT}" --node-ip-address="${CURRENT_IP}" --num-gpus="${NPUS_PER_NODE}" --resources='{"NPU": '"${NPUS_PER_NODE}"'}' --temp-dir="${RAY_TEMP_DIR}" --disable-usage-stats
      ray status && break
      sleep 5
   done
fi
