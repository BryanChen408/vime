#!/bin/bash
# vime + polar SWE-Gym coding-agent RL 启动(Qwen3-8B / dense / NPU / codex harness)。
#   派生自 run-qwen3-8b-math.sh(同一 dense-8B + polar-agentic + LB-proxy 基座;模型/并行/vLLM 块一字不改)。
#   仅换域 math→SWE。关键差异 = 提交模式走 task_request(--polar-task-template),而非 math 的 operator_samples
#   —— 因 SWE 每个 instance 的 docker 镜像不同,需按 sample.metadata 渲染 runtime.image
#   (与 jxliu swegym polar_config.yaml 的 polar_task_template 一致)。
#
# ── 与 35B-math 的差异(仅"模型相关"块,polar/eval/LB/session_pool/vime_bridge 一字不改)──
#   · 模型: source models/qwen3-8B.sh(dense 36L/hidden4096/GQA8)替 qwen3.5-35B-A3B.sh(MoE+GDN)
#   · 权重: HF=/home/docker/Qwen3-8B  REF_LOAD=/home/docker/Qwen3-8B_torch_dist(convert-qwen3-8B.sh 产物)
#   · 并行: TP=2 / PP=1 / CP=1 / EP=1(dense 无 MoE/context-parallel);去 --chunked-lm-head(MoE 专属);log-probs-chunk-size 256 保留(dense 也用)
#   · vLLM: 去 --qwen-gdn-backend / --model-name qwen3_5moe / --vllm-hf-overrides(dense=Qwen3ForCausalLM,vLLM 自识别)
#   · MISC: 去 --moe-token-dispatcher-type(dense 无 MoE 分发)
#   · env:  去 QWEN36_CP_MODE / QWEN36_CAUSAL_CONV1D_IMPL / QWEN36_CHUNK_LMHEAD(GDN/35B 专属)
#   · 卡位: 默认 rollout 2 + train 2 = 卡 4-7(8B 小;polar 仍占 0-3);其余全同。
#   dense 数值取自已跑通的 vime-polar/scripts/run_qwen3_8b_npu0_3.sh(TP2/CP1/EP1/recompute)。
#   由 scripts/swe/start_swegym_8b.sh 调用(注入数据/规模/端点 env);勿手工直接跑。
set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"   # scripts/swe -> vime root (2 levels up)
cd "${VIME_ROOT}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# ─── 运行标识 / 多节点 ───
RUN_ID=${RUN_ID:-swegym_codex_8b_$(date +%Y%m%d-%H%M%S)}
MASTER_ADDR=${MASTER_ADDR:-80.48.5.88}
CURRENT_IP=${CURRENT_IP:-}
SOCKET_IFNAME=${SOCKET_IFNAME:-}
NNODES=${NNODES:-1}
NPUS_PER_NODE=${NPUS_PER_NODE:-4}             # 8B:rollout 2 + train 2 = 卡 4-7
RAY_PORT=${RAY_PORT:-6460}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8290}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/ray_qwen3_8b_vime_polar}

# ─── 拓扑(gpu 计数;8B dense TP=2,小)───
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-2}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-2}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}

# ─── polar 数据 / 端点 ───
POLAR_OUTPUT_DIR=${POLAR_OUTPUT_DIR:-output/polar_bridge}
# SWE 训练数据(Phase 3 产物:每行含 metadata.docker_image / instance,swebench_harness 用)。
SWEGYM_JSONL=${SWEGYM_JSONL:-/home/docker/datasets/swegym/swegym_train_64.jsonl}
# task_request 模板:.in →(envsubst ${AGENT_CLI_DIR})→ .yaml;vime 再按 sample 渲染 {sample.metadata.*}。
SWE_TASK_TEMPLATE_IN=${SWE_TASK_TEMPLATE_IN:-${SCRIPT_DIR}/swe_task_template.yaml.in}
SWE_TASK_TEMPLATE=${SWE_TASK_TEMPLATE:-${VIME_ROOT}/${POLAR_OUTPUT_DIR}/swe_task_template.yaml}
# codex CLI 目录(node + @openai/codex),挂进任务容器 /opt/node:ro。
AGENT_CLI_DIR=${AGENT_CLI_DIR:-/home/docker/datasets/swe_agent_cli}
VLLM_ROUTER_PORT=${VLLM_ROUTER_PORT:-8001}    # profile.swe-8b.yaml 推理端点指向它

# ─── [稳定性] 清上一轮残留 LB proxy(每次训练启动都清)─────────────────────────
#   proxy 是 vime RolloutManager 起的子进程,`ray stop` 杀不掉它 → 旧 proxy 一直占着
#   :${VLLM_ROUTER_PORT},本轮新 proxy 绑不上(Errno 98 address already in use,只默默打
#   "LB proxy up" 但没真绑)→ polar 的 :8001 请求全打到那个僵尸 proxy(过期引擎/退化路由)
#   → 只灌一个引擎(排查根因即此:僵尸 proxy 可存活数天)。这里 pkill 进程 + fuser 放端口双保险。
#   math/operator 不同时跑同一 :8001,启动 math 时算子必然没在跑 → 不误伤 cannbot。set -ex 下全部 || true。
#   [d]p_... 方括号写法:正则匹配真实进程的 dp_...,但不匹配 pkill/祖先 argv 里的字面量 [d]p_...(自排除,防误杀自身)。
pkill -9 -f "[d]p_load_balance_proxy_server" 2>/dev/null || true
command -v fuser >/dev/null 2>&1 && fuser -k "${VLLM_ROUTER_PORT}/tcp" 2>/dev/null || true
sleep 1
echo "[cleanup] 已清残留 LB proxy + 放端口 :${VLLM_ROUTER_PORT}(ray stop 管不到这个)"

# ─── 环境 ───
export PYTHONBUFFERED=16
export PYTHONPATH="/workspace/vllm:/workspace/vllm-ascend:/workspace/Megatron-LM:${VIME_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH:-}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TASK_QUEUE_ENABLE=0                     # 两份 ref 都设 0;dense 无害,保留对齐
export TORCHDYNAMO_DISABLE=1                   # 昇腾 inductor get_gpu_type() 断言 → 走 eager
export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}
export VLLM_ASCEND_ENABLE_NZ=0                 # RL 权重同步每步换权重与 NZ 冲突,必须 0
# Qwen3-8B(instruct,非 Coder)发 Hermes 式 <tool_call> → 默认 hermes parser;
#   4B(Qwen3.5)发 <function=> XML → start_swegym_4b.sh 设 VLLM_TOOL_CALL_PARSER=qwen3_coder(治 8B hermes JSON 报错)。
export VLLM_TOOL_CALL_PARSER=${VLLM_TOOL_CALL_PARSER:-hermes}
export VLLM_REASONING_PARSER=qwen3
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=1
# HCCL(节点内 HCCS + 跨机 socket)
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-2400}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-512}
export HCCL_INTRA_ROCE_ENABLE=${HCCL_INTRA_ROCE_ENABLE:-1}
export HCCL_INTRA_PCIE_ENABLE=${HCCL_INTRA_PCIE_ENABLE:-0}
export HCCL_SOCKET_FAMILY=${HCCL_SOCKET_FAMILY:-AF_INET}
export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}
# Ascend 要求 ASCEND_RT_VISIBLE_DEVICES 升序;8B 默认用 4-7(polar 占 0-3)
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7}
export POLAR_KEEP_SESSION_DIR=${POLAR_KEEP_SESSION_DIR:-1}
export POLAR_TRAJECTORY_PG_STRICT=${POLAR_TRAJECTORY_PG_STRICT:-1}
export POLAR_ANTHROPIC_DEFAULT_MAX_TOKENS=${POLAR_ANTHROPIC_DEFAULT_MAX_TOKENS:-12288}

source "${VIME_ROOT}/scripts/models/${MODEL_ARGS_SCRIPT:-qwen3-8B.sh}"   # → MODEL_ARGS(默认 dense 8B;4B 设 MODEL_ARGS_SCRIPT=qwen3.5-4B.sh)

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
LOG_FILE=${LOG_FILE:-/home/docker/logs/train_${RUN_ID}.log}
mkdir -p logs "${POLAR_OUTPUT_DIR}" /home/docker/logs

# ─── task_request 模板渲染:替字面 ${AGENT_CLI_DIR} + 代理占位(保留 $HOME 给容器内展开)。
#   用 sed 而非 envsubst —— 本机无 gettext/envsubst;sed 精确匹配,不碰 $HOME。
#   代理:eval 容器默认无代理、直连 PyPI 超时(000)→ pip install -e . 的 build isolation 卡死重试
#   到超时(实测加代理后 ~3s)。⚠️代理【只】能注入 eval 容器,【绝不】export 到本 shell/进程环境 ——
#   否则 polar/vllm 会继承 http_proxy,vllm 的 :8001 内部推理请求走华为代理就坏了。故从独立文件读、
#   用局部变量(不 export)、仅 sed 注入 task_template。密码只落该文件(非 tracked),不进终端、不进进程。
SWE_EVAL_PROXY_FILE=${SWE_EVAL_PROXY_FILE:-/home/docker/.swe_eval_proxy}
_EVAL_PROXY=""
[ -f "${SWE_EVAL_PROXY_FILE}" ] && _EVAL_PROXY=$(head -1 "${SWE_EVAL_PROXY_FILE}" 2>/dev/null | tr -d '[:space:]')
if [ -z "${_EVAL_PROXY}" ]; then
   echo "[swe][WARN] ${SWE_EVAL_PROXY_FILE} 空/缺 → eval 容器无代理 → 评测 pip install -e . 会卡死到超时。" \
        "把代理 URL(http://user:pwd@ip:port)写进该文件(单行);它只注入 eval 容器,不碰终端/polar/vllm。" >&2
fi
# ⚠️ no_proxy 必须排掉所有内网推理端点(80.48.5.88 的 :8080/:8100/:8001)——否则容器里的
#   codex 请求 $OPENAI_BASE_URL(内网 vllm)会走华为外网代理、连不上 → traces=0 推理全崩。
#   只有外网 PyPI(eval 容器 pip)才该走代理。多机时把其它节点 IP 也加进来。
_EVAL_NO_PROXY="127.0.0.1,localhost,.huawei.com,.local,80.48.5.88,${MASTER_ADDR:-80.48.5.88}"
mkdir -p "$(dirname "${SWE_TASK_TEMPLATE}")"
sed -e "s|\${AGENT_CLI_DIR}|${AGENT_CLI_DIR}|g" \
    -e "s|\${HTTP_PROXY}|${_EVAL_PROXY}|g" \
    -e "s|\${HTTPS_PROXY}|${_EVAL_PROXY}|g" \
    -e "s|\${NO_PROXY}|${_EVAL_NO_PROXY}|g" \
    "${SWE_TASK_TEMPLATE_IN}" > "${SWE_TASK_TEMPLATE}"
if grep -q '\${AGENT_CLI_DIR}' "${SWE_TASK_TEMPLATE}"; then
   echo "[swe][FATAL] AGENT_CLI_DIR 未替换干净: ${SWE_TASK_TEMPLATE}" >&2; exit 1
fi
echo "[swe] rendered task template -> ${SWE_TASK_TEMPLATE} (AGENT_CLI_DIR=${AGENT_CLI_DIR})"

# ─── 参数分组 ───
CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT:-/home/docker/Qwen3-8B}
   --ref-load ${REF_LOAD:-/home/docker/Qwen3-8B_torch_dist}
   --save ${SAVE:-/workspace/Qwen3-8B_vime_swegym}/
   --save-interval 100
   --no-save-optim
   --megatron-to-hf-mode raw
   --optimization-level 0
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
   --prompt-data "${SWEGYM_JSONL}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --reward-key score
   --custom-reward-post-process-path vime_bridge.reward_post_process.post_process_rewards
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT:-1}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-4}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-16000}"
   --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-32768}"
   --rollout-temperature 0.7
   --global-batch-size "${GLOBAL_BATCH_SIZE:-32}"
   --use-dynamic-global-batch-size
   --rollout-seed "${ROLLOUT_SEED:-42}"
)

# ── validation/eval:设了 EVAL_TASK_JSONL 才开 —— 每 EVAL_INTERVAL 步在留出集(AIME)上评一次。
if [ -n "${EVAL_TASK_JSONL:-}" ]; then
   ROLLOUT_ARGS+=(
      --eval-prompt-data "${EVAL_TASK_JSONL}"
      --eval-interval "${EVAL_INTERVAL:-10}"
      --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL:-8}"
   )
fi

POLAR_ARGS=(
   --polar-url "${POLAR_ROLLOUT_URL}"
   --polar-run-id "${RUN_ID}"
   --polar-reward-key score
   --polar-task-template "${SWE_TASK_TEMPLATE}"
   --polar-task-id-template "{args.polar_run_id}-polar-swe-{rollout_id}-{sample.group_index}"
   --rollout-max-async-level "${POLAR_MAX_ASYNC_LEVEL:-1}"
   --rollout-request-timeout "${POLAR_ROLLOUT_REQUEST_TIMEOUT:-8000}"
   --rollout-scheduler-mode session_pool
   --rollout-max-active-sessions "${POLAR_MAX_ACTIVE_SESSIONS:-16}"
   --rollout-release-on-postrun
   --rollout-min-complete-accept-fraction "${POLAR_MIN_COMPLETE_ACCEPT_FRACTION:-0.8}"
)

# 8B dense:去 CP/EP/chunked-lm-head(MoE/GDN 专属),值取自 8B ref。
# log-probs-chunk-size 256 对齐同事(dense 也用,分块算 logprob 省显存,非 MoE 专属)。
PERF_ARGS=(
   --tensor-model-parallel-size "${TP:-2}"
   --pipeline-model-parallel-size "${PP:-1}"
   --context-parallel-size "${CP:-1}"
   --expert-model-parallel-size "${EP:-1}"
   --expert-tensor-parallel-size 1
   --sequence-parallel
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-4096}"
   --log-probs-chunk-size 256
   --micro-batch-size 1
   --seq-length "${SEQ_LENGTH:-32768}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --normalize-advantages
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
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

# 8B dense:去 --qwen-gdn-backend / --model-name / --vllm-hf-overrides;vLLM 从 config.json 自识别 Qwen3ForCausalLM。
VLLM_ARGS=(
   --rollout-backend vllm
   --vllm-router-port "${VLLM_ROUTER_PORT}"
   --vllm-weight-sync-mode native
   --no-vllm-weight-sync-packed
   --vllm-gpu-memory-utilization "${VLLM_GPU_MEM_UTIL:-0.6}"
   --vllm-max-num-seqs "${VLLM_MAX_NUM_SEQS:-32}"
   --vllm-max-model-len "${VLLM_MAX_MODEL_LEN:-32768}"
   --vllm-enable-sleep-mode
   --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
   --no-offload-train
   --no-offload-rollout
)

# 8B dense:去 --moe-token-dispatcher-type(无 MoE)。
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-flash-attn
   --no-gradient-accumulation-fusion
   --seed "${SEED:-1234}"
)

# ─── 特性开关(默认全 OFF = baseline;仅保留 dense+polar 相关,MoE 专属项已删)───
[ "${FEAT_ASYNC_SCHED:-0}" = "1" ] && VLLM_ARGS+=(--vllm-async-scheduling)
# 4B(Qwen3.5,VLM+GDN)专属:FEAT_GDN=1 补 GDN backend + VLM model-name/hf-overrides —— 照抄用户已验证
#   能跑的 run-qwen36-35b-polar-ppo.sh(35B MoE+GDN),4B 用 dense 版 model-name/architectures。
#   8B dense 默认 OFF(vLLM 从 config.json 自识别 Qwen3ForCausalLM,不需 GDN backend)。
#   ⚠️ default 必须用 if 赋值,不能写 ${VAR:-{...}} —— JSON 里的 } 会让 bash 提前闭合 ${},
#      当上游(start_swegym_4b.sh)已 export VLLM_HF_OVERRIDES 时会多吐一个字面 } → JSON 变 ...]}} 解析失败(踩过)。
if [ "${FEAT_GDN:-0}" = "1" ]; then
   [ -z "${VLLM_MODEL_NAME:-}" ] && VLLM_MODEL_NAME=qwen3_5forconditionalgeneration
   [ -z "${VLLM_HF_OVERRIDES:-}" ] && VLLM_HF_OVERRIDES='{"architectures":["Qwen3_5ForConditionalGeneration"]}'
   VLLM_ARGS+=(--qwen-gdn-backend npu --model-name "${VLLM_MODEL_NAME}" --vllm-hf-overrides "${VLLM_HF_OVERRIDES}")
fi
[ "${FEAT_PREFIX_CACHE:-0}" = "1" ] && VLLM_ARGS+=(--vllm-enable-prefix-caching --vllm-enable-chunked-prefill)
ADDCFG_PARTS=()
[ "${FEAT_STATIC_KERNEL:-0}" = "1" ] && ADDCFG_PARTS+=('"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":true}')
[ "${#ADDCFG_PARTS[@]}" -gt 0 ] && VLLM_ARGS+=(--vllm-additional-config "{$(IFS=,; echo "${ADDCFG_PARTS[*]}")}")
[ "${FEAT_HCCL_AIV:-0}" = "1" ] && export HCCL_OP_EXPANSION_MODE=AIV
[ "${FEAT_TRAIN_EXPANDABLE:-0}" = "1" ] && PERF_ARGS+=(--train-env-vars "{\"PYTORCH_NPU_ALLOC_CONF\":\"${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}\"}")
# wandb(自建 server;gated FEAT_WANDB 且 WANDB_KEY 非空)—— 与 slime 同款 pattern,
#   --wandb-host 默认指本机 :18080(wandb-local docker host 18080→容器 8080);group=RUN_ID。
WANDB_ARGS=()
if [ "${FEAT_WANDB:-0}" = "1" ] && [ -n "${WANDB_KEY:-}" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-host "${WANDB_HOST:-http://${MASTER_ADDR}:18080}"
      --wandb-key "${WANDB_KEY}"
      --wandb-project "${WANDB_PROJECT:-polar-swegym-8b}"
      --wandb-group "${RUN_ID}"
   )
fi
echo "[feat] async=${FEAT_ASYNC_SCHED:-0} prefix_cache=${FEAT_PREFIX_CACHE:-0} static_kernel=${FEAT_STATIC_KERNEL:-0} hccl_aiv=${FEAT_HCCL_AIV:-0} lb_proxy=${FEAT_LB_PROXY:-0} train_expandable=${FEAT_TRAIN_EXPANDABLE:-0} wandb=${FEAT_WANDB:-0}"

# ─── Ray(单机 NNODES=1 走 head 分支)+ 启动 ───
if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
   ray stop --force
   rm -rf "${RAY_TEMP_DIR}"
   ray start --head --port "${RAY_PORT}" --dashboard-host=0.0.0.0 --node-ip-address="${CURRENT_IP}" --dashboard-port="${RAY_DASHBOARD_PORT}" --num-gpus="${NPUS_PER_NODE}" --resources='{"NPU": '"${NPUS_PER_NODE}"'}' --temp-dir="${RAY_TEMP_DIR}" --disable-usage-stats

   while true; do
      active_node_count=$(ray status | awk '
         /^Active:/ {in_active=1; next}
         /^Pending:/ {in_active=0}
         in_active && $1 == "1" && $2 ~ /^node_/ {count++}
         END {print count + 0}')
      echo "[stage] wait Ray nodes active=${active_node_count}/${NNODES}"
      if [ "$active_node_count" -eq "$NNODES" ]; then
         unset ASCEND_RT_VISIBLE_DEVICES HCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME
         EXTRA_ARGS=()
         [ -n "${RESOURCE_LAYOUT:-}" ] && EXTRA_ARGS+=(--resource-layout "${RESOURCE_LAYOUT}")
         [ "${FEAT_LB_PROXY:-0}" = "1" ] && EXTRA_ARGS+=(--rollout-lb-proxy)
         python3 train_async.py \
            --train-backend megatron \
            ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
            ${TOPO_ARGS[@]} ${MODEL_ARGS[@]} ${ROLLOUT_ARGS[@]} ${POLAR_ARGS[@]} \
            ${OPTIMIZER_ARGS[@]} ${GRPO_ARGS[@]} ${PERF_ARGS[@]} ${VLLM_ARGS[@]} \
            ${MISC_ARGS[@]} ${CKPT_ARGS[@]} \
            ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"} \
            ${TRAIN_EXTRA_ARGS:-} \
            2>&1 | tee "${LOG_FILE}"
         break
      fi
      sleep 5
   done
else
   ray stop --force
   rm -rf "${RAY_TEMP_DIR}"
   while true; do
      ray start --address="${MASTER_ADDR}:${RAY_PORT}" --node-ip-address="${CURRENT_IP}" --num-gpus="${NPUS_PER_NODE}" --resources='{"NPU": '"${NPUS_PER_NODE}"'}' --temp-dir="${RAY_TEMP_DIR}" --disable-usage-stats
      ray status && break
      sleep 5
   done
fi
