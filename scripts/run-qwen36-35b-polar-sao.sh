#!/bin/bash
# vime + polar 算子 RL 启动(qwen3.6-35B-A3B / NPU)—— **SAO 变体**。
# 基线 = run-qwen36-35b-polar-ppo.sh(PPO)verbatim 迁移,叠加 SAO 论文可迁移的"路线 A(保守)"稳定器。
# 详见 docs/design/ppo_adaptation_findings.md §7/§9 与 ppo_adaptation_review_tables.md。
# ⚠️ fork 自 run-qwen36-35b-polar-ppo.sh @ 2026-07-16;PPO 脚本演进就要从它重新 fork(只差 header + RUN_ID + PPO_ARGS 块)。
#
# 路线 A(本脚本默认开,配置项/零代码,SAO_* env 可关):
#   - value 预训练 warmup:--num-critic-only-steps(治价值冷启动;SAO=10)          [C9]
#   - off-policy = DIS:--use-tis(clamp 近似;SAO_DIS_MASK=1 换 icepop 掩码=忠实版)  [C11]
#   - 非对称裁剪:--eps-clip 0.8 / --eps-clip-high 3.0(code 任务那套;数学用 0.3/5.0) [C11/Q-10]
#   - token 级 loss 归一:--calculate-per-token-loss(SAO_TOKEN_LEVEL_LOSS 可关)     [F-PPO-8]
#     ⚠️ 首跑先确认该 flag 形态无误(reset_arg 定义),再长跑。
#   - Faster Value Update K=2:--critic-update-steps 2(**已实现** train_async K 循环;默认开)[C12a]
#
# 已实现但默认关(env 一键开;占卡确认后启用):
#   - Frozen-Attention critic [C8]:冻结开关已 role 化(仅作用 critic,不误冻 actor);SAO_FROZEN_ATTN_CRITIC=1。
#     ⚠️ Q-1/Q-2:regex 命中需一次 dump(SAO_DUMP_CRITIC_PARAMS=/tmp/critic_params.txt)确认。
#   - critic 独立 LR [Q-12]:SAO_CRITIC_CONFIG=scripts/sao_critic_config.yaml(critic lr 5e-6,仅覆盖 critic)。
#   - Skip-Obs GAE + length-adaptive λ [C10/C12b,路线 B]:SAO_ROUTE_B=1(skip-obs + α=1.5);λ=1 时 skip-obs=no-op,须成对。
#     单测已验(tests/test_sao_gae.py):mask 全1=标准 GAE、skip-obs 手算、终止 reward 穿 masked 尾回流。
#
# ⚠️ 长 trace 必读:polar rollout trace 可达 22–40K tokens。adapter 以 max_tokens_per_gpu×cp_size 过滤,
#   默认 MAX_TOKENS_PER_GPU=512 → 512×4=2048 会把长 trace 全丢(No progress)。**必须设 MAX_TOKENS_PER_GPU=32768**
#   (=131072,同 start_ppo.sh)。见 vime_bridge/rollout.py:_resolve_max_tokens。
#
# 其余(env / 拓扑 / Ray / 推理引擎 / 特性叠加 / MISC / CKPT)全继承 PPO 脚本。
# PPO 关键点(详见 docs/design/ppo_adaptation_findings.md):
#   - --advantage-estimator ppo → use_critic=True;offload_train 被强制 True(actor+critic 共卡 CPU offload 时分)。
#   - critic backbone 从 --ref-load 自动载入,value head 重初始化;无需 --load。
#   - ⚠️ --kl-coef 必须 0(F-PPO-5);KL 走 loss 项。
#   - 拓扑:RESOURCE_LAYOUT=resource_layout_actor_domain2.yaml 钉 actor 同域避 EI0013(A1 已落 npu)。
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
cd "${VIME_ROOT}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# ─── 运行标识 / 多节点 ───
RUN_ID=${RUN_ID:-qwen36_polar_sao_$(date +%Y%m%d-%H%M%S)}
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
OPERATOR_DATA_ROOT=${OPERATOR_DATA_ROOT:-/home/docker/datasets/op_tasks/op_assets_cudallm_filtered189}
OPERATOR_TASK_JSONL=${OPERATOR_TASK_JSONL:-${OPERATOR_DATA_ROOT}/operator_tasks.jsonl}
OPERATOR_TASKS_DIR=${OPERATOR_TASKS_DIR:-${OPERATOR_DATA_ROOT}/op_tasks}
VLLM_ROUTER_PORT=${VLLM_ROUTER_PORT:-8001}    # profile.vime.yaml 推理端点指向它

# ─── 环境 ───
export PYTHONBUFFERED=16
export PYTHONPATH="/workspace/vllm:/workspace/vllm-ascend:/workspace/Megatron-LM:${VIME_ROOT}:${PYTHONPATH:-}"
# [复核-D 保留 2026-07-14] Ascend 自定义 MoE 训练算子库(moe_grouped_matmul/grouped_matmul_swiglu/swiglu)。
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
export VLLM_ASCEND_ENABLE_NZ=0                         # 必须 0:vllm-ascend wake_up 对 NZ+RL 硬 raise
export VLLM_TOOL_CALL_PARSER=qwen3_coder
export VLLM_REASONING_PARSER=qwen3
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=1
# HCCL(节点内 HCCS + 跨机 socket;长跑 EI0013 容错 + 35B 权重广播大 buffer)
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-2400}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-512}
export HCCL_INTRA_ROCE_ENABLE=${HCCL_INTRA_ROCE_ENABLE:-1}
export HCCL_INTRA_PCIE_ENABLE=${HCCL_INTRA_PCIE_ENABLE:-0}
# 跨机 HCCL 必需(对齐 slime;缺则双机权重同步 world>N 卡死在 rendezvous):
export HCCL_SOCKET_FAMILY=${HCCL_SOCKET_FAMILY:-AF_INET}
export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}
# Ascend 要求 ASCEND_RT_VISIBLE_DEVICES 升序(乱序 → torch_npu 见 0 卡)
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7,8,9,10,11,12,13,14,15}
export POLAR_KEEP_SESSION_DIR=${POLAR_KEEP_SESSION_DIR:-1}
export POLAR_TRAJECTORY_PG_STRICT=${POLAR_TRAJECTORY_PG_STRICT:-1}
export POLAR_ANTHROPIC_DEFAULT_MAX_TOKENS=${POLAR_ANTHROPIC_DEFAULT_MAX_TOKENS:-12288}

source "${VIME_ROOT}/scripts/models/qwen3.5-35B-A3B.sh"     # → MODEL_ARGS

# [双机修复 2026-07-14] CURRENT_IP 优先从 SOCKET_IFNAME 指定的网卡取(对齐 slime polar-minimal:56)。
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

# ─── 参数分组 ───
CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT:-/home/docker/Qwen3.6-35B-A3B}
   --ref-load ${REF_LOAD:-/home/docker/Qwen3.6-35B-A3B_fused_torch_dist}
   --save ${SAVE:-/workspace/Qwen3.6-35B-A3B_vime_polar}/
   --save-interval 10
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
   --rollout-temperature 0.7
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

PERF_ARGS=(
   --tensor-model-parallel-size "${TP:-2}"
   --pipeline-model-parallel-size "${PP:-1}"
   --context-parallel-size "${CP:-4}"
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

# ─── PPO 基础 + SAO 路线-A 稳定器 ───
PPO_ARGS=(
   --advantage-estimator ppo
   # ↑ 触发 use_critic=True;critic=同 qwen3.6 backbone + value head(hidden→1),与 actor 共卡 offload 时分。
   --num-critic-only-steps ${NUM_CRITIC_ONLY_STEPS:-10}
   # ↑ [SAO/C9] value 预训练 warmup:PPO 脚本默认 0;SAO=10 先热 critic 治冷启动。冒烟需 actor 也跑时设 0。
   --value-clip ${VALUE_CLIP:-0.2}
   --gamma ${GAMMA:-1.0}
   --lambd ${LAMBD:-1.0}
   # ↑ 路线 A 保持 λ=1;路线 B 才改 length-adaptive λ=1−1/(αl)(C12b,须与 skip-obs GAE 成对)。
   --kl-coef 0
   # ↑ ⚠️ 必须 0(F-PPO-5):critic 不算 ref_log_probs,kl-coef>0 → critic 取不到 ref 崩。
   --use-kl-loss
   --kl-loss-coef ${KL_LOSS_COEF:-0.001}
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --use-tis
   # ↑ [SAO/C11] DIS:vime TIS 乘法 =(π_θ/π_old)·(π_old/π_rollout)= π_θ/π_rollout,等价 SAO DIS。
   --eps-clip ${EPS_CLIP:-0.8}
   --eps-clip-high ${EPS_CLIP_HIGH:-3.0}
   # ↑ [SAO/C11/Q-10] 非对称裁剪:code 任务 0.8/3.0(数学任务 0.3/5.0);env 可覆盖。
   --critic-update-steps ${SAO_CRITIC_UPDATE_STEPS:-2}
   # ↑ [SAO/C12a] Faster Value Update K=2(已实现:train_async.py K 循环)。K=1 回退原逻辑。
   #   ⚠️ Q-4:K>1 使 critic 的 LR 调度器步进 K 倍快 → 盯 value LR/loss 曲线。
   # 注:--lr(OPTIMIZER_ARGS,继承 PPO=2e-6)actor/critic 共用;SAO critic 常需独立更大 LR(Q-12,走 megatron-config YAML)。
)
# [SAO/F-PPO-8] token 级 |M| 归一(over 全 batch 可训 token,非 per-seq);SAO_TOKEN_LEVEL_LOSS=0 关。
#   ✅ 已核实:Megatron 原生 store_true(裸 flag 正确);cp_utils 对 CP>1 已 chunk-aware(Q-16 设计上正确)。
[ "${SAO_TOKEN_LEVEL_LOSS:-1}" = "1" ] && PPO_ARGS+=(--calculate-per-token-loss)
# [SAO/C11 · Q-9] DIS 掩码:--use-tis 默认走 vanilla_tis(clamp,近似);SAO 忠实版用 icepop(mask,range 外置零)。
#   ✅ 已核实:icepop 签名与 vanilla 一致,可经 custom-tis-function-path 选(load_function 按模块路径 import)。
#   注:--tis-clip/--tis-clip-low(默认 2.0/0)界的是 off-policy 因子 π_old/π_rollout(与 --eps-clip 界的 π_θ/π_old 分两段)。
if [ "${SAO_DIS_MASK:-0}" = "1" ]; then
   PPO_ARGS+=(--custom-tis-function-path vime.backends.megatron_utils.loss.icepop_function)
fi
[ -n "${SAO_TIS_CLIP:-}" ] && PPO_ARGS+=(--tis-clip "${SAO_TIS_CLIP}")
[ -n "${SAO_TIS_CLIP_LOW:-}" ] && PPO_ARGS+=(--tis-clip-low "${SAO_TIS_CLIP_LOW}")
# [SAO/C8 已实现] Frozen-Attention critic:冻结开关已 role 化(--critic-only-train-params-name-list /
#   --critic-freeze-params-name-list 仅作用 critic,不误冻 actor)。默认 OFF;SAO_FROZEN_ATTN_CRITIC=1 启用。
#   ✅ Q-1/Q-2 已静态定 regex(读 qwen3_next.py):GDN=self_attention.linear_attn.*、full-attn=self_attention.*、
#     experts=mlp.experts.*、value head=output_layer.*。默认 "mlp.experts output_layer" = 只训 routed experts+value head,
#     冻全部 attention(GDN+full)/norm/embedding/router/shared_expert。宽版(含 shared_expert/router):
#     SAO_CRITIC_TRAIN_PATTERNS='mlp\. output_layer'。dump(SAO_DUMP_CRITIC_PARAMS)= 占卡最终确认。
if [ "${SAO_FROZEN_ATTN_CRITIC:-0}" = "1" ]; then
   PPO_ARGS+=(--critic-only-train-params-name-list ${SAO_CRITIC_TRAIN_PATTERNS:-mlp.experts output_layer})
fi
#   占卡定 regex:SAO_DUMP_CRITIC_PARAMS=/tmp/critic_params.txt 跑一次 → 看 critic 全部参数名(rank0 写文件,不中断训练)。
# [SAO/C12a] Faster Value Update K=2 已在 PPO_ARGS(--critic-update-steps ${SAO_CRITIC_UPDATE_STEPS:-2})。
# [SAO/C10+C12b 已实现 · 路线 B,默认关] skip-observation GAE + length-adaptive λ(必须成对;λ=1 时 skip-obs 是 no-op)。
#   SAO_ROUTE_B=1 一键开(skip-obs + α=1.5);或单独 SAO_SKIP_OBS_GAE=1 / SAO_GAE_LAMBDA_ALPHA=<α>。
#   ⚠️ 单测已验 mask 全1=标准 GAE、skip-obs 手算、终止 reward 穿 masked 尾回流(tests/test_sao_gae.py);占卡数值待验。
if [ "${SAO_ROUTE_B:-0}" = "1" ]; then
   PPO_ARGS+=(--skip-observation-gae --gae-lambda-alpha "${SAO_GAE_LAMBDA_ALPHA:-1.5}")
else
   [ "${SAO_SKIP_OBS_GAE:-0}" = "1" ] && PPO_ARGS+=(--skip-observation-gae)
   [ -n "${SAO_GAE_LAMBDA_ALPHA:-}" ] && PPO_ARGS+=(--gae-lambda-alpha "${SAO_GAE_LAMBDA_ALPHA}")
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 2e-6
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
   # [复核-B 回退 2026-07-14] slime 所有 NPU 脚本都用 --no-gradient-accumulation-fusion(NPU 无 CUDA fused kernel)。
   --no-gradient-accumulation-fusion
   --seed "${SEED:-1234}"
)

# ─── 特性开关(默认全 OFF = baseline 逐位不变)───
[ "${FEAT_ASYNC_SCHED:-0}" = "1" ] && VLLM_ARGS+=(--vllm-async-scheduling)
EP_ON=0
if [ "${FEAT_ROLLOUT_EP:-0}" = "1" ] || [ "${FEAT_FLASHCOMM1:-0}" = "1" ]; then
   VLLM_ARGS+=(--vllm-enable-expert-parallel); EP_ON=1     # FlashComm1 硬需 rollout EP
fi
[ "${FEAT_FLASHCOMM1:-0}" = "1" ] && export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
[ "${FEAT_PREFIX_CACHE:-0}" = "1" ] && VLLM_ARGS+=(--vllm-enable-prefix-caching --vllm-enable-chunked-prefill)
if [ "${FEAT_DP_EXTERNAL_LB:-0}" = "1" ]; then
   if [ "${FEAT_LB_PROXY:-0}" != "1" ]; then
      echo "[dp-extlb][FATAL] external-LB DP 需前置 LB 分发各 rank API server,请同开 FEAT_LB_PROXY=1" >&2; exit 1
   fi
   VLLM_ARGS+=(--vllm-data-parallel-external-lb)
fi
ADDCFG_PARTS=()
[ "${FEAT_MULTISTREAM_SHARED_EXPERT:-0}" = "1" ] && ADDCFG_PARTS+=('"multistream_overlap_shared_expert":true')
[ "${FEAT_STATIC_KERNEL:-0}" = "1" ] && ADDCFG_PARTS+=('"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":true}')
[ "${FEAT_BALANCE_SCHED:-0}" = "1" ] && ADDCFG_PARTS+=('"enable_balance_scheduling":true')
[ "${#ADDCFG_PARTS[@]}" -gt 0 ] && VLLM_ARGS+=(--vllm-additional-config "{$(IFS=,; echo "${ADDCFG_PARTS[*]}")}")
[ "${FEAT_HCCL_AIV:-0}" = "1" ] && export HCCL_OP_EXPANSION_MODE=AIV
[ "${REPRO_DETERMINISTIC:-0}" = "1" ] && VLLM_ARGS+=(--vllm-enable-deterministic-inference)
echo "[feat] async=${FEAT_ASYNC_SCHED:-0} flashcomm1=${FEAT_FLASHCOMM1:-0} ep=${EP_ON} prefix_cache=${FEAT_PREFIX_CACHE:-0} multistream=${FEAT_MULTISTREAM_SHARED_EXPERT:-0} static_kernel=${FEAT_STATIC_KERNEL:-0} hccl_aiv=${FEAT_HCCL_AIV:-0} lb_proxy=${FEAT_LB_PROXY:-0} dp_external_lb=${FEAT_DP_EXTERNAL_LB:-0} balance_sched=${FEAT_BALANCE_SCHED:-0}"
echo "[sao] num_critic_only_steps=${NUM_CRITIC_ONLY_STEPS:-10} eps_clip=${EPS_CLIP:-0.8}/${EPS_CLIP_HIGH:-3.0} token_level_loss=${SAO_TOKEN_LEVEL_LOSS:-1} lambd=${LAMBD:-1.0}(路线A) | 未启用:C8(Q-3)/K2/skip-obs"

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
         # [SAO/Q-12] critic 独立 LR/超参:SAO_CRITIC_CONFIG=scripts/sao_critic_config.yaml(critic lr 5e-6),仅覆盖 critic。
         [ -n "${SAO_CRITIC_CONFIG:-}" ] && EXTRA_ARGS+=(--megatron-config-path "${SAO_CRITIC_CONFIG}")
         python3 train_async.py \
            ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
            ${TOPO_ARGS[@]} ${MODEL_ARGS[@]} ${ROLLOUT_ARGS[@]} ${POLAR_ARGS[@]} \
            ${OPTIMIZER_ARGS[@]} ${PPO_ARGS[@]} ${PERF_ARGS[@]} ${VLLM_ARGS[@]} \
            ${MISC_ARGS[@]} ${CKPT_ARGS[@]} \
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
