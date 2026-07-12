#!/bin/bash
# vime + polar 算子 RL 启动脚本 —— 逐段对齐 slime 的 run-qwen36-35b-polar-minimal.sh。
# 与 slime 版的**全部**差异只有以下几处(其余分组/参数一字不差):
#   1. 推理引擎块  SGLANG_ARGS -> VLLM_ARGS   (vime 用 vllm/vllm-ascend)
#   2. bridge      slime_bridge -> vime_bridge
#   3. 拓扑        vime 无 --resource-layout,用 TOPO_ARGS(--actor-num-gpus/--rollout-num-gpus)
#   4. PYTHONPATH  sglang -> vllm/vllm-ascend
#   5. GRPO 里 **不设** --custom-pg-loss-reducer(Option A:vime 原生按-rollout 均权;
#      精确对齐 slime 的 per-trace 等权需 Option B 的 vime-core 改动,见
#      docs/design/vime_polar_integration.md §G1)
#
# polar 对本脚本的暴露 = 只有 --polar-url + 通用 --rollout-* 调度参数(与 slime 一致,无多余 YAML)。
# 前置:宿主机先用 profile.vime.yaml 起 polar(推理端点指向 vime 的 vllm:${VLLM_ROUTER_PORT})。
# 卡:polar 算子 agent 占 0-3(profile npu_lease),vime rollout+train 用 4-15。

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
cd "${VIME_ROOT}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

RUN_ID=${RUN_ID:-qwen36_polar_$(date +%Y%m%d-%H%M%S)}
SOCKET_IFNAME=${SOCKET_IFNAME:-}
MASTER_ADDR=${MASTER_ADDR:-80.48.5.88}
CURRENT_IP=${CURRENT_IP:-}
NNODES=${NNODES:-1}
NPUS_PER_NODE=${NPUS_PER_NODE:-12}          # vime 单机:rollout 4 + train 8 = 12 卡(4-15)
RAY_PORT=${RAY_PORT:-6460}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8290}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/ray_qwen36_vime_polar}

# 拓扑(vime 用 gpu 计数替代 slime 的 resource-layout)。TP2*CP4=8=actor;rollout 1 engine * 4 卡。
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-4}
# [PD 分离 #5] PD 要求 rollout 卡拆 ≥2 引擎(P/D 各 ≥1)。per-engine 默认:FEAT_PD_DISAGG=1 时压到 2
# (4 卡 rollout→1P1D 两引擎、各 TP2);单引擎默认 4。P/D 的 TP 必须相等(MooncakeHybridConnector 约束),
# 全引擎共用同一 per-engine 即满足。2P2D 见下方 PD 块注释(需扩 rollout 到 8 卡)。
if [ "${FEAT_PD_DISAGG:-0}" = "1" ]; then
   ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}
else
   ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}
fi

POLAR_OUTPUT_DIR=${POLAR_OUTPUT_DIR:-output/polar_bridge}
OPERATOR_DATA_ROOT=${OPERATOR_DATA_ROOT:-/home/docker/datasets/op_assets_cudallm_filtered189}
OPERATOR_TASK_JSONL=${OPERATOR_TASK_JSONL:-${OPERATOR_DATA_ROOT}/operator_tasks.jsonl}
OPERATOR_TASKS_DIR=${OPERATOR_TASKS_DIR:-${OPERATOR_DATA_ROOT}/op_tasks}
VLLM_ROUTER_PORT=${VLLM_ROUTER_PORT:-8001}  # vime serve OpenAI 的端口;profile.vime.yaml 指向它

export PYTHONBUFFERED=16
export PYTHONPATH="/workspace/vllm:/workspace/vllm-ascend:/workspace/Megatron-LM:${VIME_ROOT}:${PYTHONPATH:-}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TASK_QUEUE_ENABLE=0
# [CPU 绑核] 把 NPU 邻近 NUMA 核绑给 IRQ/worker/acl/release 线程,降延迟抖动、稳吞吐(torch_npu
# affinity.py 消费)。与训练数值正交、独立于 TASK_QUEUE;直接 bash 起不过 entrypoint.sh(其 :-1 默认
# 失效)故此处显式设。默认 on、可 CPU_AFFINITY_CONF=0 关;只加绑核不改数值,retro-compat。
export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}
# Inductor 在昇腾 get_gpu_type() 断言(assert len(avail_gpus)<=1)崩溃:训练侧 Megatron 会触发
# torch.compile;禁用 dynamo 走 eager(对齐已验证的 run_qwen36_35b_a3b_polar_npu.sh:136)。
export TORCHDYNAMO_DISABLE=1
export QWEN36_CP_MODE=ulysses
export QWEN36_CAUSAL_CONV1D_IMPL=triton
# [chunk LM-head] =1 时 patch GPTModel.forward 返回 hidden,loss 走 chunked logprob → logits
# 峰值 [chunk,V/tp] 脱离序列长 → 长 operator 序列不 OOM(修 loss.py get_log_probs OOM 根因)。
# 默认 0(retro-compat:与现状逐位一致)。验证:QWEN36_CHUNK_LMHEAD=1 MAX_TOKENS_PER_GPU=32768 拉起。
export QWEN36_CHUNK_LMHEAD=${QWEN36_CHUNK_LMHEAD:-0}
export VLLM_ASCEND_ENABLE_NZ=0
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=1
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}
export HCCL_INTRA_ROCE_ENABLE=${HCCL_INTRA_ROCE_ENABLE:-1}
# [EI0013 抗抖] slime BASELINE_SPEC 已验证的 HCCL 长跑容错(EI0013 ROCE CQE 是已知偶发问题):
# 更长 exec 超时容忍瞬态 + 512 大 buffer 利于 35B 权重广播 + 关 PCIe 内通(本拓扑 RoCE 为唯一节点内路径)。
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-2400}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-512}
export HCCL_INTRA_PCIE_ENABLE=${HCCL_INTRA_PCIE_ENABLE:-0}
# [拓扑] ⚠️ Ascend 要求 ASCEND_RT_VISIBLE_DEVICES 必须升序(乱序→torch_npu 见 0 卡)。故无法靠 env
# 乱序把 actor 钉到后八卡域 8-15;actor 只能=升序首 8=4-11(跨 7/8 域)。EI0013 跨域抖动靠上面
# HCCL 容错 env(EXEC_TIMEOUT/retry)兜(=slime 做法)。真要 actor 独占 8-15 需走 resource_layout 显式钉位(task-4 P1)。
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7,8,9,10,11,12,13,14,15}
export POLAR_KEEP_SESSION_DIR=${POLAR_KEEP_SESSION_DIR:-1}
export POLAR_TRAJECTORY_PG_STRICT=${POLAR_TRAJECTORY_PG_STRICT:-1}
export POLAR_ANTHROPIC_DEFAULT_MAX_TOKENS=${POLAR_ANTHROPIC_DEFAULT_MAX_TOKENS:-12288}

export VLLM_TOOL_CALL_PARSER=qwen3_coder
export VLLM_REASONING_PARSER=qwen3

source "${VIME_ROOT}/scripts/models/qwen3.5-35B-A3B.sh"     # -> MODEL_ARGS(vime_plugins spec)

CURRENT_IP=${CURRENT_IP:-$(hostname -I | awk '{print $1}')}
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},${CURRENT_IP}${no_proxy:+,${no_proxy}}"
export NO_PROXY="${no_proxy}"
if [ -n "${SOCKET_IFNAME}" ]; then
   export HCCL_SOCKET_IFNAME="${SOCKET_IFNAME}"
   export GLOO_SOCKET_IFNAME="${SOCKET_IFNAME}"
fi

POLAR_ROLLOUT_URL=${POLAR_ROLLOUT_URL:-http://${MASTER_ADDR}:8080}
LOG_FILE=${LOG_FILE:-${POLAR_OUTPUT_DIR}/train_${RUN_ID}.log}
LOG_FILE="/home/docker/logs/train_${RUN_ID}.log"
mkdir -p logs "${POLAR_OUTPUT_DIR}"

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
   --rollout-max-active-sessions "${POLAR_MAX_ACTIVE_SESSIONS:-8}"
   --rollout-release-on-postrun
   --rollout-min-complete-accept-fraction "${POLAR_MIN_COMPLETE_ACCEPT_FRACTION:-0.6}"
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TP:-2}"
   --pipeline-model-parallel-size "${PP:-1}"
   --context-parallel-size "${CP:-4}"       # 首次冒烟建议 export CP=1(vime 异步+CP>1 未验证);链路+G2-1 通了再切 4
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
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --use-tis
   # Option A:不设 --custom-pg-loss-reducer(vime 原生按-rollout 均权)。
   # 对齐 slime 的 slime_bridge.trajectory_loss(per-trace 等权)需 Option B,见 §G1。
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

# 推理引擎块(对应 slime 的 SGLANG_ARGS)—— 这是两份脚本唯一的实质引擎差异。
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

# ─── 特性叠加开关(§4.B,源码核实版;默认全 0 = baseline 逐位不变,retro-compat)───
# 叠加=已验特性只增不减(过闸即留,**不因 debug-bs 吞吐降而关**——debug bs 吞吐不准)。
# 闸=token-faith 正确性(TIS≈1 / logprob_abs_diff≲0.05 / session COMPLETED / 无崩 / step0+1)。
# 吞吐收益改到 realistic batch 另测,**不作叠加期门槛**。
# 全栈(待验全叠):EP + FlashComm1 + prefix-cache + multistream_overlap_shared_expert + static_kernel。
#   已验:EP(TIS1.0018)/ FlashComm1(1.0006)/ prefix-cache(step0+1)。新验:multistream / static_kernel。
# 不接线(审计):TOPK_OPTIMIZE=本分支死代码 no-op;CPU_AFFINITY 已容器默认=1;
#   TASK_QUEUE_ENABLE 保持 0 —— =1 让 GDN/ring-attn 训练出 NaN(进程级砸训练),严禁翻。
if [ "${FEAT_ASYNC_SCHED:-0}" = "1" ]; then
   VLLM_ARGS+=(--vllm-async-scheduling)
fi
# [rollout EP] FlashComm1 对 MoE 硬需 rollout 引擎开 EP(vllm-ascend platform.py:693 断言)→ FEAT_FLASHCOMM1
# 自动带上 EP。FEAT_ROLLOUT_EP=1 可单独开 EP(隔离验权重同步)。注:这是 vLLM ROLLOUT 的 EP(独立于
# actor 侧 --expert-model-parallel-size 8)。vime 既有支持(run-glm4.7-355B/minimax-m2 用过、update_weight
# 有 ep 逻辑),但 qwen3.6 首次用 → 须验权重同步 EP 分片正确(TIS≈1);错则查 update_weight/mbridge。
EP_ON=0
if [ "${FEAT_ROLLOUT_EP:-0}" = "1" ] || [ "${FEAT_FLASHCOMM1:-0}" = "1" ]; then
   VLLM_ARGS+=(--vllm-enable-expert-parallel); EP_ON=1
fi
if [ "${FEAT_FLASHCOMM1:-0}" = "1" ]; then
   export VLLM_ASCEND_ENABLE_FLASHCOMM1=1          # env,经 os.environ.copy() 传 vllm serve 子进程
fi
if [ "${FEAT_PREFIX_CACHE:-0}" = "1" ]; then
   # align 模式硬依赖 chunked-prefill(vllm config.py:384 否则 AssertionError);显式 pin 保前置。
   VLLM_ARGS+=(--vllm-enable-prefix-caching --vllm-enable-chunked-prefill)
fi
# [additional-config 合并] 多个特性各自贡献 additional-config 顶层键 → 必须合并成单个
# --vllm-additional-config JSON;否则重复 flag 后者覆盖前者、静默丢特性(叠加正确性硬要求)。
ADDCFG_PARTS=()
if [ "${FEAT_MULTISTREAM_SHARED_EXPERT:-0}" = "1" ]; then
   # shared-expert 计算放独立 stream 与 MoE dispatch/combine overlap(vllm-ascend fused_moe.py:386/748)。
   # 本模型有 shared expert(qwen3_next.py:157 shared_experts= 传入 FusedMoE)→ 激活非 no-op;
   # 开启时自带 _validate_shared_expert_consistency(验拆分==整体);无 mix_placement 冲突。
   ADDCFG_PARTS+=('"multistream_overlap_shared_expert":true')
fi
if [ "${FEAT_STATIC_KERNEL:-0}" = "1" ]; then
   ADDCFG_PARTS+=('"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":true}')
fi
if [ "${#ADDCFG_PARTS[@]}" -gt 0 ]; then
   ADDCFG_JSON="{$(IFS=,; echo "${ADDCFG_PARTS[*]}")}"
   VLLM_ARGS+=(--vllm-additional-config "$ADDCFG_JSON")
fi
if [ "${FEAT_HCCL_AIV:-0}" = "1" ]; then
   export HCCL_OP_EXPANSION_MODE=AIV               # batch5:CANN 层通信优化(HCCL 算子 offload→AIV 核),低风险 env
fi
if [ "${FEAT_KV_POOL:-0}" = "1" ]; then
   # KV 池化(对齐 ref.md §2.8 / ref2:AscendStoreConnector + mooncake;片上+DRAM 统一池、前缀跨节点可见)。
   # 🚫 前置:mooncake 库(现编 v0.3.9)+ mooncake_master 进程 + MOONCAKE_CONFIG_PATH 指向 mooncake.json
   #   (ref.md §3.2:global_segment_size 13GB/卡);未装 mooncake → 引擎 init ImportError。
   # ⚠️ GDN-hybrid 硬需 HMA:kv-transfer 默认强关 hybrid KV manager(vllm config/vllm.py:1342),
   #   但 GDN+full-attn 有两种 KV spec、必须 HMA(否则 "failed to convert KV cache specs" 崩)。
   #   AscendStoreConnector 是 SupportsHMA → --no-vllm-disable-hybrid-kv-cache-manager 保 HMA、共存。
   #   (flag 形式:vime 前缀成 --vllm-...,BooleanOptional 负向在 -- 后插 no- → --no-vllm-disable-...。)
   # 注:不用 OffloadingConnector——off-reference 弯路,且撞 vllm-ascend↔vllm 0.21.0 的 kv_offload 版本坑。
   # ⚠️ 键按我们这版:**lookup_rpc_port**(pool_scheduler.py:671 只认它/mooncake_rpc_port;ref 的 kvpool_rpc_port 无效)。
   #   还需(mooncake 建好后):mooncake_master 进程 + MOONCAKE_CONFIG_PATH(protocol:"ascend"/master_server_address/
   #   global_segment_size/local_buffer_size)+ LD_LIBRARY_PATH→mooncake。硬件 A2(910B2C):HCCL_INTRA_ROCE_ENABLE=1,
   #   不用 A3 的 ASCEND_ENABLE_USE_FABRIC_MEM/1GB 对齐。prefix-caching 保持开(两级命中)。
   export PYTHONHASHSEED=0                          # ref env:池化/mooncake 需确定性 hash
   # mooncake 现编 v0.3.9(USE_ASCEND_DIRECT):库分两处 → /usr/local/lib(libmooncake_store/transfer_engine/ascend_transport)
   #   + CANN site-packages/mooncake(store/engine.so);少 /usr/local/lib 则 import 报 libmooncake_store.so not found。
   #   (A2 片内 RoCE:HCCL_INTRA_ROCE_ENABLE 已在脚本头 L72 设=1,此处不重复。)
   export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/Ascend/cann-9.0.0/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"
   # mooncake.json 与本脚本同目录(P2PHANDSHAKE/protocol=ascend/master 127.0.0.1:30400/13GB per rank/SSD off)。
   #   ⚠️ 需先另起:mooncake_master --eviction_high_watermark_ratio 0.9 --eviction_ratio 0.15 --port 30400 --default_kv_lease_ttl 11000
   export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-${SCRIPT_DIR}/mooncake.json}"
   VLLM_ARGS+=(--no-vllm-disable-hybrid-kv-cache-manager)
   VLLM_ARGS+=(--vllm-kv-transfer-config '{"kv_connector":"AscendStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"backend":"mooncake","lookup_rpc_port":"0"}}')
fi
# [可复现] ⚠️ 经查证 --vllm-enable-deterministic-inference 不适配 polar:其"每样本 seed"只在 vime
#   原生 rollout(vllm_rollout.py:501/743)注入,polar 多轮 agent 自建请求(vime_bridge 只发任务
#   payload)→ seed 够不到 polar;仅 VLLM_BATCH_INVARIANT=1(engine 级)生效。且 polar 轨迹含环境非
#   确定性(agent 编译/跑/评测算子)→ 轨迹级复现不可得。故**默认 OFF**(batch-invariant 拖慢推理、
#   对 polar 只换前向 bit 确定,不抵成本)。polar 可复现靠"固定 seed(下 --seed + rollout-seed 42)+
#   聚合指标"。REPRO_DETERMINISTIC=1 仅在跑 vime 原生 rollout 时才值得开。
if [ "${REPRO_DETERMINISTIC:-0}" = "1" ]; then
   VLLM_ARGS+=(--vllm-enable-deterministic-inference)
fi
# [PD 分离 #5] mooncake MooncakeHybridConnector(GDN-hybrid/Mamba 必需——Nixl 无 hybrid SSM-FA,RFC #36780
#   closed not-planned)。FEAT_PD_DISAGG=1:prefill_num_servers×per_engine 卡做 prefill、余卡 decode
#   (from_prefill_num_servers,vllm_config.py:183)→ has_pd_disaggregation → rollout manager 起我们的
#   proxy(rollout.py:_start_mooncake_pd_proxy,W4)替 vllm-router,避 #3 return_token_ids 丢弃。
#   链路:vime→proxy→P prefill(max_tokens=1)返 kv_transfer_params→注入 D→D 拉 P 的 KV+decode→透传。
#   引擎侧注入 --kv-transfer-config MooncakeHybridConnector(P=producer/D=consumer,vllm_engine.py W2)。
#   最小验证 1P1D=rollout 4 卡(per-engine=2,上方已自动);2P2D 需 ROLLOUT_NUM_GPUS=8 + 缩 actor 到 8 卡 +
#   PD_PREFILL_NUM_SERVERS=2。验证阶梯:差分贪婪(temp=0,PD vs 单引擎逐字,验 Mamba-state P→D)→ token-faith TIS。
#   默认 OFF=零回归(不传 --prefill-num-servers → 单引擎路径原样)。见 docs/design/pd_disaggregation_dev_plan.md。
PD_ARGS=()
if [ "${FEAT_PD_DISAGG:-0}" = "1" ]; then
   PD_ARGS+=(--prefill-num-servers "${PD_PREFILL_NUM_SERVERS:-1}")
   PD_ARGS+=(--disaggregation-backend "${PD_BACKEND:-mooncake}")
   if [ "${PD_BACKEND:-mooncake}" = "mooncake" ]; then
      # mooncake PD(P2P KV 直传)只需 mooncake 库在 LD_LIBRARY_PATH(vllm-ascend 参考指南
      #   pd_disaggregation_mooncake_single_node.md §138/141)——**不需** mooncake_master、也**不需**
      #   MOONCAKE_CONFIG_PATH(那是 KV 池化 AscendStore 的中心池才要;PD 连接器只设 ASCEND_TRANSFER_TIMEOUT)。
      #   库分两处:/usr/local/lib(+lib64)与 CANN site-packages/mooncake;在 ray start 前 export → raylet
      #   继承 → 引擎子进程可 import mooncake(缺则 ModuleNotFoundError,PD 引擎秒崩)。
      export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64:/usr/local/Ascend/cann-9.0.0/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"
      # [HMA 硬前置,与 FEAT_KV_POOL 同款] 任何 --kv-transfer-config(W2 给 P/D 注入 MooncakeHybridConnector)
      #   都触发 vllm 默认关 hybrid KV manager(config/vllm.py:1342);GDN+full-attn 有两种 KV spec、必须 HMA,
      #   否则引擎 init 崩 "failed to convert the KV cache specs to one unified type"。MooncakeHybridConnector
      #   是 SupportsHMA(mooncake_hybrid_connector.py:969)→ 加 flag 保 HMA、连接器共存(设计文档 §5/§8:PD 复用)。
      if [[ ! " ${VLLM_ARGS[*]} " == *" --no-vllm-disable-hybrid-kv-cache-manager "* ]]; then
         VLLM_ARGS+=(--no-vllm-disable-hybrid-kv-cache-manager)
      fi
   fi
fi
echo "[feature-stacking] async=${FEAT_ASYNC_SCHED:-0} flashcomm1=${FEAT_FLASHCOMM1:-0} rollout_ep=${EP_ON} prefix_cache=${FEAT_PREFIX_CACHE:-0} multistream=${FEAT_MULTISTREAM_SHARED_EXPERT:-0} static_kernel=${FEAT_STATIC_KERNEL:-0} hccl_aiv=${FEAT_HCCL_AIV:-0} kv_pool=${FEAT_KV_POOL:-0} pd_disagg=${FEAT_PD_DISAGG:-0}(P=${PD_PREFILL_NUM_SERVERS:-1},be=${PD_BACKEND:-mooncake}) | addcfg=${ADDCFG_JSON:-none} | deterministic=${REPRO_DETERMINISTIC:-0} seed=${SEED:-1234} | TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE} (kept)"

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-flash-attn
   --moe-token-dispatcher-type alltoall
   --no-gradient-accumulation-fusion
   --seed "${SEED:-1234}"
)

# Ray(对齐 slime;单机 NNODES=1 走 head 分支。注意 ray stop --force 会停本机 ray)
if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
   ray stop --force
   rm -rf "${RAY_TEMP_DIR}"
   ray start --head --port "${RAY_PORT}" --dashboard-host=0.0.0.0 --node-ip-address="${CURRENT_IP}" --dashboard-port="${RAY_DASHBOARD_PORT}" --num-gpus="${NPUS_PER_NODE}" --resources='{"NPU": '"${NPUS_PER_NODE}"'}' --temp-dir="${RAY_TEMP_DIR}" --disable-usage-stats

   while true; do
      ray_status_output=$(ray status)
      active_node_count=$(echo "$ray_status_output" | awk '
         /^Active:/ {in_active=1; next}
         /^Pending:/ {in_active=0}
         in_active && $1 == "1" && $2 ~ /^node_/ {count++}
         END {print count + 0}
      ')
      echo "[stage] wait Ray nodes active=${active_node_count}/${NNODES}"
      if [ "$active_node_count" -eq "$NNODES" ]; then
         ray status
         unset ASCEND_RT_VISIBLE_DEVICES HCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME
         # [拓扑] RESOURCE_LAYOUT 设了则显式钉位(actor→8-15/rollout→4-7,免跨域 EI0013)
         EXTRA_ARGS=()
         [ -n "${RESOURCE_LAYOUT:-}" ] && EXTRA_ARGS+=(--resource-layout "${RESOURCE_LAYOUT}")
         python3 train_async.py \
            ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
            ${TOPO_ARGS[@]} \
            ${PD_ARGS[@]+"${PD_ARGS[@]}"} \
            ${MODEL_ARGS[@]} \
            ${ROLLOUT_ARGS[@]} \
            ${POLAR_ARGS[@]} \
            ${OPTIMIZER_ARGS[@]} \
            ${GRPO_ARGS[@]} \
            ${PERF_ARGS[@]} \
            ${VLLM_ARGS[@]} \
            ${MISC_ARGS[@]} \
            ${CKPT_ARGS[@]} \
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
