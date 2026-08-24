#!/bin/bash
# vime + polar 算子 RL 启动(qwen3.6-35B-A3B / NPU)—— **双机:.56 训练 8 卡 + .57 rollout 16 卡 PD**。
#
# 卡位(单一真源 = ${RESOURCE_LAYOUT},见 resource_layout.dual56train57infer_pd.yaml):
#   80.48.5.56  0-7   actor 训练(同 HCCS 域,免 EI0013)
#   80.48.5.56  8-15  polar agent/judge(宿主机子容器,**不进 ray**)
#   80.48.5.57  0-15  rollout PD:prefill 4 卡(tp4×1)+ decode 12 卡(tp4×3)
#
# 前置:宿主机(.56)先起 polar,且 profile 里
#   sglang_router_url: http://80.48.5.57:8001    ← proxy 起在 **rollout 节点**,不是 head。
#     原因:layout 路径下 create_rollout_manager(placement_group.py:317)把 RolloutManager 钉在
#     rollout 首个 bundle → 它进程内起的 PD proxy 也就落在 .57。
#   npu_pool: "8,9,10,11,12,13,14,15"            ← 避开 actor 的 0-7
#
# 启动(两台跑同一个脚本,靠 CURRENT_IP==MASTER_ADDR 分角色;NPUS_PER_NODE/可见卡按角色自动填):
#   head@56:   SOCKET_IFNAME=ens1f3 bash scripts/run-qwen36-35b-polar-minimal.sh
#   worker@57: CURRENT_IP=80.48.5.57 SOCKET_IFNAME=<57网卡> bash scripts/run-qwen36-35b-polar-minimal.sh
#
# 单机回退:NNODES=1 RESOURCE_LAYOUT=scripts/resource_layout.single52.yaml 并显式给
#   NPUS_PER_NODE / ASCEND_RT_VISIBLE_DEVICES / FEAT_PD_DISAGG=0。
set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
cd "${VIME_ROOT}"

# ─── CANN 9.2.0(对齐 PD 参考脚本 run-qwen36-35b-polar-minimal-single-rollout-only-pd.sh)───
# PD/mooncake 只在这套环境验证过(CANN 9.2.0 + vllm-023 + vllm-ascend-023 + 新编 mooncake)。
ASCEND_ROOT=${ASCEND_ROOT:-/usr/local/Ascend}
CANN_ROOT=${CANN_ROOT:-${ASCEND_ROOT}/cann}
CANN_TOOLKIT_ROOT="${ASCEND_ROOT}/ascend-toolkit/cann-9.0.0"
CANN_BIN_DIR="${CANN_ROOT}/bin"
CANN_LIB_DIR="${CANN_ROOT}/lib64"
CANN_PYTHON_SITE_PACKAGES="${CANN_ROOT}/python/site-packages"
CANN_TBE_DIR="${CANN_ROOT}/opp/built-in/op_impl/ai_core/tbe"
source "${CANN_ROOT}/set_env.sh"
source /usr/local/Ascend/nnal/atb/set_env.sh
# 必须在 set_env.sh **之后** 覆盖(ops_legacy 被拷到 9.2.0 树下)
export ASCEND_OPP_PATH="${CANN_ROOT}/opp"
export ASCEND_HOME_PATH="${CANN_ROOT}"
export ASCEND_TOOLKIT_HOME="${CANN_ROOT}"

# [GDN 训练算子 2026-08-05] Megatron 侧 GDN 前向要 aclnnRecomputeWUFwd
# (MindSpeed/mindspeed/ops/chunk_gated_delta_rule.py:145 → torch.ops.npu.npu_recompute_w_u_fwd,
#  op 由 site-packages/fla_npu 的 C++ 扩展注册)。该符号**只**存在于 CANN **9.0.0** 的 vendor 包:
#    cann-9.0.0/opp/vendors/custom_transformer/op_api/lib/libcust_opapi.so
#  而 cann-9.0.0/opp/vendors/ 是空的(Aug 3 切 CANN 时没带过来),ASCEND_OPP_PATH 指 9.2.0 →
#  训练一跑 GDN 就 "aclnnRecomputeWUFwd ... not in libopapi.so"。rollout-only 碰不到,故此前没暴露。
# torch_npu 从 ${ASCEND_CUSTOM_OPP_PATH}/op_api/lib/libcust_opapi.so 加载(libtorch_npu.so 内字符串
#  ASCEND_CUSTOM_OPP_PATH + /op_api/lib/ + libcust_opapi.so),所以这里指到 **vendor 目录本身**。
# 实测:设了它符号即可解析并进入 kernel(9.0.0 kernel × 9.2.0 runtime 未见加载期不兼容)。
ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-/usr/local/Ascend/cann-9.0.0/opp/vendors/fla_npu_transformer}
if [ -f "${ASCEND_CUSTOM_OPP_PATH}/op_api/lib/libcust_opapi.so" ]; then
   export ASCEND_CUSTOM_OPP_PATH
   echo "[env] ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH} (GDN aclnnRecomputeWUFwd)"
else
   echo "[FATAL] GDN 自定义算子包缺失: ${ASCEND_CUSTOM_OPP_PATH}/op_api/lib/libcust_opapi.so" >&2
   echo "        训练侧 GDN 前向会崩。确认 CANN 9.0.0 的 custom_transformer vendor 包在位。" >&2
   exit 1
fi
for d in "${CANN_BIN_DIR}" "${CANN_LIB_DIR}" "${CANN_PYTHON_SITE_PACKAGES}"; do
   [ -e "${d}" ] || { echo "[FATAL] Required CANN path missing: ${d}" >&2; exit 1; }
done

# ─── 运行标识 / 多节点 ───
RUN_ID=${RUN_ID:-qwen36_polar_$(date +%Y%m%d-%H%M%S)}
MASTER_ADDR=${MASTER_ADDR:-80.48.5.59}
ROLLOUT_NODE_IP=${ROLLOUT_NODE_IP:-80.48.5.56}   # 引擎+proxy 所在节点;须与 layout 的 rollout node 一致
CURRENT_IP=${CURRENT_IP:-}
SOCKET_IFNAME=${SOCKET_IFNAME:-data0.172}
NNODES=${NNODES:-2}
RAY_PORT=${RAY_PORT:-6461}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8291}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/ray_qwen36_vime_polar}

# ─── 拓扑:卡位由 RESOURCE_LAYOUT 唯一决定 ───
# layout loader(arguments.py:1802-1812)会用 layout 覆盖 actor_num_nodes/actor_num_gpus_per_node/
# rollout_num_gpus/rollout_num_gpus_per_engine,并把 num_gpus_per_node 设成 **rollout 节点** 的
# 每节点卡数(16)—— 端口分配器靠它反推 node_index,所以这里不必也不该再传 --num-gpus-per-node。
RESOURCE_LAYOUT=${RESOURCE_LAYOUT:-${VIME_ROOT}/scripts/resource_layout.dual56train57infer_pd.yaml}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-16}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}

# ─── colocate(训推同卡、同步 train.py)───
# FEAT_COLOCATE=1 时:
#   * --resource-layout 与 --colocate 互斥(arguments.py:1817 直接 raise),故必须清空 layout;
#   * arguments.py:1888-1908 会强制 offload_train/offload_rollout=True,并把
#     num_gpus_per_node/rollout_num_gpus 覆盖成 actor_num_gpus_per_node(*actor_num_nodes),
#     所以 ROLLOUT_NUM_GPUS 必须自己就等于 ACTOR_NUM_GPUS_PER_NODE*ACTOR_NUM_NODES,否则只是被静默改写;
#   * 入口切 train.py —— train_async.py:11 有 assert not args.colocate。
FEAT_COLOCATE=${FEAT_COLOCATE:-0}
# train_async.py:11 assert not args.colocate → colocate 只能走同步入口;
# TRAIN_ENTRY 可被环境变量覆盖(如混合同步部署用 TRAIN_ENTRY=train.py 直切,不经 FEAT_COLOCATE)。
TRAIN_ENTRY=${TRAIN_ENTRY:-$([ "${FEAT_COLOCATE}" = "1" ] && echo train.py || echo train_async.py)}
if [ "${FEAT_COLOCATE}" = "1" ]; then
   RESOURCE_LAYOUT=""
   ROLLOUT_NODE_IP="${MASTER_ADDR}"        # 引擎与 actor 同节点,metrics/proxy 发现目标随之回到本机
fi

# ─── polar 数据 / 端点 ───
POLAR_OUTPUT_DIR=${POLAR_OUTPUT_DIR:-output/polar_bridge}
OPERATOR_DATA_ROOT=${OPERATOR_DATA_ROOT:-/home/docker/datasets/op_assets_cudallm_filtered189}
OPERATOR_TASK_JSONL=${OPERATOR_TASK_JSONL:-${OPERATOR_DATA_ROOT}/operator_tasks.jsonl}
OPERATOR_TASKS_DIR=${OPERATOR_TASKS_DIR:-${OPERATOR_DATA_ROOT}/op_tasks}
VLLM_ROUTER_PORT=${VLLM_ROUTER_PORT:-8001}    # polar profile 的推理端点指向它
# PD proxy bind 在 RolloutManager 所在节点 = rollout 节点(见文件头说明),不是 head。
VLLM_ROUTER_IP=${VLLM_ROUTER_IP:-${ROLLOUT_NODE_IP}}
# rollout 侧 PD 拓扑:复用 PD 参考脚本那份 yaml(prefill 4 + decode 12,per_engine=4,共 16 卡)
VLLM_PD_CONFIG=${VLLM_PD_CONFIG:-}
FEAT_PD_DISAGG=${FEAT_PD_DISAGG:-1}

# ─── 环境 ───
export PYTHONBUFFERED=16
# 前置 site-packages = 新编 mooncake;vllm-023/vllm-ascend-023 = PD 验证过的那套(对齐 PD 参考脚本)
# [2026-08-14] 这里**不能**再列 /workspace/vllm 与 /workspace/vllm-ascend:
#   023 那套是 pip install -e 装的,靠 site-packages 的 __editable__ finder 提供
#   (vllm→/workspace/vllm-023,vllm_ascend→/workspace/vllm-ascend-023)。而该 finder 是
#   sys.meta_path.append() 注册的(finder:76)→ 排在内置 PathFinder **之后**,于是
#   PYTHONPATH 里显式写的非 023 目录会把它整个遮蔽掉。
#   rollout 节点(.64)上这两个非 023 目录恰好存在 → 引擎加载 vllm-ascend 的
#   mooncake_connector(2632 行,缺 _handle_peer_requests / set_xfer_handshake_metadata_*),
#   PD 交接握不上手:prefill 返回合法 kv_transfer_params 后 decode 永不拉 KV,
#   proxy 的 preflight 永久挂死(timeout=None)→ 8011 不 ready → rollout 全量 drop。
#   旧 rollout 机上没有这两个目录,遮蔽不成立,所以同样的脚本在那边是好的。
export PYTHONPATH="/usr/local/lib/python3.11/site-packages:/workspace/Megatron-LM:${VIME_ROOT}:${CANN_PYTHON_SITE_PACKAGES}:${CANN_TBE_DIR}:${CANN_TOOLKIT_ROOT}/python/site-packages:${PYTHONPATH:-}"
# [2026-08-14] 必须 0.23.0,且必须与上面的 PYTHONPATH 配套改:
#   /workspace/vllm-023=0.23.0(有 expert_map_manager) /workspace/vllm=0.21.0(没有);
#   两棵 vllm-ascend 都要 0.23 的 API。vllm_ascend.utils.vllm_version_is(utils.py:610)
#   **优先读这个 env**、而不是 vllm.__version__,于是它直接决定 patch 走哪个分支:
#     patch_dp_device_ids.py:33  if not vllm_version_is("0.23.0"): <取 0.23 才有的符号>
#   写 0.21.0 → 门控取反 → AttributeError: get_physical_gpu_ids_for_local_dp_rank。
#   (0.21.0 是配 /workspace/vllm 那套错误组合时打的补丁,PYTHONPATH 修好后必须跟着回来。)
export VLLM_VERSION=0.23.0  # 与 /workspace/vllm-023 的真实版本一致
export PATH="${CANN_BIN_DIR}:${PATH:-}"
export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64:${CANN_LIB_DIR}:${CANN_TOOLKIT_ROOT}/x86_64-linux/lib64:${CANN_TOOLKIT_ROOT}/x86_64-linux/devlib:${CANN_TOOLKIT_ROOT}/opp/lib64:${CANN_TOOLKIT_ROOT}/opp/lib64/plugin/opskernel:${LD_LIBRARY_PATH:-}"
# Ascend 自定义 MoE 训练算子(--moe-grouped-gemm 用)。本容器两处都不存在 → ld 直接跳过、回退原生实现,
# 保留是为了镜像里存在时行为不变。
export LD_LIBRARY_PATH="${CANN_ROOT}/opp/vendors/custom_transformer/op_api/lib:/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}"
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
export HCCL_SOCKET_FAMILY=${HCCL_SOCKET_FAMILY:-AF_INET}       # 强制 IPv4(网卡带 IPv6 地址会 socket family mismatch)
export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}     # 禁 IP 白名单(否则跨机对端 IP 不在白名单→连接被拒→卡死)
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export HCCL_INTER_HCCS_DISABLE=${HCCL_INTER_HCCS_DISABLE:-true}
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
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},${CURRENT_IP},${ROLLOUT_NODE_IP}${no_proxy:+,${no_proxy}}"
export NO_PROXY="${no_proxy}"
if [ -z "${SOCKET_IFNAME}" ]; then
   SOCKET_IFNAME=$(ip -o addr show 2>/dev/null | awk -v t="${CURRENT_IP}" '{split($4,p,"/"); if (p[1]==t){print $2; exit}}')
fi
if [ -n "${SOCKET_IFNAME}" ]; then
   export HCCL_SOCKET_IFNAME="${SOCKET_IFNAME}"
   export GLOO_SOCKET_IFNAME="${SOCKET_IFNAME}"
   export TP_SOCKET_IFNAME="${SOCKET_IFNAME}"
fi
export HCCL_IF_IP="${CURRENT_IP}"
# mooncake 连接器的 side_channel_host 走 vllm get_ip()(kv_p2p/mooncake_connector.py:1890);
# 多网卡下解析到非业务网段 → P 返回给 D 的 remote_host 不可达 → D 侧拉 KV 静默挂住。两台都钉。
export VLLM_HOST_IP=${VLLM_HOST_IP:-${CURRENT_IP}}
export VIME_HOST_IP=${VIME_HOST_IP:-${CURRENT_IP}}

# ─── 按角色定 ray 注册卡数与可见卡(与 layout 的 devices 必须对得上)───
# Ascend 要求 ASCEND_RT_VISIBLE_DEVICES 升序(乱序 → torch_npu 见 0 卡)。
if [ "${MASTER_ADDR}" = "${CURRENT_IP}" ]; then
   NODE_ROLE=head                                  # 训练节点:只暴露 0-7,8-15 留给宿主机 polar
   NPUS_PER_NODE=${NPUS_PER_NODE:-8}
   export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
else
   NODE_ROLE=worker                                # rollout 节点:16 卡全给引擎
   NPUS_PER_NODE=${NPUS_PER_NODE:-16}
   export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11}
fi
echo "[topo] role=${NODE_ROLE} ip=${CURRENT_IP} if=${SOCKET_IFNAME} npus=${NPUS_PER_NODE} devices=${ASCEND_RT_VISIBLE_DEVICES}"
echo "[topo] actor=8卡@${MASTER_ADDR}  rollout=${ROLLOUT_NUM_GPUS}卡@${ROLLOUT_NODE_IP}(PD 1P3D tp4)  proxy=${VLLM_ROUTER_IP}:${VLLM_ROUTER_PORT}"

POLAR_ROLLOUT_URL=${POLAR_ROLLOUT_URL:-http://${MASTER_ADDR}:8080}
LOG_FILE=${LOG_FILE:-/mnt/pipeline-data/train_log/train_${RUN_ID}.log}
mkdir -p logs "${POLAR_OUTPUT_DIR}" /home/docker/logs

# ─── 参数分组 ───
CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT:-/home/docker/Qwen3.6-35B-A3B}
   --ref-load ${REF_LOAD:-/home/docker/Qwen3.6-35B-A3B_fused_torch_dist}
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
   --num-rollout "${NUM_ROLLOUT:-20}"
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

# 权重更新时是否等 polar 的 in-flight session 排空。
#   分离部署(异步):等 —— 生成与训练重叠,等一下就能把整组收完,不浪费。
#   colocate(同步):不等 —— 引擎整个训练步都 sleep,等 session 纯粹是让训练干等;
#     在跑的 session 直接放弃,交给 version-span guard 在 resume 时拒绝跨界续跑。
#   POLAR_DRAIN_SESSIONS 可显式覆盖(1=等 / 0=不等)。
if [ "${POLAR_DRAIN_SESSIONS:-$([ "${FEAT_COLOCATE:-0}" = "1" ] && echo 0 || echo 1)}" = "1" ]; then
   DRAIN_ARGS=(--polar-weight-update-drain-sessions)
else
   DRAIN_ARGS=(--no-polar-weight-update-drain-sessions)
fi

# 跨权重更新的组要不要。默认(不传)沿用 max_async_level+update_weights_interval 的推导值,
# 下限恒为 2 → 跨一次更新的 staleness=1 永远被接受,也就是混权轨迹会进训练集。
# colocate 下 polar 侧没有 /admin/policy_version(version-span guard 会 404 降级),
# 只能在这里丢:0 = 只收当轮生成的组,上一轮遗留的一律丢弃。
POLAR_MAX_OFF_POLICY_STEPS=${POLAR_MAX_OFF_POLICY_STEPS:-$([ "${FEAT_COLOCATE:-0}" = "1" ] && echo 0 || echo "")}
if [ -n "${POLAR_MAX_OFF_POLICY_STEPS}" ]; then
   STALENESS_ARGS=(--rollout-max-off-policy-steps "${POLAR_MAX_OFF_POLICY_STEPS}")
else
   STALENESS_ARGS=()
fi

POLAR_ARGS=(
   --polar-url "${POLAR_ROLLOUT_URL}"
   --polar-run-id "${RUN_ID}"
   --polar-reward-key score
   --polar-task-id-template "{args.polar_run_id}-polar-op-{rollout_id}-{sample.group_index}"
   --operator-tasks-dir "${OPERATOR_TASKS_DIR}"
   --rollout-max-async-level "${POLAR_MAX_ASYNC_LEVEL:-1}"
   --rollout-request-timeout "${POLAR_ROLLOUT_REQUEST_TIMEOUT:-9000}"
   --rollout-scheduler-mode session_pool
   --rollout-max-active-sessions "${POLAR_MAX_ACTIVE_SESSIONS:-16}"
   --rollout-release-on-postrun
   --rollout-min-complete-accept-fraction "${POLAR_MIN_COMPLETE_ACCEPT_FRACTION:-0.6}"
   ${DRAIN_ARGS[@]+"${DRAIN_ARGS[@]}"}
   ${STALENESS_ARGS[@]+"${STALENESS_ARGS[@]}"}
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
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-32768}"
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
   # 共卡混合:同步期 trainer 与引擎同卡,512MB bucket 的 all_gather+IPC 瞬时块会
   # 顶穿卡余量(20260824-142800 实锤 rank13 free 0.03G OOM)。默认 256MB,可 env 调。
   --update-weight-buffer-size "${UPDATE_WEIGHT_BUFFER_SIZE:-268435456}"
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

# actor 与 rollout 分居不同节点、卡不重叠 → 无需 offload 腾显存。
# colocate 下两者共卡,必须 offload 才腾得出显存(不传的话 arguments.py:1888 也会强制置 True,
# 这里显式留空是为了避免 --no-offload-* 与那段强制逻辑冲突)。
# FEAT_OFFLOAD=1(混合部署):不经 colocate 开关,显式打开训/推双侧 offload。
if [ "${FEAT_OFFLOAD:-0}" = "1" ]; then
   OFFLOAD_ARGS=(--offload-train --offload-rollout)
elif [ "${FEAT_COLOCATE:-0}" = "1" ]; then
   OFFLOAD_ARGS=()
else
   OFFLOAD_ARGS=(--no-offload-train --no-offload-rollout)
fi

VLLM_ARGS=(
   --rollout-backend vllm
   --qwen-gdn-backend npu
   --model-name qwen3_5moeforconditionalgeneration
   --vllm-hf-overrides '{"architectures":["Qwen3_5MoeForConditionalGeneration"]}'
   --vllm-router-ip "${VLLM_ROUTER_IP}"
   --vllm-router-port "${VLLM_ROUTER_PORT}"
   --vllm-weight-sync-mode native
   --no-vllm-weight-sync-packed
   --vllm-gpu-memory-utilization "${VLLM_GPU_MEM_UTIL:-0.85}"
   --vllm-max-num-seqs "${VLLM_MAX_NUM_SEQS:-96}"
   --vllm-max-model-len "${VLLM_MAX_MODEL_LEN:-131072}"
   --vllm-enable-sleep-mode
   --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
   --vllm-max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
   ${OFFLOAD_ARGS[@]+"${OFFLOAD_ARGS[@]}"}
   # renderer 多 worker(前端渲染/tokenize 并行,长 prompt 提速)。注意必须与 mm-processor-cache-gb 0 同开:
   # vllm 校验"renderer_num_workers>1 与多模态缓存互斥"(缓存非线程安全),纯文本任务关它零损失。
   --vllm-renderer-num-workers 4
   --vllm-mm-processor-cache-gb 0
)

# ─── rollout 侧 PD 分离(拓扑对齐 run-qwen36-35b-polar-minimal-single-rollout-only-pd.sh)───
# 卡怎么切由 ${VLLM_PD_CONFIG} 的 server_groups 决定(prefill 4 / decode 12,per_engine=4),
# 其总卡数必须 == layout 里 rollout 的卡数(16),否则 rollout_validation.py 拦。
if [ "${FEAT_PD_DISAGG:-1}" = "1" ]; then
   VLLM_ARGS+=(
      --vllm-config "${VLLM_PD_CONFIG}"
      --disaggregation-backend "${PD_BACKEND:-mooncake}"
      # --vllm-speculative-config "${VLLM_SPEC_CONFIG:-{\"method\":\"mtp\",\"num_speculative_tokens\":1}}"
   )
   FEAT_ROLLOUT_EP=${FEAT_ROLLOUT_EP:-1}       # PD 参考脚本开着 EP(经下面的 EP 闸门统一加 flag)
   FEAT_PREFIX_CACHE=${FEAT_PREFIX_CACHE:-1}   # 同参考脚本;yaml 里 P/D 两组也各自置 true
fi

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
# ─── RL profiling(默认全 OFF = baseline 逐位不变;2026-08-12 重做,照 verl mstx_profile 实证模式)───
# PROFILE_TRAIN=1:训练侧 NPU 采集。命中目标步才建 profiler(start→跑→step→stop),
#   录制窗精确等于目标步/阶段本身,无 schedule 歧义。
#   PROFILE_TARGET=train_overall(整步,默认)| train_actor(前反向更新)
#                 | ref_log_probs / teacher_log_probs / actor_log_probs(细分)
#                 | train_log_probs(旧别名=三个 log_prob 阶段的集合)
#                 ※ 多 target 可同 run(各阶段独立临时 profiler,如 train_actor actor_log_probs)。
#   PROFILE_STEP_START=N → 采第 N 个 train 步(1-based);END=M → 采到第 M-1 步止。
#   PROFILE_RANKS="0"(默认只 rank 0;"0,1" 多 rank)。
#   PROFILE_LEVEL=level0/1/2(默认 level1);PROFILE_EXCLUDE_COMM=1 排通信域降噪。
#   落盘:${TENSORBOARD_DIR:-outputs/profile}/<target>_step<id>_rank<N>/.../*_ascend_pt,离线 analyse。
if [ "${PROFILE_TRAIN:-0}" = "1" ]; then
   MISC_ARGS+=(--use-pytorch-profiler --profile-target "${PROFILE_TARGET:-train_overall}"
               --profile-step-start "${PROFILE_STEP_START:-2}" --profile-step-end "${PROFILE_STEP_END:-4}")
   echo "[profile-train] ON target=${PROFILE_TARGET:-train_overall} steps=(${PROFILE_STEP_START:-2},${PROFILE_STEP_END:-4}]" >&2
fi
# PROFILE_OP=1:rollout(vLLM)侧算子级采集,经 --vllm-profiler-config 转发给 vllm serve。
#   ※ 引擎在 140:140 的 raylet env 也要 PROFILE_OP=1(给 VLLM_RPC_TIMEOUT),所以 worker 脚本里同样设。
#   max_iterations(默认 20)个 engine step 后自动停并落盘到 PROFILE_DIR。
if [ "${PROFILE_OP:-0}" = "1" ]; then
   export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"
   PROFILE_DIR="${PROFILE_DIR:-/home/docker/logs/opprof/$(date +%Y%m%d-%H%M%S)}"
   mkdir -p "${PROFILE_DIR}"
   _PROF_JSON="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILE_DIR}\",\"ignore_frontend\":true,\"max_iterations\":${PROFILE_MAX_ITERS:-20},\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true}"
   VLLM_ARGS+=(--vllm-profiler-config "${_PROF_JSON}")
   echo "[profile-op] ON dir=${PROFILE_DIR} max_iters=${PROFILE_MAX_ITERS:-20} rpc_timeout=${VLLM_RPC_TIMEOUT}" >&2
fi
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
   # [2026-08-10] PD proxy(pd_mooncake_proxy_server.py)进程名不含 "vllm",下面的 pgrep 扫不到,
   #   会跨 run 残留并占用 ${VLLM_ROUTER_PORT}(8011)→ 新 proxy bind 失败、旧 proxy 用过期路由应答
   #   → polar 全报 'no completions'。这里按进程名单独清(不依赖卡号匹配,它就是本节点的)。
   local proxy_pid
   for proxy_pid in $(pgrep -f "pd_mooncake_proxy" 2>/dev/null); do
      echo "[cleanup] kill stale pd proxy pid=$proxy_pid"
      kill "$proxy_pid" 2>/dev/null || true
   done
   local pid env_devs overlap pass hit
   for pass in TERM KILL; do
      hit=0
      for pid in $(pgrep -fi "vllm|pd_mooncake_proxy" 2>/dev/null); do
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
         # ─── 卡位闸门:layout 要求的 (node, devices) 必须真的在集群里 ───
         # layout 的 node 必须与 Ray 看到的节点 IP 一致、devices 必须与该节点暴露的卡号一致,
         # 否则 _build_layout_bundles(placement_group.py:108)会在建 PG 时才报错。这里提前拦。
         # [hybrid] 共卡(share)布局不再是"单 rollout 节点 NPU==rollout_num_gpus":
         # 改按 layout 逐节点校验「专用卡数」(actor 段 + rollout 专用段;共卡段复用 actor 的 bundle,不另计)。
         python3 - "${MASTER_ADDR}" "${ROLLOUT_NODE_IP}" "${ACTOR_NUM_GPUS_PER_NODE}" "${ROLLOUT_NUM_GPUS}" <<'PY'
import os, sys, ray
actor_ip, rollout_ip, want_actor, want_rollout = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
npu = {n["NodeManagerAddress"]: int(n["Resources"].get("NPU", 0)) for n in ray.nodes() if n.get("Alive")}
print(f"[gate] per-node NPU: {npu}")
errs = []
layout = None
layout_path = os.environ.get("RESOURCE_LAYOUT") or None
if layout_path and os.path.exists(layout_path):
    try:
        from vime.ray.resource_layout import load_resource_layout
        layout = load_resource_layout(layout_path)
    except Exception as e:
        print(f"[gate] layout 解析失败({e}),退回扁平校验")
if layout is not None and getattr(layout, "rollout_has_share", False):
    want: dict[str, int] = {}
    for item in layout.actor:
        want[item.node] = want.get(item.node, 0) + len(item.devices)
    for item in layout.rollout:
        if not item.share:
            want[item.node] = want.get(item.node, 0) + len(item.devices)
    for node, cnt in sorted(want.items()):
        if npu.get(node, 0) != cnt:
            errs.append(f"节点 {node} NPU={npu.get(node, 0)},期望 {cnt}(layout 专用卡;共卡段复用 actor 不另计)")
    if not errs:
        print(f"[gate] OK(hybrid): {want}")
else:
    if npu.get(actor_ip, 0) != want_actor:
        errs.append(f"训练节点 {actor_ip} NPU={npu.get(actor_ip, 0)},期望 {want_actor}")
    if npu.get(rollout_ip, 0) != want_rollout:
        errs.append(f"rollout 节点 {rollout_ip} NPU={npu.get(rollout_ip, 0)},期望 {want_rollout}")
    if not errs:
        print(f"[gate] OK: actor {want_actor}卡@{actor_ip} + rollout {want_rollout}卡@{rollout_ip}")
if errs:
    print("[gate][FATAL] 卡位不对,拒绝启动:", *(" - " + e for e in errs), sep="\n")
    sys.exit(1)
PY
         # layout 路径:清全局可见卡,交给 Ray 按 layout 钉卡(只影响 driver,raylet 不受影响)
         unset ASCEND_RT_VISIBLE_DEVICES HCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME TP_SOCKET_IFNAME HCCL_IF_IP
         EXTRA_ARGS=()
         # 注意:本文件开头是 `set -ex`。`[ cond ] && arr+=(...)` 在 cond 为假时整条 AND-list
         # 退出码为 1 → 直接被 set -e 终止。colocate 会把 RESOURCE_LAYOUT 清空、非 colocate 又会
         # 让 FEAT_COLOCATE 判假,两条都会踩到,所以这里一律用 if 而不是 &&。
         if [ -n "${RESOURCE_LAYOUT:-}" ]; then
            EXTRA_ARGS+=(--resource-layout "${RESOURCE_LAYOUT}")
         fi
         if [ "${FEAT_COLOCATE:-0}" = "1" ]; then
            EXTRA_ARGS+=(--colocate)
         fi
         # FEAT_LB_PROXY=1:Python 透传 LB proxy 替 Rust router(保 return_token_ids + 会话亲和);
         #   需把 polar 推理端点指向 :${VLLM_ROUTER_PORT}。见 docs/design/router_return_token_ids_passthrough.md §10。
         if [ "${FEAT_LB_PROXY:-0}" = "1" ]; then
            EXTRA_ARGS+=(--rollout-lb-proxy)
         fi
         # ─── 启动 vLLM metrics 监控面板(旁路,失败不影响训练)───
         # 引擎由 head 的 driver 远程创建在 rollout 节点 → 发现目标必须是
         # ROLLOUT_NODE_IP(140),不是 CURRENT_IP(141 本机没有任何 engine)。
         # 面板跑在 head 上,人在 141 直接看 http://<141>:5000。
         # METRICS_ENGINE_INTERNAL_PORTS = 1(nccl)+tp:发现 engine 后自动隔离其
         # Mooncake bootstrap 端口,避免 GET /metrics 被当二进制长度前缀读,
         # 在 vllm 日志刷 readString/SocketHandShakePlugin 报错。
         METRICS_DASHBOARD_PORT=${METRICS_DASHBOARD_PORT:-5000} \
         METRICS_HOST_IP="${ROLLOUT_NODE_IP}" \
         METRICS_ENGINE_INTERNAL_PORTS=$((1 + ROLLOUT_NUM_GPUS_PER_ENGINE)) \
            source "${VIME_ROOT}/scripts/common/start_metrics_monitor.sh"
         python3 "${TRAIN_ENTRY}" \
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
   # worker(rollout 节点):只加入集群。引擎与 PD proxy 由 head 的 driver 远程创建,
   # 从本节点 raylet 继承环境 → 上面那些 export 必须在 ray start 之前完成。
   ray stop --force
   rm -rf "${RAY_TEMP_DIR}"
   cleanup_rollout_residue
   while true; do
      ray start --address="${MASTER_ADDR}:${RAY_PORT}" --node-ip-address="${CURRENT_IP}" --num-gpus="${NPUS_PER_NODE}" --resources='{"NPU": '"${NPUS_PER_NODE}"'}' --temp-dir="${RAY_TEMP_DIR}" --disable-usage-stats
      ray status && break
      sleep 5
   done
   set +x
   echo "[worker] 已加入 ${MASTER_ADDR}:${RAY_PORT},注册 NPU=${NPUS_PER_NODE}。"
   echo "[worker] 引擎/PD proxy 由 head 的 driver 远程创建;本脚本结束,raylet 常驻后台。"
   echo "[worker] 引擎日志:${RAY_TEMP_DIR}/session_latest/logs/"
fi
