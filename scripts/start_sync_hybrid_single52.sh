# ─────────────────────────────────────────────────────────────────────────────
# start_sync_hybrid_single52.sh — 单机 16 卡(.52)异构共卡冒烟
#
# 目的:在一台机器上把「训推共卡 + 专用推理」的**混合**形态跑起来,验证同步训推切换。
#   layout: scripts/resource_layout.single52_hybrid_colocate.yaml
#     0-3    polar agent/judge(不进 Ray)
#     4-11   actor 训练 + 4 台共卡 TP2 引擎(训练窗口 sleep,权重走 IPC)
#     12-15  2 台专用 TP2 引擎(常驻不 sleep,权重走 HCCL)
#   已用真实加载器离线验证过切分:4 IPC+sleep / 2 HCCL+常驻(异构成立)。
#
# 为什么不用同构(训推完全重合):同构只会走 IPC 一条路,测不到混合更新器的
# IPC/HCCL 分叉,也测不到 _offload_engine_indices 的部分 sleep。异构才是这批
# 改动的真实覆盖面。
#
# ⚠ 训练卡数从 .56 的 16 砍到 8,**每 rank 承担的参数翻倍**。若首训步 OOM,先动并行度
#   或换小模型 —— 不要靠调低 VLLM_GPU_MEM_UTIL 硬凑:训练窗口里共卡引擎全在睡,调它
#   对训练侧毫无帮助,只会把爆点挪到后面 KV 唤醒时。
#
# ⚠ **长度是能力参数,不是显存旋钮;要压显存请压 batch**。
#   长度三件套按 MAX_TOKENS_PER_GPU × CP 拉齐 = 32768 × 4 = **131072**(CP4 比生产 .56 的
#   CP8 少一半,所以是 131072 而非 262144 —— 这个上界来自 vime_bridge 的
#   _resolve_max_tokens:超过它的轨迹会被 batcher 丢弃,配更大无意义,配更小是白扔能力)。
#   冒烟靠的是 batch 降到生产的 1/4:
#     ROLLOUT_BATCH_SIZE 8→4, N_SAMPLES_PER_PROMPT 8→4, GLOBAL_BATCH_SIZE 64→16。
#   20260828 首跑违反了这条:把三件套压到 32768 想省显存,结果 64/64 个 session 在第 0 轮
#   就被引擎打回 400 —— agent 的第 0 轮 prompt 已 20481 token,polar profile 的
#   max_output_tokens=12288,相加 32769,比 32768 多 1:
#     "maximum context length is 32768 ... requested 12288 output tokens and your
#      prompt contains at least 20481 input tokens, for a total of at least 32769"
#   零 completion → 零可训 token → 同步路径把每个组全拒掉并无限 top up,一个训练步都没
#   进去。**长度存在硬下限,压它不是"少跑一点",而是直接让 agent 发不出第一个请求。**
#
# 训练并行度:TP2 × PP1 × CP4 = 8 rank,恰好占满 actor 的 4-11 八卡(EP8 在 8 rank 上切专家)。
#   runner 默认 CP8(为 .56 的 16 卡准备),8 卡必须改 CP4,否则乘积对不上卡数。
#   注:这个切分来自 vime_polar_sync_colocate_design.md §1.2,但**仓内无脚本先例**
#   (grep CP=4 只命中本文件),Megatron 侧是否接受 CP4+EP8 的组合首跑才知道。
#
# 端口与 polar 侧 profile.sing52.yaml 严格对齐(不匹配就是全量 404 / 连不上):
#   POLAR_ROLLOUT_URL=:8080   ← profile 的 service.rollout_url
#   VLLM_ROUTER_PORT=8001     ← profile 的 service.sglang_router_url(LB proxy 端口)
#
# ── 启动 ────────────────────────────────────────────────────────────────
#   1) polar 侧(宿主机):
#        POLAR_PROFILE=deploy/ascend_operator/profile.sing52.yaml \
#        POLAR_RUN_ID=polar_$(date +%Y%m%d_%H%M%S) \
#        bash deploy/ascend_operator/restart_polar_host.sh
#      该 profile 无需改动:npu_lease.pool 已是 [0,1,2,3]、inference_engine 已是 vllm、
#      model_served 已是 /home/docker/Qwen3.6-35B-A3B。
#   2) 本脚本:
#        NUM_ROLLOUT=2 bash scripts/start_sync_hybrid_single52.sh
#
# ⚠ judge 池是瓶颈:ROLLOUT_BATCH_SIZE×N_SAMPLES_PER_PROMPT = 16 个 session,判题却只有
#   4 张卡(npu_lease.pool)。16 个 session 同时完成时会在 NPU lease 上排队,这会**放大
#   polar/sync/tail_ratio**。别把这个尾巴误读成"需要超订+abort" —— 它是判题池串行化,
#   不是生成长尾。要区分:看 group_seconds_min 是否也被拖长(是→池contention)。
#
# 验收看两处日志:
#   * polar/sync/accepted_groups == ROLLOUT_BATCH_SIZE(=4);
#   * handoff:rollout 0 after train offload 的 non_torch 占比 —— 决定 util 天花板
#     和 Phase C(TMS)值不值得做。
# ─────────────────────────────────────────────────────────────────────────────
# 清掉上一轮崩溃残留的 LB proxy。它由 rollout.py:_start_lb_proxy 用裸
# subprocess.Popen 起,driver 一崩就被 init 收养、继续占着端口,而 `ray stop --force`
# 管不到它(不是 Ray actor)。注意不能用 fuser/lsof —— 本机都没装,静默无操作。
_LB_PORT="${VLLM_ROUTER_PORT:-8001}"
pkill -f "vime\.ray\.lb_proxy .*--port ${_LB_PORT}" 2>/dev/null || true
pkill -f "dp_load_balance_proxy_server.*${_LB_PORT}" 2>/dev/null || true
for _ in 1 2 3 4 5; do
   ss -tln 2>/dev/null | grep -q ":${_LB_PORT} " || break
   sleep 1
done
if ss -tln 2>/dev/null | grep -q ":${_LB_PORT} "; then
   echo "[start_sync_hybrid][FATAL] 端口 ${_LB_PORT} 仍被占用,LB proxy 起不来:" >&2
   ss -tlnp 2>/dev/null | grep ":${_LB_PORT} " >&2
   exit 1
fi

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7,8,9,10,11,12,13,14,15 \
CURRENT_IP=80.48.5.52  MASTER_ADDR=80.48.5.52  NNODES=1  NPUS_PER_NODE=12  SOCKET_IFNAME=ens1f3 \
ACTOR_NUM_NODES=1 \
ACTOR_NUM_GPUS_PER_NODE=8 \
TRAIN_ENTRY=train.py \
FEAT_OFFLOAD=1 \
FEAT_SYNC_ROLLOUT=1 \
RESOURCE_LAYOUT=/workspace/vime/scripts/resource_layout.single52_hybrid_colocate.yaml \
ROLLOUT_NODE_IP=80.48.5.52 \
ROLLOUT_NUM_GPUS=12 \
ROLLOUT_NUM_GPUS_PER_ENGINE=2 \
FEAT_PD_DISAGG=0 \
VLLM_SERVED_MODEL_NAME=/home/docker/Qwen3.6-35B-A3B \
VLLM_GPU_MEM_UTIL=0.70 \
VLLM_GPU_MEM_UTIL_DEDICATED=0.85 \
MAX_TOKENS_PER_GPU=32768 \
SEQ_LENGTH=131072 \
ROLLOUT_MAX_CONTEXT_LEN=131072 \
VLLM_MAX_MODEL_LEN=131072 \
VIME_MEM_PROBE=1 \
RAY_memory_usage_threshold=0.95 \
no_proxy=127.0.0.1,localhost,80.48.5.52,.huawei.com,local,.local \
NO_PROXY=127.0.0.1,localhost,80.48.5.52,.huawei.com,local,.local \
TP=2 \
PP=1 \
CP=4 \
EP=8 \
POLAR_TRAJECTORY_PG_FLOOR=0.05 \
POLAR_ROLLOUT_URL=http://80.48.5.52:8080 \
VLLM_ROUTER_PORT=8001 \
FEAT_TRAIN_EXPANDABLE=1 \
VIME_EMPTY_CACHE_PER_STEP=1 \
TRANSFORMERS_VERBOSITY=error \
HCCL_INTER_HCCS_DISABLE=false \
HCCL_INTRA_ROCE_ENABLE=1 \
HCCL_INTRA_PCIE_ENABLE=0 \
HCCL_BUFFSIZE=512 \
HCCL_HOST_SOCKET_PORT_RANGE=60000-60255 \
HCCL_NPU_SOCKET_PORT_RANGE=61000-61255 \
ROLLOUT_BATCH_SIZE=4  N_SAMPLES_PER_PROMPT=4  GLOBAL_BATCH_SIZE=16  NUM_ROLLOUT=${NUM_ROLLOUT:-2} \
FEAT_DP_EXTERNAL_LB=0 FEAT_BALANCE_SCHED=0 FEAT_LB_PROXY=1 FEAT_CROSS_DP_EP=0 \
FEAT_ROLLOUT_EP=0 FEAT_FLASHCOMM1=0 FEAT_PREFIX_CACHE=1 FEAT_MULTISTREAM_SHARED_EXPERT=1 FEAT_STATIC_KERNEL=0 FEAT_HCCL_AIV=1 \
OPERATOR_DATA_ROOT=/home/docker/datasets/op_tasks/op_assets_cudallm_filtered189 \
OPERATOR_TASK_JSONL=/home/docker/datasets/op_tasks/op_assets_cudallm_filtered189/operator_tasks.16.jsonl \
PROFILE_TRAIN=0 \
bash scripts/run-qwen36-35b-polar-multi-pd.sh
# ── 回退:改用 bash scripts/start_pd.sh(异步分离)即可,两脚本互不影响。
