# ─────────────────────────────────────────────────────────────────────────────
# start_sync_homo_single52.sh — 单机 16 卡(.52)**部分共卡(T5)**冒烟
#
# ⚠ 文件名里的 "homo" 是历史遗留,**这不是同构**。actor 占 16 卡而 rollout 只占其中
#   12 张(4-15),是 share ⊊ actor 的部分共卡。真同构要求两者完全同一批卡,但这台机器上
#   polar 必须占 0-3,做不出来(actor 只能 4-15 = 12 卡,TP2×PP1×CP6=12 而 EP8 除不进 12)。
#   20260828 我把它当同构,因而漏掉「rollout 段起点不在 actor 第 0 张卡」这件事,
#   IPC gather group 拿 rollout 槽位当 actor rank,整体错位 4 张卡 →
#   "IPC handle not found for GPU UUID ...";已由 engine_roles 的 actor_ranks 修正,
#   回归见 tests/test_engine_roles.py。
#
# layout: scripts/resource_layout.single52_homo_colocate.yaml
#   0-3    polar agent/judge(与 actor 重叠,layout 中无法声明 —— 见该文件注释)
#   0-15   actor 训练(16 卡)
#   4-15   6 台 TP2 引擎,全部共卡,训练窗口 sleep,权重走 NPU IPC
#
# 为什么先跑同构:**变量最少**。TP2×PP1×CP8=16 与 .56 生产配置逐位相同,长度也对齐
#   262144,唯一的差异只有 batch(生产 1/4)。用它先把「训练步能不能跑通」和「显存残留
#   到底多少」这两个未验证项拿下来,再回异构。
#
#   注:单机异构曾崩在三处共卡判据分叉(引擎侧 backend 按节点 / 权重通道按总 actor 卡 /
#   offload 按 share 卡),「专用段与 actor 同节点」时引擎起成 npu_ipc 而 trainer 发 HCCL
#   init info → NPUIPCWeightTransferInitInfo.__init__() got an unexpected keyword
#   argument 'master_address' → 500。已由 1448b630 归一到 vime/ray/engine_roles.py,并在
#   20260828 真机跑中验证(权重同步 /finish_weight_update 200,全日志无该异常;
#   Sleep mode 只出现在 4 台共卡引擎上,专用 2 台常驻,offload 判据也正确)。
#   所以异构现在**可跑**,先同构只是为了少变量,不是因为异构还坏着。
#   详见 docs/design/colocate_topology_robustness_plan.md。
#
# 并行度 TP2 × PP1 × CP8 = 16,占满 actor 全部 16 卡 —— **与 .56 生产配置逐位相同**,
#   不引入 CP4 那个仓内无先例的变量。
#
# 端口与 polar 侧 profile.sing52.yaml 严格对齐(不匹配 = 全量 404):
#   POLAR_ROLLOUT_URL=:8080   ← service.rollout_url
#   VLLM_ROUTER_PORT=8001     ← service.sglang_router_url(LB proxy)
#
# ── 启动 ────────────────────────────────────────────────────────────────
#   1) polar(宿主机),profile 无需编辑:
#        POLAR_PROFILE=deploy/ascend_operator/profile.sing52.yaml \
#        POLAR_RUN_ID=polar_$(date +%Y%m%d_%H%M%S) \
#        bash deploy/ascend_operator/restart_polar_host.sh
#   2) 本脚本:
#        NUM_ROLLOUT=2 bash scripts/start_sync_homo_single52.sh
#
# ⚠ **长度是能力参数,不是显存旋钮;要压显存请压 batch**。
#   长度三件套按 MAX_TOKENS_PER_GPU × CP 拉齐 = 32768 × 8 = **262144**,与生产 .56 逐位相同
#   (这个上界来自 vime_bridge 的 _resolve_max_tokens:超过它的轨迹会被 batcher 丢弃,
#    所以配更大无意义,配更小则是白扔能力)。冒烟靠的是 batch 降到生产的 1/4:
#     ROLLOUT_BATCH_SIZE 8→4, N_SAMPLES_PER_PROMPT 8→4, GLOBAL_BATCH_SIZE 64→16。
#   20260828 异构首跑违反了这条:把三件套压到 32768 想省显存,结果 64/64 个 session 在
#   第 0 轮就被引擎打回 400 —— agent 的第 0 轮 prompt 已 20481 token,polar profile 的
#   max_output_tokens=12288,相加 32769,比 32768 多 1。零 completion → 零可训 token →
#   同步路径把每个组全拒掉并无限 top up,一个训练步都没进去。**长度存在硬下限,压它
#   不是"少跑一点",而是直接让 agent 发不出第一个请求。**
#
# ⚠ judge 池是瓶颈:ROLLOUT_BATCH_SIZE×N_SAMPLES = 16 个 session 抢 4 张判题卡
#   (npu_lease.pool)。这会**放大 polar/sync/tail_ratio**,别误读成"需要超订+abort"
#   —— 那是判题池串行化,不是生成长尾。区分:看 group_seconds_min 是否也被拖长。
#
# 验收:
#   * Sleep mode (level=2) freed ... GiB   × 6(全部引擎都该 sleep)
#   * polar/sync/accepted_groups == 4
#   * handoff:rollout 0 after train offload 的 non_torch 占比 → 定 util 天花板
#     与 Phase C(TMS)做不做
# ─────────────────────────────────────────────────────────────────────────────

# 清掉上一轮崩溃残留的 LB proxy。它由 rollout.py:_start_lb_proxy 用裸
# subprocess.Popen 起,driver 一崩就被 init 收养、继续占着端口,而 `ray stop --force`
# 管不到它(不是 Ray actor)。注意不能用 fuser/lsof —— 本机都没装,
# `fuser ... || true` 会静默无操作。
_LB_PORT="${VLLM_ROUTER_PORT:-8001}"
pkill -f "vime\.ray\.lb_proxy .*--port ${_LB_PORT}" 2>/dev/null || true
pkill -f "dp_load_balance_proxy_server.*${_LB_PORT}" 2>/dev/null || true
for _ in 1 2 3 4 5; do
   ss -tln 2>/dev/null | grep -q ":${_LB_PORT} " || break
   sleep 1
done
if ss -tln 2>/dev/null | grep -q ":${_LB_PORT} "; then
   echo "[start_sync_homo][FATAL] 端口 ${_LB_PORT} 仍被占用,LB proxy 起不来:" >&2
   ss -tlnp 2>/dev/null | grep ":${_LB_PORT} " >&2
   exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Keep the JSON out of a ${var:-...} expansion: a caller-provided value must
# not inherit the closing brace from the default literal.
if [ "${FEAT_MTP:-1}" = "1" ] && [ -z "${VLLM_SPEC_CONFIG:-}" ]; then
   VLLM_SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":3}'
fi

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
CURRENT_IP=80.48.5.52  MASTER_ADDR=80.48.5.52  NNODES=1  NPUS_PER_NODE=16  SOCKET_IFNAME=ens1f3 \
ACTOR_NUM_NODES=1 \
ACTOR_NUM_GPUS_PER_NODE=16 \
TRAIN_ENTRY=train.py \
FEAT_OFFLOAD=1 \
FEAT_SYNC_ROLLOUT=1 \
FEAT_MTP=${FEAT_MTP:-1} \
FEAT_MTP_TRAIN=${FEAT_MTP_TRAIN:-1} \
RESOURCE_LAYOUT="${SCRIPT_DIR}/resource_layout.single52_homo_colocate.yaml" \
ROLLOUT_NODE_IP=80.48.5.52 \
ROLLOUT_NUM_GPUS=12 \
ROLLOUT_NUM_GPUS_PER_ENGINE=2 \
FEAT_PD_DISAGG=0 \
HF_CKPT="${HF_CKPT:-/home/docker/Qwen3.6-35B-A3B}" \
REF_LOAD="${REF_LOAD:-/workspace/Qwen3.6-35B-A3B_mtp_torch_dist}" \
VLLM_SERVED_MODEL_NAME=/home/docker/Qwen3.6-35B-A3B \
VLLM_SPEC_CONFIG="${VLLM_SPEC_CONFIG}" \
VLLM_GPU_MEM_UTIL=0.70 \
MAX_TOKENS_PER_GPU=32768 \
SEQ_LENGTH=262144 \
ROLLOUT_MAX_CONTEXT_LEN=262144 \
VLLM_MAX_MODEL_LEN=262144 \
VIME_MEM_PROBE=1 \
RAY_memory_usage_threshold=0.95 \
no_proxy=127.0.0.1,localhost,80.48.5.52,.huawei.com,local,.local \
NO_PROXY=127.0.0.1,localhost,80.48.5.52,.huawei.com,local,.local \
TP=2 \
PP=1 \
CP=8 \
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
ROLLOUT_BATCH_SIZE=2  N_SAMPLES_PER_PROMPT=2  GLOBAL_BATCH_SIZE=4  NUM_ROLLOUT=${NUM_ROLLOUT:-2} \
FEAT_DP_EXTERNAL_LB=0 FEAT_BALANCE_SCHED=0 FEAT_LB_PROXY=1 FEAT_CROSS_DP_EP=0 \
FEAT_ROLLOUT_EP=0 FEAT_FLASHCOMM1=0 FEAT_PREFIX_CACHE=1 FEAT_MULTISTREAM_SHARED_EXPERT=1 FEAT_STATIC_KERNEL=0 FEAT_HCCL_AIV=1 \
OPERATOR_DATA_ROOT=/home/docker/datasets/op_tasks/op_assets_cudallm_filtered189 \
OPERATOR_TASK_JSONL=/home/docker/datasets/op_tasks/op_assets_cudallm_filtered189/operator_tasks.16.jsonl \
PROFILE_TRAIN=0 \
bash "${SCRIPT_DIR}/run-qwen36-35b-polar-multi-pd.sh"
# ── 回退:bash scripts/start_pd.sh(异步分离),两脚本互不影响。
