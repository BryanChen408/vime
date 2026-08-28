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
# ⚠ 训练卡数从 .56 的 16 卡砍到 8 卡,**每 rank 承担的参数翻倍**。35B-A3B 在 8 卡上
#   能否放下未经验证 —— 先按下面「阶段 1」用 --debug-train-only 单独确认训练侧,
#   再跑本脚本。若放不下,优先降 SEQ_LENGTH,其次换小模型,不要靠调低
#   VLLM_GPU_MEM_UTIL 硬凑(那只会把问题挪到 KV 唤醒时爆)。
#
# ── 阶段 1:训练侧可行性(不占推理卡,不用 layout)──────────────────────────
#   --resource-layout 与 --debug-train-only 互斥(arguments.py 直接 raise),
#   所以可行性探测必须**清空 layout** 再从 VIME_EXTRA_ARGS 传进去:
#     ASCEND_RT_VISIBLE_DEVICES=4,5,6,7,8,9,10,11 NPUS_PER_NODE=8 \
#     ACTOR_NUM_GPUS_PER_NODE=8 ACTOR_NUM_NODES=1 NNODES=1 \
#     CURRENT_IP=80.48.5.52 MASTER_ADDR=80.48.5.52 \
#     RESOURCE_LAYOUT="" TRAIN_ENTRY=train.py \
#     VIME_EXTRA_ARGS="--debug-train-only" \
#     SEQ_LENGTH=32768 NUM_ROLLOUT=1 \
#     bash scripts/run-qwen36-35b-polar-multi-pd.sh
#   看它能不能走完首个 train step。走不完就先解决这个,别急着上全链路。
#
# ── 阶段 2:全链路(本脚本)──────────────────────────────────────────────
#   NUM_ROLLOUT=2 bash scripts/start_sync_hybrid_single52.sh
#
# 前置(polar 侧,手工):
#   1) profile 的 npu_lease.pool 改成 "0,1,2,3" —— 必须落在训练卡之外,否则 judge 抢卡;
#   2) 推理端点指向 http://80.48.5.52:${VLLM_ROUTER_PORT};
#   3) 模型名保持 /home/docker/Qwen3.6-35B-A3B(靠 VLLM_SERVED_MODEL_NAME 别名解耦,
#      换模型目录不用动 polar)。
#
# 验收看两处日志:
#   * polar/sync/accepted_groups == ROLLOUT_BATCH_SIZE,且 tail_ratio 记下来
#     (>1 明显说明尾部空转,超订+abort 才值得做);
#   * handoff:rollout 0 after train offload 的 non_torch 占比 —— 决定 util 天花板
#     和 Phase C(TMS)值不值得做。
# ─────────────────────────────────────────────────────────────────────────────
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
SEQ_LENGTH=32768 \
ROLLOUT_MAX_CONTEXT_LEN=32768 \
VLLM_MAX_MODEL_LEN=32768 \
VIME_MEM_PROBE=1 \
RAY_memory_usage_threshold=0.95 \
no_proxy=127.0.0.1,localhost,80.48.5.52,.huawei.com,local,.local \
NO_PROXY=127.0.0.1,localhost,80.48.5.52,.huawei.com,local,.local \
TP=1 \
PP=2 \
POLAR_TRAJECTORY_PG_FLOOR=0.05 \
POLAR_ROLLOUT_URL=http://80.48.5.52:8180 \
VLLM_ROUTER_PORT=8011 \
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
