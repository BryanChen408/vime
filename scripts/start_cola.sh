# ─────────────────────────────────────────────────────────────────────────────
# colocate(训推同卡 + 同步 train.py)启动配置
#
# 拓扑变更(相对 start_pd.sh 的分离部署):
#   .56  16 卡  训练 + 推理共卡(colocate),单节点 Ray,NNODES=1
#   .64  16 卡  全部交给 polar(不再进 Ray 集群,不再跑 vime worker)
#
# 与分离部署的关键差异 —— 这些是代码强制的,不是风格选择:
#   * train_async.py:11  assert not args.colocate      → 入口必须是 train.py(FEAT_COLOCATE=1 自动切)
#   * arguments.py:1817  layout 与 --colocate 互斥      → RESOURCE_LAYOUT 必须为空(runner 里自动清)
#   * arguments.py:1888  强制 offload_train/rollout=True → 不能再传 --no-offload-*(runner 里自动去掉)
#   * arguments.py:1907  rollout_num_gpus 被改写成
#                        actor_num_gpus_per_node*actor_num_nodes = 16
#                        → 这里必须自己就写 16,否则只是被静默覆盖
#
# 代价(已知,非 bug):生成与训练不再重叠。每轮训练那段时间引擎 sleep,polar 网关被
#   prepare_policy_update 暂停;训练结束后 session 从头开始,没有预取的存量可用。
#   若发现每轮总时长反而变长,退回 start_pd.sh(把 FEAT_COLOCATE 去掉即可)。
#
# 权重边界怎么处理(polar 侧没有 /admin/policy_version,version-span guard 会 404 降级,
# 所以跨界的组只能在 vime 侧丢):
#   POLAR_MAX_OFF_POLICY_STEPS=0  只收本轮开的组,上一轮遗留的一律丢 —— "在跑的就不要了"。
#                                 代价:每轮尾部 polar 抢跑出来的组作废。
#   POLAR_MAX_OFF_POLICY_STEPS=1  接受跨一次更新的组(混权轨迹,靠 TIS 兜),不浪费。
#   不设(留空)                     沿用 max_async_level+update_weights_interval 的推导值(=2)。
# colocate 默认取 0;要换就在上面显式写 POLAR_MAX_OFF_POLICY_STEPS=1。
#
# 需要在 .64 上同步改的两件事(vime 侧管不到):
#   1) polar 的推理端点 → http://80.48.5.56:8011(引擎跟着 actor 搬回 .56 了)
#   2) polar 的 ascend 卡池 pool: "0-15"(原来只有 4 张;见
#      ProRL-Agent-Server/src/polar/runtime/docker.py:102 → ascend.py:70 parse_pool,
#      支持 "0-15" 区间写法)
# ─────────────────────────────────────────────────────────────────────────────
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
CURRENT_IP=80.48.5.56  MASTER_ADDR=80.48.5.56  NNODES=1  NPUS_PER_NODE=16  SOCKET_IFNAME=ens1f3 \
FEAT_COLOCATE=1 \
ACTOR_NUM_NODES=1 \
ACTOR_NUM_GPUS_PER_NODE=16 \
ROLLOUT_NUM_GPUS=16 \
ROLLOUT_NUM_GPUS_PER_ENGINE=2 \
FEAT_PD_DISAGG=0 \
VLLM_GPU_MEM_UTIL=0.80 \
MAX_TOKENS_PER_GPU=32768 \
SEQ_LENGTH=262144 \
ROLLOUT_MAX_CONTEXT_LEN=262144 \
VLLM_MAX_MODEL_LEN=262144 \
VIME_MEM_PROBE=1 \
no_proxy=127.0.0.1,localhost,80.48.5.56,80.48.5.64,80.5.25.140,.huawei.com,local,.local \
NO_PROXY=127.0.0.1,localhost,80.48.5.56,80.48.5.64,80.5.25.140,.huawei.com,local,.local \
TP=2 \
POLAR_TRAJECTORY_PG_FLOOR=0.05 \
POLAR_ROLLOUT_URL=http://80.48.5.64:8180 \
VLLM_ROUTER_PORT=8011 \
FEAT_TRAIN_EXPANDABLE=1 \
VIME_EMPTY_CACHE_PER_STEP=1 \
POLAR_MAX_ACTIVE_SESSIONS=32 \
TRANSFORMERS_VERBOSITY=error \
HCCL_INTER_HCCS_DISABLE=false \
HCCL_INTRA_ROCE_ENABLE=0 \
HCCL_INTRA_PCIE_ENABLE=1 \
HCCL_BUFFSIZE=512 \
ROLLOUT_BATCH_SIZE=4  N_SAMPLES_PER_PROMPT=8  GLOBAL_BATCH_SIZE=32  NUM_ROLLOUT=100 \
POLAR_MAX_OFF_POLICY_STEPS=1 \
FEAT_DP_EXTERNAL_LB=0 FEAT_BALANCE_SCHED=0 FEAT_LB_PROXY=1 FEAT_CROSS_DP_EP=0 \
FEAT_ROLLOUT_EP=0 FEAT_FLASHCOMM1=0 FEAT_PREFIX_CACHE=1 FEAT_MULTISTREAM_SHARED_EXPERT=1 FEAT_STATIC_KERNEL=0 FEAT_HCCL_AIV=1 \
OPERATOR_DATA_ROOT=/home/docker/datasets/op_tasks/op_assets_cudallm_filtered189 \
OPERATOR_TASK_JSONL=/home/docker/datasets/op_tasks/op_assets_cudallm_filtered189/operator_tasks.ascendc.jsonl \
PROFILE_TRAIN=0 \
PROFILE_TARGET=train_overall \
PROFILE_STEP_START=1 PROFILE_STEP_END=2 \
TENSORBOARD_DIR=/home/docker/logs/prof/$(date +%Y%m%d-%H%M%S) \
PROFILE_OP=0 \
PROFILE_RANKS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" \
bash scripts/run-qwen36-35b-polar-multi-pd.sh
# ── 显存不够时的调节顺序(colocate 下引擎在 create 阶段按空卡算 0.80,训练模型随后加载)──
#   1) VLLM_GPU_MEM_UTIL 0.80 → 0.70
#   2) VLLM_MAX_NUM_SEQS(默认 96)调小
#   3) MAX_TOKENS_PER_GPU 32768 → 16384
# ── drain 超时:polar session 排空最多等 300s,可用 --polar-weight-update-pause-timeout 调;
#    超时不再中断训练(train.py 里已改为 best-effort),只是少数 in-flight session 变 ERROR 被丢弃。
