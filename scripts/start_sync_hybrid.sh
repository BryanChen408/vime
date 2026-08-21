# ─────────────────────────────────────────────────────────────────────────────
# start_sync_hybrid.sh — 混合同步部署:.56 训推共卡 + .64 专用推理(只做推理)
#
# 与 start_pd.sh(异步分离)/ start_cola.sh(纯 colocate)的差异:
#   * 入口 train.py(同步循环:收齐 → 训练 → 权重同步 → 再收),TRAIN_ENTRY 直切。
#   * 布局 resource_layout.hybrid56cola64infer.yaml:
#       .56 16 卡 = 训练 + 8 个 TP2 引擎(共卡分时);.64 4-15 = 6 个 TP2 引擎(纯推理)。
#   * FEAT_OFFLOAD=1:训推双侧 offload(训练窗口全部引擎 sleep;.56 引擎靠 needs_offload
#     命中,.64 引擎不重叠不 sleep、常驻 —— 但网关被 prepare_policy_update 暂停,语义仍是同步)。
#   * 权重同步走分布式 HCCL(world=1+28),不经 colocate 的 IPC 路径。
#
# 前置(.64 手工):
#   1) polar 推理端点 → http://80.48.5.56:8011;
#   2) polar 卡池 "0-15" 改为 "0-3"(或按实际剩余);
#   3) .64 先跑 start_pd_worker.sh 加入 Ray(同 pd 流程);老的分离部署进程先清干净。
#
# 内存账(61G/卡,VLLM_GPU_MEM_UTIL=0.80 → 引擎预留 ~49G):
#   rollout 窗口:.56 引擎 49G + 训练参数驻留 ~5G(default offloader 不卸参数) ✓
#   train 窗口:训练 ~52G + 引擎 sleep 释放 ✓
#   权重同步:引擎醒(权重 35G,无 KV)+ 训练参数+聚合缓冲 ~7G ✓
#   host 预算(共卡 4 引擎驻留 4×70G=280G):训练 host ~700G + 驻留 280G + plasma 200G
#     + 基线 ~150G ≈ 1.33T / 2T,富余充足。(磁盘重载 level=2 路径已否决:reload 在丢弃后
#     storage 上 ACL 失败;故共卡引擎从 8 减到 4 来控制驻留。)
#   切勿设 VIME_OFFLOAD_PARAM_BUFFER=1(B 模式卸参数,会在权重同步时无参数可发)。
# ─────────────────────────────────────────────────────────────────────────────
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
CURRENT_IP=80.48.5.56  MASTER_ADDR=80.48.5.56  NNODES=2  NPUS_PER_NODE=16  SOCKET_IFNAME=ens1f3 \
ACTOR_NUM_GPUS_PER_NODE=16 \
TRAIN_ENTRY=train.py \
FEAT_OFFLOAD=1 \
RESOURCE_LAYOUT=/workspace/vime/scripts/resource_layout.hybrid56cola64infer.yaml \
ROLLOUT_NODE_IP=80.48.5.56 \
ROLLOUT_NUM_GPUS=28 \
ROLLOUT_NUM_GPUS_PER_ENGINE=2 \
FEAT_PD_DISAGG=0 \
VLLM_GPU_MEM_UTIL=0.80 \
MAX_TOKENS_PER_GPU=32768 \
SEQ_LENGTH=262144 \
ROLLOUT_MAX_CONTEXT_LEN=262144 \
VLLM_MAX_MODEL_LEN=262144 \
VIME_MEM_PROBE=1 \
RAY_memory_usage_threshold=0.99 \
no_proxy=127.0.0.1,localhost,80.48.5.56,80.48.5.64,80.5.25.140,.huawei.com,local,.local \
NO_PROXY=127.0.0.1,localhost,80.48.5.56,80.48.5.64,80.5.25.140,.huawei.com,local,.local \
TP=2 \
POLAR_TRAJECTORY_PG_FLOOR=0.05 \
POLAR_ROLLOUT_URL=http://80.48.5.64:8180 \
POLAR_DRAIN_SESSIONS=0 \
POLAR_MAX_OFF_POLICY_STEPS=1 \
VLLM_ROUTER_PORT=8011 \
FEAT_TRAIN_EXPANDABLE=1 \
VIME_EMPTY_CACHE_PER_STEP=1 \
POLAR_MAX_ACTIVE_SESSIONS=32 \
TRANSFORMERS_VERBOSITY=error \
HCCL_INTER_HCCS_DISABLE=false \
HCCL_INTRA_ROCE_ENABLE=1 \
HCCL_INTRA_PCIE_ENABLE=0 \
HCCL_BUFFSIZE=512 \
HCCL_HOST_SOCKET_PORT_RANGE=60000-60255 \
HCCL_NPU_SOCKET_PORT_RANGE=61000-61255 \
ROLLOUT_BATCH_SIZE=4  N_SAMPLES_PER_PROMPT=8  GLOBAL_BATCH_SIZE=32  NUM_ROLLOUT=100 \
FEAT_DP_EXTERNAL_LB=0 FEAT_BALANCE_SCHED=0 FEAT_LB_PROXY=1 FEAT_CROSS_DP_EP=0 \
FEAT_ROLLOUT_EP=0 FEAT_FLASHCOMM1=0 FEAT_PREFIX_CACHE=1 FEAT_MULTISTREAM_SHARED_EXPERT=1 FEAT_STATIC_KERNEL=0 FEAT_HCCL_AIV=1 \
OPERATOR_DATA_ROOT=/home/docker/datasets/op_tasks/op_assets_cudallm_filtered189 \
OPERATOR_TASK_JSONL=/home/docker/datasets/op_tasks/op_assets_cudallm_filtered189/operator_tasks.16.jsonl \
PROFILE_TRAIN=0 \
PROFILE_TARGET=train_overall \
PROFILE_STEP_START=1 PROFILE_STEP_END=2 \
TENSORBOARD_DIR=/home/docker/logs/prof/$(date +%Y%m%d-%H%M%S) \
PROFILE_OP=0 \
PROFILE_RANKS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" \
bash scripts/run-qwen36-35b-polar-multi-pd.sh
# ── 冒烟建议:先 NUM_ROLLOUT=2 跑通"建组 → 引擎 offload → 训练模型加载 → 首个 generate →
#   drain → sleep → train → onload → 权重同步 → resume"全链路,再放大。
# ── 回退:改用 bash scripts/start_pd.sh(异步分离)即可,两脚本互不影响。
