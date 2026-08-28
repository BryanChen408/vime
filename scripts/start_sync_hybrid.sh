# ─────────────────────────────────────────────────────────────────────────────
# start_sync_hybrid.sh — 混合同步部署:.56 训推共卡 + .64 专用推理(只做推理)
#
# 与 start_pd.sh(异步分离)/ start_cola.sh(纯 colocate)的差异:
#   * 入口 train.py(同步循环:收齐 → 训练 → 权重同步 → 再收),TRAIN_ENTRY 直切。
#   * 布局 resource_layout.hybrid56cola64infer.yaml:
#       .56 16 卡 = 训练 + 8 个 TP2 引擎(共卡分时);.64 4-15 = 6 个 TP2 引擎(纯推理)。
#   * 权重同步走混合更新器(UpdateWeightFromTensor):.56 共卡引擎 NPU IPC 直传(不进
#     HCCL 域,规避同域同卡约束);.64 引擎 HCCL 广播(world=1+12,与异步 PD 同构)。
#   * FEAT_OFFLOAD=1:训推双侧 offload(训练窗口全部引擎 sleep;.56 引擎靠 needs_offload
#     命中,.64 引擎不重叠不 sleep、常驻 —— 但网关被 prepare_policy_update 暂停,语义仍是同步)。
#
# 前置(.64 手工):
#   1) polar 推理端点 → http://80.48.5.56:8011;
#   2) polar 卡池 "0-15" 改为 "0-3"(或按实际剩余);
#   3) .64 先跑 start_pd_worker.sh 加入 Ray(同 pd 流程);老的分离部署进程先清干净。
#
# 内存账(61G/卡,按引擎分叉:共卡 VLLM_GPU_MEM_UTIL=0.70,专用 DEDICATED=0.85):
#   共卡 .56(引擎 + trainer 同卡,run 203413 实测):
#     rollout 窗口 = 引擎(util×61 ≈ 42.7G:权重 35.3 + KV 7.4)+ trainer 常驻 ~9.8G
#       + 杂项(图/驱动/HCCL)~8G ≈ 60.2G,余量仅 ~0.8G —— 0.70 已是天花板。
#     ⚠ 全局 0.85 物理不可能:引擎自身 51.85G + trainer 9.8G > 60.95G 总量,
#       KV 唤醒必 aclrtMallocPhysical OOM(20260824-203413 实锤,8 台共卡引擎全灭)。
#   专用 .64(整卡独占,无 trainer):0.85 对齐 PD 基线 → KV 16.5G/引擎,
#     256K 并发 6.2x,长上下文 session 主要由这 6 台承接。
#   权重同步:共卡引擎醒(权重壳 35.3G,KV 不驻留)+ trainer ~10G+瞬时 ≈ 47G ✓;
#     同步窗口安全由「KV 移出同步窗口」结构保证(权重壳→同步→再醒 KV,0aab9283)。
#   host 预算:level=2 驻留归零,训练 host ~1010G + plasma 200G + 基线 ~150G ≈ 1.35T / 2T,
#     远低于 Ray 阈值,RAY_memory_usage_threshold=0.99 可回默认。
# 模型名契约(20260824 no_completions 根因):
#   引擎以 --served-model-name 同时 serve「真实路径 + VLLM_SERVED_MODEL_NAME 别名」,
#   polar profile 的模型名固定写 /home/docker/Qwen3.6-35B-A3B,换模型目录
#   (bare/_fused/-bf16 变体)不用动 polar;两边错开 = 全部请求 404 = no_completions。
#   混合布局会自动释放 trainer 的 param+grad flat buffer；同步直接读取 actor 已有的
#   pinned-CPU 权重副本，不需要设置 VIME_OFFLOAD_PARAM_BUFFER，也不会额外复制一份模型。
# ─────────────────────────────────────────────────────────────────────────────
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
CURRENT_IP=80.48.5.56  MASTER_ADDR=80.48.5.56  NNODES=2  NPUS_PER_NODE=16  SOCKET_IFNAME=ens1f3 \
ACTOR_NUM_GPUS_PER_NODE=16 \
TRAIN_ENTRY=train.py \
FEAT_OFFLOAD=1 \
RESOURCE_LAYOUT=/workspace/vime/scripts/resource_layout.hybrid56cola64infer.yaml \
ROLLOUT_NODE_IP=80.48.5.56 \
ROLLOUT_NUM_GPUS=24 \
ROLLOUT_NUM_GPUS_PER_ENGINE=2 \
FEAT_PD_DISAGG=0 \
VLLM_SERVED_MODEL_NAME=/home/docker/Qwen3.6-35B-A3B \
VLLM_GPU_MEM_UTIL=0.75 \
VLLM_GPU_MEM_UTIL_DEDICATED=0.85 \
MAX_TOKENS_PER_GPU=32768 \
SEQ_LENGTH=262144 \
ROLLOUT_MAX_CONTEXT_LEN=262144 \
VLLM_MAX_MODEL_LEN=262144 \
VIME_MEM_PROBE=1 \
RAY_memory_usage_threshold=0.95 \
no_proxy=127.0.0.1,localhost,80.48.5.56,80.48.5.64,80.5.25.140,.huawei.com,local,.local \
NO_PROXY=127.0.0.1,localhost,80.48.5.56,80.48.5.64,80.5.25.140,.huawei.com,local,.local \
TP=1 \
PP=2 \
POLAR_TRAJECTORY_PG_FLOOR=0.05 \
POLAR_ROLLOUT_URL=http://80.48.5.64:8180 \
POLAR_DRAIN_SESSIONS=0 \
POLAR_MAX_OFF_POLICY_STEPS=0 \
VLLM_ROUTER_PORT=8011 \
FEAT_TRAIN_EXPANDABLE=1 \
VIME_EMPTY_CACHE_PER_STEP=1 \
POLAR_MAX_ACTIVE_SESSIONS=64 \
TRANSFORMERS_VERBOSITY=error \
HCCL_INTER_HCCS_DISABLE=false \
HCCL_INTRA_ROCE_ENABLE=1 \
HCCL_INTRA_PCIE_ENABLE=0 \
HCCL_BUFFSIZE=512 \
HCCL_HOST_SOCKET_PORT_RANGE=60000-60255 \
HCCL_NPU_SOCKET_PORT_RANGE=61000-61255 \
ROLLOUT_BATCH_SIZE=8  N_SAMPLES_PER_PROMPT=8  GLOBAL_BATCH_SIZE=64  NUM_ROLLOUT=100 \
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
