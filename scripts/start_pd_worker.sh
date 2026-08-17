ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
CURRENT_IP=80.48.5.56  MASTER_ADDR=80.48.5.56  NNODES=2  NPUS_PER_NODE=16  SOCKET_IFNAME=enp48s3u1u1 \
ACTOR_NUM_GPUS_PER_NODE=16 \
ROLLOUT_NODE_IP=80.48.5.64 \
ROLLOUT_NUM_GPUS=12 \
ROLLOUT_NUM_GPUS_PER_ENGINE=2 \
FEAT_PD_DISAGG=1 \
VLLM_PD_CONFIG=/workspace/vime/scripts/vllm_qwen36_35b_polar_dual140_pd_12card.yaml \
RESOURCE_LAYOUT=/workspace/vime/scripts/resource_layout.dual56train57infer_pd.yaml \
MAX_TOKENS_PER_GPU=32768 \
SEQ_LENGTH=262144 \
ROLLOUT_MAX_CONTEXT_LEN=262144 \
VLLM_MAX_MODEL_LEN=262144 \
VIME_MEM_PROBE=1 \
no_proxy=127.0.0.1,localhost,80.48.5.64,80.48.5.56,.huawei.com,local,.local \
NO_PROXY=127.0.0.1,localhost,80.48.5.64,80.48.5.56,.huawei.com,local,.local \
TP=2 \
POLAR_TRAJECTORY_PG_FLOOR=0.05 \
POLAR_ROLLOUT_URL=http://80.48.5.64:8180 \
VLLM_ROUTER_PORT=8011 \
FEAT_TRAIN_EXPANDABLE=1 \
VIME_EMPTY_CACHE_PER_STEP=1 \
POLAR_MAX_ACTIVE_SESSIONS=32 \
TRANSFORMERS_VERBOSITY=error \
HCCL_INTER_HCCS_DISABLE=false \
HCCL_INTRA_ROCE_ENABLE=1 \
HCCL_INTRA_PCIE_ENABLE=0 \
HCCL_BUFFSIZE=512 \
ROLLOUT_BATCH_SIZE=8  N_SAMPLES_PER_PROMPT=4  GLOBAL_BATCH_SIZE=32  NUM_ROLLOUT=100 \
FEAT_DP_EXTERNAL_LB=0 FEAT_BALANCE_SCHED=0 FEAT_LB_PROXY=1 FEAT_CROSS_DP_EP=0 \
FEAT_ROLLOUT_EP=0 FEAT_FLASHCOMM1=0 FEAT_PREFIX_CACHE=1 FEAT_MULTISTREAM_SHARED_EXPERT=1 FEAT_STATIC_KERNEL=0 FEAT_HCCL_AIV=1 \
OPERATOR_DATA_ROOT=/home/docker/polar_can/ProRL-Agent-Server/datasets/op_tasks/op_assets_cudallm_filtered189 \
OPERATOR_TASK_JSONL=/home/docker/polar_can/ProRL-Agent-Server/datasets/op_tasks/op_assets_cudallm_filtered189/operator_tasks.ascendc.jsonl \
PROFILE_TRAIN=0 \
PROFILE_TARGET=train_overall \
PROFILE_STEP_START=1 PROFILE_STEP_END=2 \
TENSORBOARD_DIR=/home/docker/logs/prof/$(date +%Y%m%d-%H%M%S) \
PROFILE_OP=0 \
PROFILE_RANKS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" \
bash scripts/run-qwen36-35b-polar-multi-pd.sh
# ── 【当前:单算子冒烟(3_Add)】验功能正确性,不是训练 ──
#   OPERATOR_TASK_JSONL 只决定"跑哪些算子";OPERATOR_TASKS_DIR(默认 ${OPERATOR_DATA_ROOT}/op_tasks)
#   保持全量目录不变,按 op_name 找 3_Add.py,多余文件无害。
#   ROLLOUT_BATCH_SIZE 必须是 1:数据集只有 1 行 prompt,=2 会要两条不同的 prompt。
#   N_SAMPLES_PER_PROMPT=4 → 同一算子 4 个并行 session,正好铺满 polar 的 4 张卡池(0-3)。
#   验完恢复全量训练:把这 3 处改回 POLAR_MAX_ACTIVE_SESSIONS=24 /
#   ROLLOUT_BATCH_SIZE=2 N_SAMPLES_PER_PROMPT=2 GLOBAL_BATCH_SIZE=4 NUM_ROLLOUT=200,
#   并删掉 OPERATOR_TASK_JSONL 那行(缺省即回落到 ${OPERATOR_DATA_ROOT}/operator_tasks.jsonl 全 31 个)。
#   单算子 jsonl 的生成方式(可复现):
#     python3 ProRL-Agent-Server/deploy/ascend_operator/gen_ascendc_tasks.py \
#         --benchmark-dir /home/docker/NPUKernelBench --level 1 --ops 3_Add \
#         --out /home/docker/datasets/op_tasks/smoke_3add/operator_tasks.jsonl
# ── 备选:跨 DP EP(DP+EP 同开)。冒烟通过 external-LB+Balance 后,把上面 FEAT_CROSS_DP_EP=0 改成 1 即可(其余不动;EP world=dp×tp=16,experts/card=256/16=16)。整行等价形式如下: ──
# FEAT_DP_EXTERNAL_LB=1 FEAT_BALANCE_SCHED=1 FEAT_LB_PROXY=1 FEAT_CROSS_DP_EP=1 FEAT_ROLLOUT_EP=0 FEAT_FLASHCOMM1=0 FEAT_PREFIX_CACHE=1 FEAT_MULTISTREAM_SHARED_EXPERT=1 FEAT_STATIC_KERNEL=1 FEAT_HCCL_AIV=1 bash scripts/run-qwen36-35b-polar-minimal.sh
# ── RL profiling 开关(默认全关;需要时把注释放到续行链里)──
#   PROFILE_TRAIN=1                       训练侧 NPU 采集(torch_npu.profiler Level1,离线解析)
#   PROFILE_TARGET=train_actor            train_overall(默认整步)| train_actor(前反向)| train_log_probs
#   PROFILE_STEP_START=2 PROFILE_STEP_END=4   采第 3-4 个 rollout;落 ${TENSORBOARD_DIR:-outputs/profile}/
#   PROFILE_OP=1                          rollout(vLLM)侧算子采集(140 的 worker 脚本也要设)
#   注意:NPU 一次只能采一个 target;多目标分多次跑。

