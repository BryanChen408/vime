#!/usr/bin/env bash
# vime + polar【AscendC / rollout-only 纯推理】一键启动。
# 参照 scripts/start.sh(triton 那份),保留其全部 FEAT/HCCL/vllm 特性;相对它的改动:
#   - ASCEND_RT_VISIBLE_DEVICES: 0-11 → 4-15  (polar 验证独占 0-3;vime 推理用后 12 卡 4-15)
#   - +DEBUG_ROLLOUT_ONLY=1  ROLLOUT_NUM_GPUS=12  (纯推理:actor gpus=0,12 卡 → 3 engine×4)
#   - 去掉 RESOURCE_LAYOUT(那是 actor 训练钉位 actor_domain,rollout-only 无 actor → 走 gpu 计数拓扑)
#   - 调 run-qwen36-35b-polar-ascendc.sh(已内置 数据集/context 262144/--num-gpus-per-node)
# 前置:宿主机先用 profile.t2a.yaml 起 polar(npu_lease 0-3);采集用 deploy 里的 start_telemetry_run.sh。
# 换机:改 CURRENT_IP / MASTER_ADDR / SOCKET_IFNAME(三者本机值)。
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7,8,9,10,11,12,13,14,15 \
CURRENT_IP=80.5.25.119  MASTER_ADDR=80.5.25.119  NNODES=1  NPUS_PER_NODE=12  SOCKET_IFNAME=enp48s3u1u1 \
DEBUG_ROLLOUT_ONLY=1  ROLLOUT_NUM_GPUS=12 \
MAX_TOKENS_PER_GPU=32768  FEAT_TRAIN_EXPANDABLE=1 \
VIME_MEM_PROBE=0  VIME_EMPTY_CACHE_PER_STEP=1 \
ROLLOUT_BATCH_SIZE=2  N_SAMPLES_PER_PROMPT=2  GLOBAL_BATCH_SIZE=4  NUM_ROLLOUT=200 \
FEAT_DP_EXTERNAL_LB=0 FEAT_BALANCE_SCHED=0 FEAT_LB_PROXY=1 FEAT_CROSS_DP_EP=0 \
FEAT_ROLLOUT_EP=0 FEAT_FLASHCOMM1=0 FEAT_PREFIX_CACHE=1 FEAT_MULTISTREAM_SHARED_EXPERT=1 FEAT_STATIC_KERNEL=1 FEAT_HCCL_AIV=1 \
bash scripts/run-qwen36-35b-polar-ascendc.sh
