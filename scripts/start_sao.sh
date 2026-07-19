#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SAO 功能验证(smoke)启动器 —— 类比 start_ppo.sh,但跑 SAO 脚本 + **小 batch / 小 critic warmup**。
# 用途:占卡后验证 SAO 各特性能不能跑通(不是长跑调参)。从 sao-adapt worktree 运行:
#     bash /home/docker/sao_adapt_wt/scripts/start_sao.sh
# 前置:①宿主机 polar 已起(profile.vime.yaml,推理端点指向 :8001);②卡 4-15 空闲(先停占卡的旧 run)。
#
# ⚠️ 功能验证专用小配置(可 env 覆盖):
#   - 小 batch:ROLLOUT_BATCH_SIZE=2 · N_SAMPLES=2 · GLOBAL_BATCH_SIZE=4
#     (硬约束:GLOBAL_BATCH_SIZE 必须 = ROLLOUT_BATCH_SIZE×N_SAMPLES,否则 train_iters=0 → assert lr_decay_steps>0 崩)
#   - 小 critic warmup:NUM_CRITIC_ONLY_STEPS=2(前 2 步只训 critic、之后 actor 加入 → 快速验"critic-only→joint"两条路径)
#     (SAO 真训用 10;这里小是为了尽快看到 actor 训练。设 0 = actor 从 step0 训)
#   - MAX_TOKENS_PER_GPU=32768(=131072 过滤上限;**不设会退默认 512→过滤 2048→长 trace 全丢、No progress**)
#
# 验证顺序建议(逐个加,坏了好定位):
#   1) 本脚本原样跑 = 路线 A(value warmup + DIS clamp + 非对称 clip + token 归一 + K=2)。先确认这个通。
#   2) 若怀疑 K=2:SAO_CRITIC_UPDATE_STEPS=1 退回标准 1:1,隔离。
#   3) frozen-attn critic:先 SAO_DUMP_CRITIC_PARAMS=/tmp/critic_params.txt 跑一次看参数名 → 确认 regex →
#      再 SAO_FROZEN_ATTN_CRITIC=1 开。
#   4) route-B(skip-obs + 长度自适应 λ):SAO_ROUTE_B=1(注意 λ<1 才有效)。
#   5) DIS 忠实掩码:SAO_DIS_MASK=1。critic 独立 LR:SAO_CRITIC_CONFIG=scripts/sao_critic_config.yaml。
# ─────────────────────────────────────────────────────────────────────────────
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

VIME_OFFLOAD_PARAM_BUFFER=1 \
CURRENT_IP="${CURRENT_IP:-80.48.5.88}"  MASTER_ADDR="${MASTER_ADDR:-80.48.5.88}"  NNODES=1  NPUS_PER_NODE=12  SOCKET_IFNAME="${SOCKET_IFNAME:-ens1f3}" \
RESOURCE_LAYOUT="${RESOURCE_LAYOUT:-${SCRIPT_DIR}/resource_layout_actor_domain2.yaml}" \
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}" \
VIME_MEM_PROBE=1  VIME_EMPTY_CACHE_PER_STEP=1  FEAT_TRAIN_EXPANDABLE=1 \
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"  N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"  NUM_ROLLOUT="${NUM_ROLLOUT:-200}" \
NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-2}" \
SAO_CRITIC_UPDATE_STEPS="${SAO_CRITIC_UPDATE_STEPS:-2}" \
FEAT_DP_EXTERNAL_LB=0  FEAT_BALANCE_SCHED=0  FEAT_LB_PROXY=1  FEAT_CROSS_DP_EP=0 \
FEAT_ROLLOUT_EP=0  FEAT_FLASHCOMM1=0  FEAT_PREFIX_CACHE=1  FEAT_MULTISTREAM_SHARED_EXPERT=1  FEAT_STATIC_KERNEL=1  FEAT_HCCL_AIV=1 \
bash "${SCRIPT_DIR}/run-qwen36-35b-polar-sao.sh"
