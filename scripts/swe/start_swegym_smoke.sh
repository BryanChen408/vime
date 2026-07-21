#!/bin/bash
# ═══ 1-task SWE-Gym codex SMOKE — Phase 7 step 2(Risk #1:codex↔vllm-ascend↔swebench 端到端)═══
#   最小闭环:1 题(getmoto__moto-4950,已 gold-check)× 2 samples × 1 rollout;卡 4-9(rollout 2 / actor 4 TP=4)。
#   目的看四件事:
#     ① completions≠0 —— codex 通(/responses wire-api + hermes 解到 <tool_call>);全 0 = tool turn 坏。
#     ② swebench_harness 对 codex 产出的 patch 判分(docker 后端 evaluator 跑通)。
#     ③ reward 回灌 vime(SessionResult→sample.reward["score"]);train/tis≈1 = token-faith。
#     ④ 1 步训练不崩(dynamic batch + 权重同步)。
#   非学习:n=2 + binary reward,组内可能 0 advantage,正常 —— 只验管线通不通。
#
#   前置(宿主机):
#     1) SWE polar 起:POLAR_PROFILE=deploy/ascend_operator/profile.swe-8b.yaml bash \
#          /home/docker/cannbot_debug/ProRL-Agent-Server/deploy/ascend_operator/restart_polar_host.sh
#     2) Qwen3-8B 权重 + torch_dist 就绪;codex CLI 在 ${AGENT_CLI_DIR};swegym 包在 polar venv;64 镜像已 load。
#   跑:bash scripts/swe/start_swegym_smoke.sh
#   (调:NUM_ROLLOUT=2 顺带验一次权重同步;FEAT_WANDB=1 + WANDB_KEY 看曲线。)
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ── 1-task 数据集(确定性;就是 gold-check 那题,grading 已知好 → 隔离 codex 侧)──
export SWEGYM_JSONL=${SWEGYM_JSONL:-/home/docker/datasets/swegym/swegym_smoke_1.jsonl}

# ── 卡位:保留已验 actor 训练档(4 卡 TP=4,防 OOM)+ 最小 rollout(2 卡 1 引擎)= 卡 4-9 ──
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7,8,9}
export NPUS_PER_NODE=${NPUS_PER_NODE:-6}
export ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-2}
export ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}
export ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-4}
export TP=${TP:-4}

# ── 最小规模:1 题 × 2 samples × 1 rollout ──
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-1}
export N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-2}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2}
export NUM_ROLLOUT=${NUM_ROLLOUT:-1}                 # =2 顺带验一次权重同步
export POLAR_MAX_ACTIVE_SESSIONS=${POLAR_MAX_ACTIVE_SESSIONS:-2}
export POLAR_MAX_ASYNC_LEVEL=${POLAR_MAX_ASYNC_LEVEL:-1}
export FEAT_WANDB=${FEAT_WANDB:-0}                   # smoke 少动件

# SEQ_LENGTH / MAX_TOKENS_PER_GPU / response/context 保留 start_swegym_8b 的 8B-NPU 已验值(不覆盖)。
exec bash "${SCRIPT_DIR}/start_swegym_8b.sh"
