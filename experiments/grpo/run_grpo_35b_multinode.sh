#!/bin/bash
# GRPO：Qwen3.5-35B-A3B MoE + polar operator 环境，多节点 NPU。
# 环境/拓扑/模型参数全部来自 experiments/common/run_base.sh，这里只放算法相关的。

ALGORITHM_TAG=grpo

# GRPO 系用 group 内 baseline，n 必须 > 1（n=1 时 reward - mean(reward) 恒为 0）。
ROLLOUT_BATCH_SIZE=4
N_SAMPLES_PER_PROMPT=8

# session 数按 n 放大，否则 8 个样本的组要排队，async level 形同虚设。
ROLLOUT_MAX_ACTIVE_SESSIONS=64

ALGO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.001
   --eps-clip 0.2
   # 异步 rollout 下数据最多旧 6 步（max_async_level + update_weights_interval），
   # 用 TIS 修训推不一致。
   --use-tis
)

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/../common/run_base.sh"
