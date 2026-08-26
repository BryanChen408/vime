#!/bin/bash
# DAPO：Qwen3.5-35B-A3B MoE + polar operator 环境，多节点 NPU。
# 环境/拓扑/模型参数全部来自 experiments/common/run_base.sh，这里只放算法相关的。
#
# 当前 Polar rollout 已接入 dynamic sampling；仍未实现的是 DAPO 原版
# overlong reward shaping（Polar 现有的是基于截断事件的固定扣分）。

ALGORITHM_TAG=dapo

# GRPO 系用 group 内 baseline，n 必须 > 1（n=1 时 reward - mean(reward) 恒为 0）。
ROLLOUT_BATCH_SIZE=4
N_SAMPLES_PER_PROMPT=8

# session 数按 n 放大，否则 8 个样本的组要排队，async level 形同虚设。
ROLLOUT_MAX_ACTIVE_SESSIONS=32

ALGO_ARGS=(
   --advantage-estimator grpo
   # clip-higher：上界放宽到 0.28，给低概率 token 更多上升空间
   --eps-clip 0.2
   --eps-clip-high 0.28
   # token-level loss：长回答按 token 数加权，不再每条序列等权
   --calculate-per-token-loss
   # Dr.GRPO：去掉 group std 归一化，消除长度/难度偏置（arxiv 2503.20783）
   --disable-grpo-std-normalization
   # dynamic sampling：丢弃组内 reward 全相同的 prompt，并继续补采样。
   --dynamic-sampling-filter-path \
   vime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --kl-coef 0.00
   --kl-loss-coef 0.00
   --entropy-coef 0.00
   # 异步 rollout 下数据最多旧 6 步（max_async_level + update_weights_interval），
   # 用 TIS 修训推不一致。
   --use-tis
)

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/../common/run_base.sh"
