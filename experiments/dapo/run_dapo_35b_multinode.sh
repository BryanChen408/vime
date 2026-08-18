#!/bin/bash
# DAPO：Qwen3.5-35B-A3B MoE + polar operator 环境，多节点 NPU。
# 环境/拓扑/模型参数全部来自 experiments/common/run_base.sh，这里只放算法相关的。
#
# 注意：DAPO 四件套里的 dynamic sampling 在这套 polar rollout 下用不了。
# --dynamic-sampling-filter-path 只被 vime/rollout/vllm_rollout.py:596 读取，
# vime_bridge 的 polar rollout 不实现重采样，传了会被静默忽略。
# 所以这里是 clip-higher + token-level loss + Dr.GRPO，缺 dynamic sampling。

ALGORITHM_TAG=dapo

# GRPO 系用 group 内 baseline，n 必须 > 1（n=1 时 reward - mean(reward) 恒为 0）。
ROLLOUT_BATCH_SIZE=4
N_SAMPLES_PER_PROMPT=8

# session 数按 n 放大，否则 8 个样本的组要排队，async level 形同虚设。
ROLLOUT_MAX_ACTIVE_SESSIONS=64

ALGO_ARGS=(
   --advantage-estimator grpo
   # clip-higher：上界放宽到 0.28，给低概率 token 更多上升空间
   --eps-clip 0.2
   --eps-clip-high 0.28
   # token-level loss：长回答按 token 数加权，不再每条序列等权
   --calculate-per-token-loss
   # Dr.GRPO：去掉 group std 归一化，消除长度/难度偏置（arxiv 2503.20783）
   --disable-grpo-std-normalization
   --kl-coef 0.00
   --kl-loss-coef 0.00
   --entropy-coef 0.00
   # 异步 rollout 下数据最多旧 6 步（max_async_level + update_weights_interval），
   # 用 TIS 修训推不一致。
   --use-tis
)

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/../common/run_base.sh"
