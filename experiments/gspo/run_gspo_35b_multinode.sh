#!/bin/bash
# GSPO：Qwen3.5-35B-A3B MoE + polar operator 环境，多节点 NPU。
# 环境/拓扑/模型参数全部来自 experiments/common/run_base.sh，这里只放算法相关的。

ALGORITHM_TAG=gspo

# GSPO 用 group 内 baseline，n 必须 > 1（n=1 时 reward - mean(reward) 恒为 0，没有梯度）。
# 批量沿用 examples/coding_agent_rl 同规模模型的配置：8 prompt * 8 sample = gbs 64。
ROLLOUT_BATCH_SIZE=8
N_SAMPLES_PER_PROMPT=8

# session 数按 n 放大，否则 8 个样本的组要排队，async level 形同虚设。
ROLLOUT_MAX_ACTIVE_SESSIONS=32

ALGO_ARGS=(
   --advantage-estimator gspo
   # GSPO 的 ratio 是序列级的（loss.py:909 走 compute_gspo_kl，对整条序列的 mean KL
   # 取指数），量级远小于 token-level，所以 clip 比 PPO 的 0.2 小三个数量级。
   # 取值同 scripts/run-glm4.7-355B-A32B.sh 和 examples/coding_agent_rl。
   --eps-clip 1e-4
   --eps-clip-high 2e-4
   --kl-coef 0.00
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   # 异步 rollout 下数据最多旧 6 步（max_async_level + update_weights_interval），
   # 用 TIS 修训推不一致。
   --use-tis
)

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/../common/run_base.sh"
