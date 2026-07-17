# 单机 PPO 功能测试启动 —— 仿 scripts/start.sh，改用 PPO run 脚本 + PPO 专属拓扑/BS。
# 前置：宿主机先起 polar（profile.vime.yaml，推理端点指向 vime vLLM 的 :8001）。
#
# 与 GRPO start.sh 的差异（PPO 特有）：
#   - run-qwen36-35b-polar-ppo.sh（--advantage-estimator ppo → use_critic=True → offload_train 强制 True，
#     actor+critic 共卡时分复用；offload 用 NPUWeightOffloader（Python 层，已在 HEAD，非 torch_memory_saver）。
#   - 单机：NNODES=1；卡位 polar 0-3 / rollout 4-7 / actor+critic 8-15。
#   - RESOURCE_LAYOUT=resource_layout_actor_domain2.yaml —— route-b：actor 钉 8-15 同 HCCS 域，避 EI0013 跨域。
#   - GLOBAL_BATCH_SIZE=4 必须 = ROLLOUT_BATCH_SIZE×N_SAMPLES（2×2）；否则 train_iters=0 触发
#     optimizer_param_scheduler.py:73 `assert lr_decay_steps>0`。
#
# FEAT 与 start.sh（工作正常 GRPO）保持一致：FLASHCOMM1=0、async 不开、LB_PROXY=1、
#   PREFIX_CACHE/MULTISTREAM/STATIC_KERNEL/HCCL_AIV=1、TRAIN_EXPANDABLE=1、EMPTY_CACHE_PER_STEP=1。
#
# ⚠️ 已知坑（2026-07-16 smoke7b 实测）：rollout 生成的 trace 长 22K–40K tokens，会被 adapter 以
#   `total_len > max_tokens=2048` 全部丢弃 → "no usable trace, 发占位符" → 训练没数据、永远 No progress。
#   引擎/pause/resume 都正常，卡的是这个长度过滤。跑之前确认 polar 侧任务的 max_tokens / 生成长度，
#   或把 rollout 生成压短到 < 该 max_tokens。查 vime_bridge/adapter.py 的 max_tokens 来源。
VIME_OFFLOAD_PARAM_BUFFER=1 \
CURRENT_IP=80.48.5.88  MASTER_ADDR=80.48.5.88  NNODES=1  NPUS_PER_NODE=12  SOCKET_IFNAME=ens1f3 \
RESOURCE_LAYOUT=/workspace/vime/scripts/resource_layout_actor_domain2.yaml \
MAX_TOKENS_PER_GPU=32768 \
VIME_MEM_PROBE=1 \
VIME_EMPTY_CACHE_PER_STEP=1 \
FEAT_TRAIN_EXPANDABLE=1 \
ROLLOUT_BATCH_SIZE=2  N_SAMPLES_PER_PROMPT=2  GLOBAL_BATCH_SIZE=4  NUM_ROLLOUT=200 \
FEAT_DP_EXTERNAL_LB=0 FEAT_BALANCE_SCHED=0 FEAT_LB_PROXY=1 FEAT_CROSS_DP_EP=0 \
FEAT_ROLLOUT_EP=0 FEAT_FLASHCOMM1=0 FEAT_PREFIX_CACHE=1 FEAT_MULTISTREAM_SHARED_EXPERT=1 FEAT_STATIC_KERNEL=1 FEAT_HCCL_AIV=1 \
bash scripts/run-qwen36-35b-polar-ppo.sh
