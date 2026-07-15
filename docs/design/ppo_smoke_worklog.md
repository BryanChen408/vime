# PPO 单机功能测试 · 工作日志(自主 /loop)

> 目标:单机小 batch 打通 vime PPO(actor+critic 共卡 offload 时分)训练功能。
> 起始:2026-07-15。执行模式:自主 /loop,遇 bug 最小改动修复、全程记录、语义 commit。

## 用户决策(2026-07-15 离开前拍板)
1. **Commit 范围**:整棵脏树按语义全提交(不只本次改动)。
2. **Polar**:若不在 → hostctl 自动重启(restart_polar_stack,保 :8001,绝不 kill)。
3. **修复边界**:允许必要时做较大改动(在日志记录清楚的前提下)。
4. **验收标准**:首步 PPO e2e 通过(F-PPO-1/3、critic value [T,1]、policy/value 双 loss 有限、TIS≈1)+ ~3 步跨权重更新稳定 → 停并汇报。

## 就绪前置(本 session 已完成)
- A2:PPO 脚本 `scripts/run-qwen36-35b-polar-ppo.sh` 从当前 GRPO 重 fork,diff 只剩 4 处 delta(header/RUN_ID/PPO_ARGS/调用处),含 OOM 修复 FEAT_TRAIN_EXPANDABLE。bash -n ✓。
- F-PPO-1 gate:`chunked_lm_head_patch.py` 补 value-head 例外(shape[0]==1 不旁路),py_compile ✓。
- 机制确认:critic/actor 共卡(同 PG)+ offload 时分(train 进 onload/出 offload)+ value 依赖串行化;每步 HCCL churn(destroy/reload PG ×3 轮 + rollout 组断/连),F-PPO-3 最高风险 regime。

## 测试计划
- **Run-A**(NUM_ROLLOUT=1,单步破冰):position 路径(不设 RESOURCE_LAYOUT),小 batch(RB=2/N=2/GBS=4),QWEN36_CHUNK_LMHEAD=1,OOM 三件套开。验:offload/wake+HCCL churn 穿过不崩/不卡、critic value [T,1]、双 loss 有限、TIS≈1。
- **Run-B**(NUM_ROLLOUT=3~4):跨权重更新稳定,无 OOM、无 HCCL 累积失败。
- 命令基线见下方运行记录。

## 运行记录
<!-- 按时间追加:每次动作、观察、bug、修复、commit -->

### [启动] 2026-07-15 — 迭代 1
- 建本日志。摸 git 状态 + polar 健康。
