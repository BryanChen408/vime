# PPO 适配 · Landing 清单(无卡已就绪 · 待占卡治理脏树时应用)

日期 2026-07-11。所有件均已 py_compile / bash -n + 对活树 diff/apply 验证。
目标树 = `/workspace/vime`(分支 npu,脏)。应用前先 checkpoint 团队 WIP(用户在场)。

## 应用顺序
1. **F-PPO-1**(chunk-lm-head 不旁路 critic value head)
   - `cd /workspace/vime && patch -p1 < /home/docker/ppo_adapt_dev/F-PPO-1_chunked_lm_head_gate.patch`
     (活树的 `chunked_lm_head_patch.py` 当前**无 gate**——我早先撤回了,见 memory)
   - 或从分支 cherry-pick `67dae52c`
2. **F-PPO-2 步骤1**(Lock `num_cpus=0`,修 dist-init 7/8)
   - `patch -p1 < .../F-PPO-2_lock_num_cpus.patch`  或 cherry-pick `05c40ea7`
3. **A1+A3**(resource_layout critic placement + critic save 分目录)
   - `patch -p0 vime/ray/placement_group.py < .../A1-A3_placement_critic.patch`
   - ⚠️ **只能 patch**:活树的 resource_layout 未提交、分支(HEAD 基)没有此函数,无法 cherry-pick。
4. **A2**(PPO 启动脚本)
   - `cp .../run-qwen36-35b-polar-ppo.sh /workspace/vime/scripts/`  或 cherry-pick `d5254cee`

## 应用后自检(无卡)
- `python -m py_compile` 三改动文件;`bash -n scripts/run-qwen36-35b-polar-ppo.sh`。
- `git diff` 只含上述改动 + 团队 WIP,无意外。

## 占卡验证(详见 `vime-polar/docs/design/ppo_adaptation_findings.md` §6.B)
- **Run-1**:dist-init **8/8**(验 F-PPO-2)。拓扑二选一:
  - 默认位置路径(不设 RESOURCE_LAYOUT)→ actor+critic=4-11 跨域、rollout=12-15,EI0013 靠 HCCL env 兜(简单,先隔离验 PPO 逻辑)。
  - 设 RESOURCE_LAYOUT(actor 8-15 yaml)→ 单域免 EI0013(依赖 A1+F-PPO-2 已落)。
- **Run-2**:PPO 首步 e2e(F-PPO-1/F-PPO-3 offload + value head reinit + 回归闸 policy&value 双侧 + CP/F-PPO-4)。
- **Run-3**:多步 soak(ratio×TIS、async×value、num-critic-only-steps warmup)。

## 分支 / 备份
- `ppo-adapt` @ worktree `/home/docker/ppo_adapt_wt`:`67dae52c`(F-PPO-1)/`05c40ea7`(F-PPO-2)/`d5254cee`(A2 脚本)。
- A1+A3 **不在分支**(活树 resource_layout 基础差异)→ 仅 patch + `placement_group.py.ppo`(完整改后文件备查)。
- dev 副本区:`/home/docker/ppo_adapt_dev/`(patches + 脚本 + 改后文件 + test_gate.py)。

## 未做(非无卡 / 可选)
- **A5 landing 本身**(此清单的应用)= 占卡+治理脏树步,用户在场。
- **A6**(可选):`--megatron-config-path` role=critic 配 critic 独立 LR/save(替代 A3 的 `_critic` 目录)。
- **C 类调参**(warmup/gamma/lambda/critic-LR/normalize/clip)= 占卡试出。
