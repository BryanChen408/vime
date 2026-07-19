# SAO 适配 · 交接 & 启动说明(分支 `sao-adapt`)

- 日期:2026-07-19
- 分支/worktree:**`sao-adapt`** @ `/home/docker/sao_adapt_wt`,基于 `feature/lb-proxy`(含全部 PPO 落地:F-PPO-1/A1/F-PPO-3)。
- 与运行中内容**完全隔离**:`/workspace/vime`(live 树)零改动。
- 设计/决策全量见 `/home/docker/vime-polar/docs/design/ppo_adaptation_findings.md` + `ppo_adaptation_review_tables.md`。

---

## 0. 一句话
SAO 论文可迁移特性已**全部实现**(路线 A 默认开 + 路线 B/C8/DIS-mask 默认关待占卡验),card-free 测试闭环(10 单测全过)。**下一步 = 占卡功能验证**,已备好 smoke 启动器。

## 1. 已实现特性 & 默认值
| 特性 | 开关 | smoke 默认 | 说明 |
|---|---|---|---|
| value 预训练 warmup [C9] | `NUM_CRITIC_ONLY_STEPS` | **2**(小,验功能用;真训 10) | 前 N 步只训 critic |
| off-policy DIS [C11] | `--use-tis`(默认开) | clamp | `SAO_DIS_MASK=1` → icepop 忠实掩码 |
| 非对称 clip [C11] | `EPS_CLIP`/`EPS_CLIP_HIGH` | 0.8/3.0 | code 任务那套 |
| token-|M| 归一 [F-PPO-8] | `SAO_TOKEN_LEVEL_LOSS` | 1(开) | `--calculate-per-token-loss` |
| **K=2 快速价值更新** [C12a] | `SAO_CRITIC_UPDATE_STEPS` | **2** | 每 rollout 训 K 次 critic |
| Frozen-Attn critic [C8] | `SAO_FROZEN_ATTN_CRITIC` | 0(关) | 冻结已 role 化;regex 见 §4 |
| critic 独立 LR [Q-12] | `SAO_CRITIC_CONFIG` | 空(关) | `=scripts/sao_critic_config.yaml`(5e-6) |
| route-B:skip-obs GAE + 长度自适应 λ [C10/C12b] | `SAO_ROUTE_B` | 0(关) | λ<1 才有效;须成对 |

## 2. 启动(功能验证 / smoke)
**前置**:①宿主机 polar 已起(profile.vime.yaml,推理端点 :8001);②卡 4-15 空闲(先停占卡的旧 run,注意别碰 polar 卡 0-3)。

```bash
bash /home/docker/sao_adapt_wt/scripts/start_sao.sh
```

`start_sao.sh` 已固化**小 batch + 小 critic warmup**(你的验功能要求):
- 小 batch:`ROLLOUT_BATCH_SIZE=2 · N_SAMPLES=2 · GLOBAL_BATCH_SIZE=4`
- 小 warmup:`NUM_CRITIC_ONLY_STEPS=2`
- `MAX_TOKENS_PER_GPU=32768`、RESOURCE_LAYOUT=worktree 内 domain2 yaml、FEAT 开关对齐 start_ppo.sh。
均可 env 覆盖,如 `NUM_CRITIC_ONLY_STEPS=0 bash scripts/start_sao.sh`。

## 3. 硬约束 / 坑(务必看)
- **`GLOBAL_BATCH_SIZE` 必须 = `ROLLOUT_BATCH_SIZE × N_SAMPLES`**(2×2=4),否则 train_iters=0 → `assert lr_decay_steps>0` 崩。
- **`MAX_TOKENS_PER_GPU` 必须设 32768**:不设退默认 512 → adapter 以 512×cp_size=2048 过滤 → polar 长 trace(22–40K)全丢 → "No progress"。(过滤逻辑 `vime_bridge/rollout.py:_resolve_max_tokens`;×cp_size 是对的,真因是没设 mtpg。)
- `--kl-coef` 必须 0(critic 不算 ref,>0 崩)——SAO 脚本已固定。
- `ASCEND_RT_VISIBLE_DEVICES` 升序;polar 卡 0-3 绝不碰;崩后 relaunch 前清残留 vLLM。

## 4. 逐步验证阶梯(坏了好定位)
1. **原样跑** = 路线 A(warmup + DIS clamp + 非对称 clip + token 归一 + K=2)。先确认通、看 `value_explained_var`(EV,判 critic 学得如何)。
2. 疑 K=2 → `SAO_CRITIC_UPDATE_STEPS=1` 退标准 1:1 隔离。
3. **frozen-attn**:先 `SAO_DUMP_CRITIC_PARAMS=/tmp/critic_params.txt` 跑一次 dump critic 参数名,核对 regex(见下),再 `SAO_FROZEN_ATTN_CRITIC=1`。
   - 静态已定:GDN=`self_attention.linear_attn.*`、full-attn=`self_attention.*`、experts=`mlp.experts.*`、value head=`output_layer.*`。
   - 默认训 `mlp.experts output_layer`(strict);宽版含 shared_expert/router:`SAO_CRITIC_TRAIN_PATTERNS='mlp\. output_layer'`。
4. **route-B**:`SAO_ROUTE_B=1`(skip-obs + α=1.5)。⚠️ λ=1 时 skip-obs 是 no-op,必须成对。
5. **DIS 忠实掩码**:`SAO_DIS_MASK=1`。**critic 独立 LR**:`SAO_CRITIC_CONFIG=scripts/sao_critic_config.yaml`。

## 5. 离线 value 预训练(可选,省真占卡 warmup)
`--load-debug-rollout-data <dir>/{rollout_id}.pt` + `--num-critic-only-steps N` → 自动跳 vLLM、只训 critic。
需先 dump:`--save-debug-rollout-data` 存 num_rollout 个文件(无 fallback,每个文件样本数须够 global-batch)。

## 6. card-free 测试(占卡前可跑)
```bash
cd /home/docker/sao_adapt_wt
PYTHONPATH=. python tests/test_sao_gae.py                 # GAE 5 + EV 1 + K2 1
PYTHONPATH=.:/workspace/Megatron-LM:/workspace/vllm:/workspace/vllm-ascend python tests/test_sao_freeze_role.py  # freeze 3
```
覆盖:mask 全1=标准 GAE(backward-compat)· skip-obs 手算 · 终止 reward 穿 masked 尾回流 · EV · K 计数 · critic 冻结不误伤 actor。

## 7. 残留 / 待占卡验(非阻塞路线 A)
- route-B 在 **CP>1** 下的 mask all-gather 路径未占卡数值验(逻辑与 values 对称、倾向正确;默认关)。
- **F-PPO-4 已确认正确**(CP zigzag 下 `k[-1]`=真末 token),仅 local-tensor 排序可占卡最终复核。
- `chunked_gae` mask-aware 有意延后(vanilla 正确、仅长序列略慢)。
- frozen-attn 的 regex 最终值 = dump 后确认(§4)。

## 8. 维护
- SAO 脚本 = 从 `run-qwen36-35b-polar-ppo.sh` fork,PPO 脚本一更新就要从当前 PPO 重新 fork(只差 header + RUN_ID + PPO_ARGS 块)。
- 提交历史:`fcd6a179`(脚本)→ `5b075533`(C8+K2)→ `1b6cce7d`(选项1加固)→ `2e1d0017`(A1-4+C1)→ `e2c55829`(CP闭环+EV+测试)。
