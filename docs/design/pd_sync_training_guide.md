# PD 分离同步训推 开启指南(Qwen3.6-35B-A3B / 12卡 2P4D)

最后更新:2026-09-14(本形态已 E2E 验证:两轮 rollout + 两次权重更新 + 两次 sleep/wake 全过)

## 1. 形态与前提

- **形态**:同步训推 + PD 分离。actor 训练 16 卡(0-15),rollout 12 卡(4-15),prefill 2 引擎(TP2×2,卡 4-7)+ decode 4 引擎(TP2×4,卡 8-15);KV 走 Mooncake/ADXL(RoCE)。
- **代码**:
  - vime:a3-pd 分支(含 PD 配置下发修复 + post-wake 探针 + 启动脚本);
  - 引擎:`vllm-ascend-023` 的 `pr-11976` 分支,须含根因 A(gdn split import)与根因 B(sleep 注销/wake 重注册)两个修复——patch 存档:`docs/design/patches/vllm_ascend_023_pd_sync_20260913.patch`;
  - Polar:sync-rollout 系(不带 epoch-enforcement 的版本),如 `polar-sync-rollout` worktree。
- **模型 checkpoint**:`/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16`(HF)+ 对应 `..._torch_dist`(ref)。

## 2. 需要的 config(已随仓库提供,通常不用改)

| 文件 | 作用 |
|---|---|
| `scripts/vllm_qwen36_35b_polar_dual140_pd_12card.yaml` | PD 引擎拓扑与 per-group 覆盖(prefill/decode 各自的 additional_config、cudagraph、recompute_scheduler_enable 等)。**D 组必须有 `recompute_scheduler_enable: true`,P 组必须 `enable_flashcomm1: true`** |
| `scripts/start_sync_pd_single52.sh` | 启动入口(资源布局/引擎数/日志) |
| `scripts/resource_layout.single52_homo_colocate.yaml` | 单机 16 卡布局 |

## 3. 必需的环境变量

```bash
export NO_PROXY="127.0.0.1,localhost,80.48.5.52,.huawei.com"   # 公司代理会劫持 polar 长连接,必须直连
export no_proxy="$NO_PROXY"
export POLAR_ROLLOUT_URL="http://80.48.5.52:8080"              # 指向你实际启动的 polar rollout 端口
export FEAT_FLASHCOMM1=1                                       # 引擎开 EP,满足 P 组 flashcomm1 断言
export VIME_PD_KV_REREGISTER_ON_WAKE=1                         # 根因B修复:sleep 注销 / wake 重注册
export ASCEND_USE_SHORT_CONNECTION=1                           # ADXL 每次传输后断链(注销的前提)
export VIME_PD_POST_WAKE_PROBE=1                               # (建议)每次 wake 后跑 P/D 一致性哨兵
```

互斥/注意:
- **不要**再设 `VIME_KV_SLEEP_PERSISTENT=1`(旧 workaround,与本修复互斥,同开即 assert);
- polar URL 指错(比如指到残留实例)的典型症状:health 全 200 但 rollout 永远不派发;
- `ASCEND_USE_SHORT_CONNECTION=1` 带来每传输 ~15ms 建链开销(实测 KV 传输 ~35ms → ~50ms);要去掉就做 Mooncake `disconnect_all` 补丁(见设计文档 §方案 P)。

## 4. 启动示例

```bash
cd /workspace/vime
NO_PROXY="127.0.0.1,localhost,80.48.5.52,.huawei.com" no_proxy="$NO_PROXY" \
POLAR_ROLLOUT_URL="http://80.48.5.52:8080" \
FEAT_FLASHCOMM1=1 VIME_PD_KV_REREGISTER_ON_WAKE=1 ASCEND_USE_SHORT_CONNECTION=1 \
VIME_PD_POST_WAKE_PROBE=1 \
NUM_ROLLOUT=2 RUN_ID=my_pd_run \
bash scripts/start_sync_pd_single52.sh
```

## 5. 验证清单(日志判定点)

1. startup:`PD-POST-WAKE-PROBE pass cycle=startup ... pairs=8/8`;
2. 每个权重边界:引擎 `Sleep mode (level=2)` → `unregistered 10 regions` → wake → `re-registered 10 regions` → `PD-POST-WAKE-PROBE pass cycle=rollout N`;
3. rollout 输出无乱码/无跨任务串线;显存不常驻 8G workaround。

## 6. 快速诊断模式(不起训练)

```bash
VIME_PD_LIFECYCLE_DIAG_ONLY=1 VIME_PD_SLEEP_WAKE_RELOAD_DIAG=1 VIME_PD_KV_XFER_HASH=1 \
  + 上述同样 env,bash scripts/start_sync_pd_single52.sh
```
只跑引擎生命周期自检(基线 direct-vs-PD + sleep/wake+reload 后对照 + 40 层 KV hash),~15 分钟。

## 7. 已知边界

- 当前 RoCE 路径;HCCS 全联通环境(A3 超节点)不需要 `VIME_PD_KV_REREGISTER_ON_WAKE`(探针实证免疫),只需 `HCCL_INTRA_ROCE_ENABLE` 不设 + 短连接;
- rollout-only(不训练)不需要本修复(不 sleep);
- 非 PD 路径不受任何影响(所有改动均有门控/仅 PD 生效)。

更完整的根因与设计:`docs/design/pd_sleep_wake_kv_fix_design_20260912.md`。
