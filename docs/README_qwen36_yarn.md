# Qwen3.6 YaRN 开启与配置

本文说明如何在 `dev/yarn` 分支为 Qwen3.6-35B-A3B 同时开启 Megatron 训练侧和 vLLM rollout 侧的 YaRN，以及如何配置 262K 以上的上下文长度。

当前配置仅面向纯文本训练。已经验证的运行组合是：

- VIME：`/workspace/vime-a3pd-integrated`，分支 `dev/yarn`
- vLLM core：`/workspace/vllm-023`
- vLLM Ascend plugin：`/workspace/vllm-ascend-023`
- 训练并行：TP2、CP8、EP8
- YaRN 原始窗口：262144 tokens
- 已验证目标长度：262144、270336 和 300000 tokens

Polar 是已经启动的外部服务。启动 YaRN 训练不需要、也不允许修改 Polar 本地配置。

## 0. 外部代码前置

`dev/yarn` 以 patch bundle 的形式保存本轮依赖的 Megatron-LM 和 vLLM 0.23
改动。部署新环境时，必须先按顺序应用
[Qwen3.6 YaRN external patch bundle](./design/patches/qwen36_yarn_20260915/README.md)，
再执行本文后续命令。该 bundle 同时提供可取得基线到已测运行前置的补丁和独立 YaRN
补丁；只使用原始 Megatron-LM 或原始 vLLM 0.23 不属于已验证组合。

补丁只作用于 Megatron-LM 和 vLLM core。不要在 Polar、MindSpeed 或
`vllm-ascend-023` 中复制同一套 YaRN 数学实现。

## 1. 最短启动路径

在 VIME YaRN 工作区执行，不要在 Polar 目录执行：

```bash
cd /workspace/vime-a3pd-integrated
git branch --show-current
```

分支应为 `dev/yarn`。

先只做配置预检，不启动训练：

```bash
YARN_PREFLIGHT_ONLY=1 \
  bash scripts/start_qwen36_yarn_smoke_single52.sh
```

预检通过后启动 262K 集成训练：

```bash
NUM_ROLLOUT=2 \
  bash scripts/start_qwen36_yarn_smoke_single52.sh
```

该入口会复用已验证的 `start_sync_homo_single52.sh` 拓扑，并且不会修改 Polar。`YARN_SMOKE_CONTEXT_LEN` 被有意固定为 262144；不要用它启动 270K/300K。

跨越原始 262K 边界的 270336-token 单轮测试使用独立入口：

```bash
NUM_ROLLOUT=1 \
  bash scripts/start_qwen36_yarn_stage8_270k_single52.sh
```

## 2. YaRN 参数

正常运行只需要在 source 模型脚本之前设置环境变量。模型脚本会同时生成 Megatron 参数和一份合并后的 vLLM HF override，避免训推两侧分别手写后不一致。

```bash
export FEAT_YARN=1
export YARN_ROPE_THETA=10000000
export YARN_PARTIAL_ROTARY_FACTOR=0.25
export YARN_FACTOR=4.0
export YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS=262144
export YARN_BETA_FAST=32.0
export YARN_BETA_SLOW=1.0
export YARN_MSCALE=1.0
export YARN_MSCALE_ALL_DIM=0.0
export YARN_CORRECTION_RANGE_ROUND_TO_INT=1

source scripts/models/qwen3.5-35B-A3B.sh
```

这些值构成完整的 YaRN 指纹：

| 环境变量 | 当前值 | 含义 |
| --- | ---: | --- |
| `FEAT_YARN` | `1` | 同时开启训练和 rollout YaRN |
| `YARN_ROPE_THETA` | `10000000` | RoPE theta |
| `YARN_PARTIAL_ROTARY_FACTOR` | `0.25` | Qwen3.6 部分旋转维度比例 |
| `YARN_FACTOR` | `4.0` | YaRN 缩放因子 |
| `YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS` | `262144` | 模型应用 YaRN 前的原始上下文长度 |
| `YARN_BETA_FAST` | `32.0` | 高频修正边界参数 |
| `YARN_BETA_SLOW` | `1.0` | 低频修正边界参数 |
| `YARN_MSCALE` | `1.0` | attention scale 参数 |
| `YARN_MSCALE_ALL_DIM` | `0.0` | 全维度 attention scale 参数 |
| `YARN_CORRECTION_RANGE_ROUND_TO_INT` | `1` | Megatron 侧取整，并映射为 vLLM `truncate=true` |

不要通过 MindSpeed 的 `--rope-scaling-type yarn` 另开一套 YaRN。本实现使用 Megatron `YarnRotaryEmbedding`，rollout 侧使用 vLLM `rope_parameters`。

不要再单独传第二个 `--vllm-hf-overrides`。Qwen architecture 和 YaRN rope 参数必须在同一个 JSON 中；模型脚本已经生成在 `QWEN36_VLLM_HF_OVERRIDES`。

## 3. 运行容量

YaRN 指纹和运行容量是两组不同的配置。改变目标长度时，下面四个值必须完全相等：

```bash
export SEQ_LENGTH=<目标长度>
export MAX_POSITION_EMBEDDINGS=<目标长度>
export ROLLOUT_MAX_CONTEXT_LEN=<目标长度>
export VLLM_MAX_MODEL_LEN=<目标长度>
```

`YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS` 必须继续保持 `262144`，不能改成目标长度。

CP8 下还要保证每卡 token 容量覆盖整个序列：

| 目标长度 | 四个容量字段 | `MAX_TOKENS_PER_GPU` | 入口/状态 |
| ---: | ---: | ---: | --- |
| 262144 | 262144 | 32768 | `start_qwen36_yarn_smoke_single52.sh` |
| 270336 | 270336 | 33792 | `start_qwen36_yarn_stage8_270k_single52.sh` |
| 300000 | 300000 | 37500 | 已完成专项验证；正常集成运行需使用独立 launcher |

300K launcher 的核心配置应为：

```bash
export FEAT_YARN=1
export SEQ_LENGTH=300000
export MAX_POSITION_EMBEDDINGS=300000
export ROLLOUT_MAX_CONTEXT_LEN=300000
export VLLM_MAX_MODEL_LEN=300000
export MAX_TOKENS_PER_GPU=37500
export TP=2
export CP=8
export EP=8
```

这段只描述容量配置，不是完整启动命令。300K 正常集成任务应复制 270K 专用 launcher 的拓扑并单独命名；不要解除 262K smoke launcher 的固定长度保护。

## 4. vLLM 前置条件

当前 YaRN/MRoPE 修复位于 vLLM 0.23 代码树。实际进程必须加载：

```text
/workspace/vllm-023
/workspace/vllm-ascend-023
```

不得加载旧的 `/workspace/vllm` 或 `/workspace/vllm-ascend`。可在启动前检查解析路径：

```bash
python3 - <<'PY'
import inspect
import vllm
import vllm_ascend

print(inspect.getfile(vllm))
print(inspect.getfile(vllm_ascend))
PY
```

当 `FEAT_YARN=1` 时，正式 runner 会添加 `--vllm-allow-long-max-model-len`，VIME 再把 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` 注入实际 vLLM 子进程。不要只在父 shell 中设置后就假定子进程已经收到；以 VIME 参数和预检结果为准。

## 5. 独立预检

专用 launcher 会自动运行预检。若在开发 launcher 时需要手动检查，可以执行：

```bash
cd /workspace/vime-a3pd-integrated

export FEAT_YARN=1
export YARN_ROPE_THETA=10000000
export YARN_PARTIAL_ROTARY_FACTOR=0.25
export YARN_FACTOR=4.0
export YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS=262144
export YARN_BETA_FAST=32.0
export YARN_BETA_SLOW=1.0
export YARN_MSCALE=1.0
export YARN_MSCALE_ALL_DIM=0.0
export YARN_CORRECTION_RANGE_ROUND_TO_INT=1

source scripts/models/qwen3.5-35B-A3B.sh

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
python3 tools/qwen36_yarn_preflight.py \
  --model /home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16 \
  --hf-overrides "${QWEN36_VLLM_HF_OVERRIDES}" \
  --max-model-len 262144
```

预检会检查：checkpoint 配置可读、请求的 `rope_type` 是 `yarn`、vLLM 最终解析出的完整 YaRN 指纹没有变化、最终 `max_model_len` 等于请求值。

## 6. 启动后的通过条件

启动日志必须出现：

```text
Resolved matched YaRN fingerprint: ...; capacity: ...
```

并确认：

1. 指纹中的 `rope_type=yarn`，其余字段与第 2 节完全一致。
2. `seq_length`、`max_position_embeddings`、`rollout_max_context_len` 和 `vllm_max_model_len` 四项完全相等。
3. vLLM engine 实际进入 ready，Polar 收到 session，而不是只有 `/health`、`/metrics` 轮询。
4. rollout token、rollout logprob、训练侧 replay/TIS 证据正常落盘到 `VIME_SAVE_TIS_LOGPROBS` 指定目录。
5. 所有 logprob、ratio、TIS 数值有限，不出现 NaN/Inf；精度结论以对应测试阶段保存的证据为准，不能只以“任务跑通”为准。

300K 真生成专项验证已覆盖 262144-token prompt 后继续生成到 300000-token 总长。当前 checkpoint 和上述指纹下，实测逐 token 对比为：`abs diff mean=0.00095896`、`p95=6.77e-5`、`p99=7.63e-4`、`TIS mean=1.0002768`，位置覆盖完整且无 NaN/Inf。完整验证边界和证据说明见 [YaRN 设计与验证记录](./design/qwen36_yarn_rl_training_enablement.md)。

## 7. 常见错误

`YaRN must be enabled on both training and rollout`

: 只开启了 Megatron 或只给 vLLM 配了 YaRN。使用 `FEAT_YARN=1` 并让模型脚本统一生成两侧参数。

`Training/rollout YaRN fingerprint mismatch`

: 两侧至少一个 YaRN 字段不同。不要手写第二份 HF override。

`YaRN training/rollout capacity mismatch`

: 四个容量字段没有设置成同一个值。

vLLM 拒绝超过 checkpoint 原始长度

: 检查正式参数中是否存在 `--vllm-allow-long-max-model-len`，并检查实际 vLLM 子进程环境是否包含 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`。

启动后长期只有 `/health` 和 `/metrics`

: 这不表示 rollout 正常。继续检查 Polar session 派发、gateway 返回码和 rollout server 日志。不要通过修改 Polar 本地配置来绕过。

## 8. 关闭 YaRN

回到非 YaRN 配置时：

```bash
export FEAT_YARN=0
```

同时恢复该任务原本的四项容量和 `MAX_TOKENS_PER_GPU`，重新 source 模型脚本并启动新进程。不要让旧 YaRN 进程或旧 HF override 留在环境中，也不要混用不同 position 配置产生的 checkpoint。
