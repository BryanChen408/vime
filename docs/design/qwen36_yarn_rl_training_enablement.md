# Qwen3.6 Agentic RL YaRN 训练与 Rollout 适配方案

> 状态：功能已实施；300K 外推数学、推理边界、TP2/CP8 训练和真实 decode 训推精度验收通过
> 更新时间：2026-09-15
> 适用范围：VIME + Megatron-LM/MindSpeed 训练、vllm-ascend rollout、Qwen3.6-35B-A3B 纯文本 Agentic RL
> 目标配置：静态 YaRN，`factor=4.0`，`original_max_position_embeddings=262144`，本轮验收最大上下文 `300000`

## 1. 结论摘要

这次适配不需要自行实现 YaRN 数学公式，也不需要改 MindSpeed 核心代码。推荐路径是：

1. 从 NVIDIA Megatron-LM 上游回移通用 YaRN 的参数解析和 `TransformerConfig` 映射。
2. 对上游 `YarnRotaryEmbedding` 补一个 Qwen3.6 必需的 `rotary_percent` 修正。Qwen3.6 的 rotary head dimension 应为 `256 * 0.25 = 64`，而当前实现使用了完整的 `kv_channels=256`。
3. 在 VIME 中同时开启训练侧 YaRN 与 rollout 侧 HF override，并显式把 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` 传进 Ray rollout actor 创建的 vLLM 子进程。
4. 先在已验证 homo launcher 的原生 `262144` 长度验证 YaRN 数值和训推一致性，再逐级扩到 `270K` 和 `300K`。开启 YaRN 与提升到原始窗口之外是两个独立动作，不应一次完成。

实施最终涉及 VIME、Megatron-LM 和 vLLM core 三个仓库。原计划认为 vLLM 的 HF override
路径可直接复用；262K 训推证据对比随后发现，vLLM 0.23 的 `MRotaryEmbedding` 会把用于缓存
扩容的 4 倍长度误当成 YaRN correction range 的原始上下文长度，因此必须补一个 vLLM core
修复。MindSpeed 和 vllm-ascend 插件仍不需要新增 YaRN 数学实现。

由于最终只推送 VIME 仓库，已测 Megatron-LM 和 vLLM 状态以有序 patch series 纳入
[`docs/design/patches/qwen36_yarn_20260915/`](./patches/qwen36_yarn_20260915/README.md)。
每个外部仓库分别保存运行前置和 YaRN 专项补丁，并钉死输入 revision、应用顺序和 SHA-256；
不通过运行时 monkey-patch 修改外部包。

`CP=8` 不是 YaRN 的必选项。它是当前生产拓扑下的必测项：YaRN 本身不依赖 Context Parallel，但必须证明它在 VIME 当前 packed THD 和 CP 切分路径上没有产生位置错位。

## 2. 背景与目标

上游 SFT 已使用静态 YaRN 外推。RL 阶段如果仍使用默认 RoPE，即使训练序列尚未超过 262144，actor 的位置编码也已经与 SFT 模型语义不一致。Qwen 官方也明确提示：静态 YaRN 会影响短上下文行为，因此不能把它理解为“只有超过原始长度后才生效”。

RL 侧需要同时满足：

- training 使用与 SFT 完全相同的 YaRN 参数；
- rollout 使用与 training 完全相同的 YaRN 参数；
- rollout 后端允许 `max_model_len` 超过模型原始声明；
- 不改变 checkpoint 参数结构，现有 SFT checkpoint 可直接加载；
- feature 关闭时保持当前默认 RoPE 行为不变；
- 先覆盖当前纯文本 Agentic RL，不在本次范围内自行设计 YaRN + 多模态 MRoPE。

## 3. 参数基线与唯一事实源

当前计划参数来自 Qwen3.6-35B-A3B 官方示例：

```json
{
  "text_config": {
    "rope_parameters": {
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10],
      "rope_type": "yarn",
      "rope_theta": 10000000,
      "partial_rotary_factor": 0.25,
      "factor": 4.0,
      "original_max_position_embeddings": 262144
    }
  }
}
```

但真正的唯一事实源必须是本次 RL 所承接的 SFT checkpoint 配置和 SFT 启动日志，而不是模型主页示例。实施前必须拿到并固化下列参数：

| 字段 | 当前预期值 | 必须与 SFT 核对 |
| --- | ---: | --- |
| `rope_type` | `yarn` | 是 |
| `rope_theta` | `10000000` | 是 |
| `partial_rotary_factor` | `0.25` | 是 |
| `factor` | `4.0` | 是 |
| `original_max_position_embeddings` | `262144` | 是 |
| `beta_fast` | `32` | 是，若 SFT 未显式设置则确认默认值 |
| `beta_slow` | `1` | 是，若 SFT 未显式设置则确认默认值 |
| `mscale` | `1.0` | 是，若 SFT 未显式设置则确认默认值 |
| `mscale_all_dim` | `0.0` | 是，若 SFT 未显式设置则确认默认值 |
| correction range rounding | truncate/向下取整 | 是 |

当前检查到的本地默认 HF 配置仍是 `rope_type=default`，因此不能依赖 checkpoint 的默认 metadata 自动开启 YaRN。启动时应打印训练侧和 rollout 侧的 YaRN fingerprint，并在字段不一致时直接失败，避免训练已经运行后才从 KL 或精度异常反推配置错误。

建议 fingerprint 至少包含：

```text
rope_type, rope_theta, partial_rotary_factor, factor,
original_max_position_embeddings, beta_fast, beta_slow,
mscale, mscale_all_dim, max_position_embeddings/max_model_len
```

## 4. 当前代码状态与阻塞

### 4.1 训练侧已有的基础

VIME 当前 Qwen3.6 模型配置已经包含：

- `--position-embedding-type rope`
- `--rotary-percent 0.25`
- `--rotary-base 10000000`

本地 Megatron-LM 的 `GPTModel` 已存在 `position_embedding_type == "yarn"` 分支，attention 也已经读取 YaRN concentration factor。这意味着主体执行路径已具备，不需要新写 attention 或 RoPE kernel。

### 4.2 训练侧的实际阻塞

阻塞一：参数入口没有接通。

- 本地 Megatron 参数解析器尚未把 `yarn` 加入 `--position-embedding-type` 可选项。
- 本地 `core_transformer_config_from_args` 尚未把 YaRN CLI 参数映射到 `TransformerConfig`。
- NVIDIA 上游已有参考实现，应直接回移其参数名、默认值和映射逻辑。

阻塞二：通用 `YarnRotaryEmbedding` 没有应用 `rotary_percent`。

- Qwen3.6 当前 `kv_channels=256`。
- `partial_rotary_factor/rotary_percent=0.25`。
- 正确 rotary dimension 是 64，对应 32 个 inverse frequencies。
- 当前 YaRN 实现直接令 `self.dim = kv_channels`，会按 256 维计算；这不是性能差异，而是位置编码数学对象错误。
- 标准 `RotaryEmbedding` 已有成熟规则：当 `rotary_percent < 1.0` 时按比例缩小 rotary dimension。YaRN 应复用同一规则。

阻塞三：MTP 校验只接受 `rope/none`。

- 如果生产启用 MTP，当前参数校验会拒绝 `yarn`。
- MTP 内部复用同一个 `rotary_pos_emb`，没有第二套 YaRN 公式，因此预计只需扩大 allowlist。
- 但 NVIDIA 上游最新实现也尚未声明这条组合已验证，所以该改动必须单独提交和测试。
- 如果本次 RL 不启用 MTP，则不做此改动，减少风险面。

### 4.3 Rollout 侧的实际阻塞

vllm-ascend 可通过 HF overrides 接收官方 YaRN 配置，VIME 也已有
`--vllm-hf-overrides` 参数并能传到后端；但当前实际调用的是 vLLM core 的
`MRotaryEmbedding`，不能据此推导“核心能力无需开发”。

2026-09-14 的逐位置对比确认了一个 vLLM 0.23 core 缺陷：

- `MRotaryEmbedding` 为视频位置预留缓存时先把 `max_position_embeddings` 扩成 4 倍；
- YaRN `_compute_inv_freq` 随后读取这个已扩大的字段计算 correction range；
- 目标配置应以 `262144` 得到 `(14, 22)`，旧实现实际以 `1048576` 计算；
- 结果有 9/32 个 inverse-frequency bin 不同；位置 32768 的 cos 最大绝对误差已为
  `1.17`，位置 131071 达 `2.24`；
- 旧实现还会在 4 倍 MRoPE 缓存上再次乘 YaRN factor，生成 16 倍而非 4 倍缓存。

修复要求明确分开 `original_max_position_embeddings` 与 cache capacity：前者只参与 YaRN
频率计算，后者只决定 cos/sin cache 长度。Qwen3.6 目标配置修复后 cache shape 应为
`(1048576, 64)`，并与 Transformers 在目标位置逐元素一致。该改动属于 vLLM core，
不修改 Polar，也不在 vllm-ascend 另写一套公式。

但 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` 不能只在外层 shell 中 `export`：

- rollout Ray actor 的 `runtime_env` 当前只显式传递少量环境变量；
- 代码注释也说明不能依赖 Ray actor 继承 driver 环境；
- vLLM engine 子进程环境来自 rollout actor 内部构造的 environment；
- 因此外层 shell 有值，不代表实际 engine 进程一定有值。

推荐在 VIME 增加显式开关，并在 `build_vllm_subprocess_env` 中注入：

```text
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
```

启动验证应检查真实 vLLM 子进程环境或其启动日志，而不是只检查 launcher shell。

### 4.4 MindSpeed 的边界

MindSpeed 自带的通用 YaRN patch 当前按 `hidden_size // num_attention_heads` 计算维度，即 128；Qwen3.6 所需维度是 64。并且该 wrapper 在当前代码中没有看到完整注册链路。

因此本方案明确不使用 MindSpeed 的 `--rope-scaling-type yarn`，也不修改 MindSpeed。位置编码对象由 Megatron-LM 的 `YarnRotaryEmbedding` 创建，MindSpeed 继续承担现有 NPU 执行和并行优化职责。

## 5. 设计方案

### 5.1 总体数据流

```text
SFT YaRN fingerprint
        |
        +--> training launcher
        |      `--position-embedding-type yarn`
        |      Megatron TransformerConfig
        |      YarnRotaryEmbedding(dim=64)
        |
        +--> rollout launcher
               merged `--vllm-hf-overrides`
               `--vllm-max-model-len`
               explicit engine subprocess env

training fingerprint == rollout fingerprint == SFT fingerprint
```

### 5.2 Megatron-LM 参数映射

采用 NVIDIA 上游已有的 YaRN CLI 和 `TransformerConfig` 字段，不创建 VIME 私有数学参数。

| HF/SFT 字段 | Megatron 训练参数 |
| --- | --- |
| `rope_type=yarn` | `--position-embedding-type yarn` |
| `rope_theta=10000000` | `--rotary-base 10000000` |
| `partial_rotary_factor=0.25` | `--rotary-percent 0.25` |
| `factor=4.0` | `--rotary-scaling-factor 4.0` |
| `original_max_position_embeddings=262144` | `--yarn-original-max-position-embeddings 262144` |
| `beta_fast=32` | `--yarn-beta-fast 32` |
| `beta_slow=1` | `--yarn-beta-slow 1` |
| `mscale=1.0` | `--mscale 1.0`，或确认上游默认值 |
| `mscale_all_dim=0.0` | `--mscale-all-dim 0.0`，或确认上游默认值 |
| truncate correction range | 上游 correction-range-round/truncate 配置 |

这里的实现原则是“按当前本地 Megatron 代码结构手工回移上游 mapper”，而不是直接 cherry-pick。上游函数所在文件已经移动，直接 cherry-pick 容易混入无关参数重构和产生冲突。

### 5.3 Rotary dimension 修正

在 `YarnRotaryEmbedding` 初始化时复用标准 `RotaryEmbedding` 的既有规则：

```python
self.dim = kv_channels
if rotary_percent < 1.0:
    self.dim = int(self.dim * rotary_percent)
```

这 1 至 3 行是本方案唯一需要补充、且上游最新代码尚未覆盖的核心数学适配。后续 YaRN inverse-frequency、correction range 和 attention scaling 仍完全使用上游实现。

RoPE apply 路径已经根据 `freqs.shape[-1]` 只旋转 Q/K 的前 `rot_dim`，其余维度原样透传。因此修正后应得到：

```text
head dimension:          256
rotary dimension:         64
inverse-frequency count:  32
pass-through tail:        192
```

### 5.4 VIME 训练配置

建议在共享 Qwen3.6 model script 中添加 opt-in feature gate，例如 `FEAT_YARN=1` 时追加 YaRN 参数，而不是直接改变所有 Qwen 任务的默认行为。生产 launcher 显式设置该 gate。

这样做有三个目的：

- baseline 可以一键退回默认 RoPE；
- 未承接 YaRN SFT 的其他任务不被静默影响；
- A/B 数值回归可以在相同代码版本下完成。

feature gate 关闭时，生成的参数必须与当前脚本完全一致。

VIME 当前会把 `max_position_embeddings` 设成 `seq_length`。因此训练的实际 sequence cap、Megatron 的 position capacity、rollout 请求长度和 vLLM `max_model_len` 需要作为一组显式配置检查，但它们不应与 YaRN 的 `original_max_position_embeddings=262144` 混淆：

- `original_max_position_embeddings` 是 YaRN 公式的原始训练长度基准；
- `seq_length/max_position_embeddings/max_model_len` 是当前运行允许的容量上限。

### 5.5 VIME Rollout 配置

当前 launcher 已经使用一次 `--vllm-hf-overrides` 修改 architecture。开启 YaRN 时必须把 architecture 与 `text_config.rope_parameters` 合并到同一个 JSON，不能重复传两个同名参数并依赖覆盖顺序。

示意配置：

```bash
--vllm-hf-overrides '{
  "architectures": ["Qwen3_5MoeForConditionalGeneration"],
  "text_config": {
    "rope_parameters": {
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10],
      "rope_type": "yarn",
      "rope_theta": 10000000,
      "partial_rotary_factor": 0.25,
      "factor": 4.0,
      "original_max_position_embeddings": 262144
    }
  }
}'
```

同时：

- 增加 VIME 参数，如 `--vllm-allow-long-max-model-len`；
- 在 vLLM 子进程 environment builder 中显式注入 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`；
- 将 `--vllm-max-model-len` 与本阶段测试长度匹配；
- 启动日志打印最终解析出的 rope config 和最大长度；
- 不依赖 checkpoint 默认 config，也不依赖 driver shell 隐式继承环境变量。

### 5.6 CP、MTP 与 fused RoPE

`CP=8` 与 YaRN 没有配置依赖。YaRN 负责“某个绝对 position 应该得到什么频率”，CP 负责“序列 token 如何分给设备”。理论上 CP 可以是 1、2、4、8；只有两者组合时 position ids 或 packed sequence 元数据处理错误，才会出现结果差异。

所以本次结论是：

- `CP=8`：不是必选项，是生产拓扑必测项；
- `CP=1`：作为数值基准；
- `CP=1` 与 `CP=8`：同一输入 gather 后结果应在 NPU kernel 合理容差内一致；
- MTP：若生产启用则单独适配和验收，否则从本次实现删除；
- fused RoPE：首轮保持关闭，先验证非融合参考路径；融合路径后续单独开关和验收。

### 5.7 纯文本范围

当前 Agentic RL 是纯文本时，Qwen 配置里的 `mrope_interleaved` 和 `mrope_section` 不会引入额外的图像/视频位置轴计算；所有轴共享文本 position ids，训练侧使用 partial rotary YaRN 即可建立对应关系。

本次不承诺多模态 YaRN + MRoPE。若未来 rollout 输入包含图像或视频，需要重新核对三轴 position ids、section 拆分与 YaRN frequency scaling，不能直接把本文结论外推过去。

## 6. 改动面

### 6.1 Megatron-LM

| 文件/模块 | 预计改动 | 来源 | 风险 |
| --- | ---: | --- | --- |
| `megatron/training/arguments.py` | 40-50 行 | NVIDIA 上游回移 | 低至中：本地版本结构不同，需映射测试 |
| `megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py` | 1-3 行 | 本地必要修正，复用标准 RoPE 规则 | 中：核心数值点，但逻辑很小 |
| YaRN parser/config/numerical tests | 80-140 行 | 新增 | 低 |
| MTP 参数校验 | 1 行，条件项 | 本地适配 | 中：上游未验证组合 |
| MTP/CP 相关测试 | 30-60 行，条件项 | 新增 | 低 |

### 6.2 VIME

| 文件/模块 | 预计改动 | 来源 | 风险 |
| --- | ---: | --- | --- |
| `vime/backends/vllm_utils/arguments.py` | 5-8 行 | 本地集成 | 低 |
| `vime/backends/vllm_utils/vllm_engine.py` | 2-4 行 | 本地集成，沿用已有 env builder 模式 | 低 |
| `tests/unit/backends/vllm_utils/test_vllm_engine.py` | 15-30 行 | 新增 | 低 |
| `scripts/models/qwen3.5-35B-A3B.sh` | 约 8 行 | 启动配置 | 低 |
| `scripts/run-qwen36-35b-polar-multi-pd.sh` | 10-20 行 | 启动配置与 merged overrides | 中：当前文件存在其他未提交修改，实施时需逐段合并 |
| Megatron 参数/启动校验测试 | 20-40 行 | 新增 | 低 |

### 6.3 MindSpeed 与 Polar

| 仓库 | 计划改动 | 说明 |
| --- | ---: | --- |
| MindSpeed | 0 | 不启用其通用 YaRN wrapper；沿用当前 NPU 执行路径 |
| Polar | 0 | agent 协议和生成流程不需要感知位置编码实现 |

### 6.4 总量估算

- 生产代码和配置：约 75-110 行；
- 测试：约 150-250 行；
- 生产文件：6-8 个，分布在 Megatron-LM 和 VIME；
- 若 MTP 关闭：减少 1 个生产校验改动和约 30-60 行测试；
- MindSpeed、Polar：无代码改动。

### 6.5 上游参考与本地设计的比例

可以直接采用上游的部分：

- YaRN inverse-frequency 公式；
- correction range 计算；
- attention scaling/concentration factor；
- 参数名称、默认值和 `TransformerConfig` 映射；
- `GPTModel` 中 YaRN 对象的现有接入路径。

必须本地补充的部分：

- `rotary_percent` 生效：1-3 行核心逻辑；
- VIME 显式传递 vLLM 长上下文环境变量：约 7-15 行生产代码；
- 双侧启动参数、feature gate 和 fingerprint：约 10-25 行脚本/校验；
- MTP allowlist：约 1 行，仅在生产启用 MTP 时需要。

因此，不含测试时，本地特有的生产逻辑与接线约 20-45 行；其中真正影响 YaRN 数学计算的只有 1-3 行。其余不应自行发明算法。

## 7. 预期 Commit 数量与结构

由于 Megatron-LM 和 VIME 是两个独立 Git 仓库，不可能用一个原子 commit 同时提交。建议形成一组兼容 SHA，并在发布记录或 launcher 注释中记录配对关系。

生产启用 MTP 时，推荐 4 个功能 commit；不启用 MTP 时为 3 个。本文档本身单独作为 docs commit，因此总数分别为 5 或 4。

### Commit 1：Megatron 上游能力回移

```text
backport(yarn): wire non-MLA YaRN CLI and TransformerConfig
```

内容：

- 增加 `yarn` position embedding 选项；
- 回移 YaRN CLI 参数；
- 回移 args 到 `TransformerConfig` 的映射；
- 增加 parser/config 单测；
- 不包含 Qwen 私有修正，不开启任何生产脚本。

### Commit 2：Qwen partial rotary 数值修正

```text
fix(yarn): honor rotary_percent and add Transformers parity tests
```

内容：

- 让 `YarnRotaryEmbedding` 按 `rotary_percent` 计算 dimension；
- 增加 Transformers oracle 数值测试；
- 验证非 rotary tail 不变；
- 验证 feature 未开启时标准 RoPE 无回归。

把它与上游回移拆开，是为了明确标记“上游已有能力”和“本地发现的上游缺口”，便于后续向上游提 issue/PR 或删除本地 patch。

### Commit 3：MTP 组合支持，可选

```text
feat(mtp): allow YaRN rotary embeddings in MTP
```

内容：

- 扩大 MTP position embedding allowlist；
- 增加 MTP on/off、CP=1/8 的最小执行测试；
- 若生产任务确认不启用 MTP，则整个 commit 不做。

### Commit 4：VIME 双侧启用

```text
feat(qwen36): enable matched YaRN for training and vLLM rollout
```

内容：

- 增加 VIME vLLM long-max-model-len 开关；
- 显式注入 engine 子进程环境变量；
- model script 增加 opt-in YaRN 参数；
- production launcher 合并 HF overrides；
- 配置 rollout max length；
- 增加 environment forwarding 和启动参数测试；
- 打印或校验训推 YaRN fingerprint。

该 commit 同时切 training 和 rollout，避免出现仓库内部只开一侧的中间状态。它依赖前述 Megatron-LM commit SHA。

### Commit 5：本文档

```text
docs(yarn): document Qwen3.6 RL enablement and validation plan
```

文档 commit 可以先提交，也可以在功能实现合入时 rebase 到最后；不与任何运行逻辑混合。

## 8. 验证方案

验证按“配置 -> 公式 -> 单卡/单进程 -> 分布式 -> 训推一致性 -> 长度扩容”的顺序推进。不能直接以 1M 训练能启动作为正确性证明，因为错误的 rotary dimension 同样可能正常运行且不报错。

### 8.1 配置和结构单测

必须覆盖：

- parser 接受 `--position-embedding-type yarn`；
- 所有 CLI 字段无损映射到 `TransformerConfig`；
- Qwen3.6 的 rotary dimension 为 64、inverse-frequency 数量为 32；
- 默认 feature 关闭时仍构造标准 RoPE；
- YaRN buffer 不引入 checkpoint 参数 key 变化；
- 旧 SFT checkpoint 加载没有 missing/unexpected parameter；
- VIME 开关开启时，真实 vLLM 子进程 environment 包含 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`；
- 开关关闭时不注入该变量。

### 8.2 Transformers 数值 Oracle

使用与目标 checkpoint 兼容的 Transformers `ROPE_INIT_FUNCTIONS`/`_compute_yarn_parameters` 作为独立 oracle，不复制一份相同公式到测试里。

测试参数：

```text
head_dim=256
partial_rotary_factor=0.25
rotary_dim=64
rope_theta=10000000
factor=4
original_max_position_embeddings=262144
beta_fast=32
beta_slow=1
```

重点位置：

```text
0, 1, 131071, 262143, 262144, 524287, 1009999
```

验收条件：

- float32 inverse frequencies 与 Transformers 最大绝对误差为 0，或在版本差异下小于 `1e-7`；
- cos/sin 的 `atol/rtol <= 1e-6`；
- attention scaling 与 Transformers 一致；当前参数预期约为 `1.1386294361`；
- Q/K 前 64 维按 YaRN 旋转；后 192 维 bitwise 不变；
- `factor=1` 回归结果与标准 RoPE 一致。

前期探针已经得到：Transformers 与采用 64 维的 Megatron YaRN inverse frequencies 最大差异为 0，attention scale 也一致；当前 256 维路径则结构性不匹配。正式实现后需要把该探针固化为 CI 测试。

### 8.3 NPU 与并行验证

先关闭 fused RoPE，使用最容易比对的参考路径。

最小矩阵：

| 场景 | 目的 |
| --- | --- |
| CP=1，MTP off | 数值基线 |
| CP=8，MTP off | 验证生产 packed THD/position 切分 |
| CP=1，MTP on | MTP 基线，仅生产启用时 |
| CP=8，MTP on | 生产组合，仅生产启用时 |

在当前生产可用拓扑上再覆盖 TP/EP 组合，至少运行一个完整 microbatch，检查：

- logits、loss、grad norm 全部 finite；
- CP=1 与 CP=8 gather 后输出在既有 NPU kernel 容差内一致；
- packed sequence 的每个样本 position 从正确起点重置；
- save/resume 后首步结果与连续运行一致；
- checkpoint 没有新增 trainable state。

这里不能要求不同并行 kernel bitwise 一致，容差应先用改动前的默认 RoPE 跑一遍建立平台基线，再用同一阈值验收 YaRN。

### 8.4 Rollout 验证

必须验证最终进程状态，而不是只验证命令行字符串：

- 从实际 engine 进程 `/proc/<pid>/environ` 或可信启动日志确认 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`；
- 从 vLLM 最终 resolved config 确认所有 rope parameters；
- 确认 architecture override 与 rope override 同时存在；
- 确认 `max_model_len` 是本阶段配置值；
- 发起超过 262144 token 的边界请求，第一阶段使用约 270K，而不是立即 1.01M；
- 检查生成无 position/index 越界、无 KV cache 配置回退、无静默截断。

### 8.5 训推一致性验证

使用同一 checkpoint 和同一批 token：

- rollout 保存 token ids、positions、attention metadata 与 rollout logprobs；
- training actor 对相同 token 重算 logprobs；
- 比较已有 TIS mismatch 指标、KL、entropy 和逐 token 差异分布；
- 阈值使用改动前同模型、同 kernel、同精度的基线，不人为设定跨后端 bitwise 相等；
- 固定样本运行 5-10 个 RL step，检查 loss、KL、entropy、reward、grad norm 没有阶跃式异常。

如果只验证“vLLM 能生成”和“Megatron 能反传”，仍无法证明两侧用的是同一套位置编码，因此本项是上线硬门槛。

本仓库的 homo smoke launcher 默认设置 `VIME_SAVE_TIS_LOGPROBS=<POLAR_OUTPUT_DIR>/tis_evidence`。
训练 loss 的实际 microbatch 计算点会在该目录写入 `tis_evidence_step*_mb*_rank*.pt`，每个文件包含
schema 版本、token SHA256、response mask、logit position、target token、rollout/train/current
logprob、packed `cu_seqlens`、CP/TP/DP 配置和 YaRN fingerprint。使用
`tools/analyze_qwen36_yarn_evidence.py summary <dir>` 可复算 finite、position coverage、
逐 token logprob 差异、TIS importance ratio 和长度覆盖；使用 `compare <reference> <candidate>`
可在 token/position 完全相同的前提下比较 CP=1/CP=8 或 YaRN off/on 的训练 logprob。

因此第 1 轮不是只看日志：必须有该目录的完整文件集；第 2/3 轮必须先生成两组证据再比较，
比较结果保存为 JSON 并与同 checkpoint、同 token 的基线一起归档。没有证据文件的运行只能算
功能观察，不能算训推一致性通过。

### 8.6 长度分阶段

建议依次推进：

1. `262144`：在 homo 已验证的原生容量上只改变位置编码，跑完整 rollout + 1 个训练 step，隔离公式和接线问题。
2. `~270K`：首次跨过原始 262144 边界，验证外推和环境变量。
3. `300000`：本轮最终外推精度上限，使用多样 token 完成独立 rollout 边界实算、
   TP=2/CP=8 训练 forward/backward 和逐位置 NPU 数值对照。

每一阶段分别调整并记录：

```text
training seq_length
training max_position_embeddings
rollout request/token cap
vLLM max_model_len
KV cache/memory-related capacity settings
```

2026-09-15 用户将本轮范围明确收敛到最长 `300000`，因此原计划的 `~524K` 和
`1010000` 容量/恢复压测取消，不属于本轮通过条件。YaRN 解决频率外推，不解决显存、
KV cache、通信量和训练吞吐问题。

### 8.7 2026-09-14 实测状态

262K homo 首轮 `qwen36_yarn_s1_oldpolar_20260914_032034` 完成完整 rollout 和一个训练
step：CP=8、TP=2、EP=8，150792 个有效 response token，24 份 rank/microbatch 证据均
finite，最大实际 logit position 为 125385。它证明配置接线、checkpoint 加载、packed CP
覆盖和反向链路可执行，但由于该轮 rollout 使用了上述错误的 vLLM MRoPE correction range，
不能作为训推精度通过证据。

使用该轮固定 token 做的 standard RoPE/YaRN training replay 都能完成 3/3 microbatch，
但 YaRN 没有相对 standard RoPE 明显收敛到 rollout logprob；同时相同配置重复运行的逐 token
差异也较大。继续追查后用实际 Qwen3.5 构造参数直接比较两侧频率，定位到 vLLM core 缺陷，
因此这些旧 replay 只保留为问题发现证据，不作为门槛判定。

修复后的静态验证结果：

- Megatron YaRN 对 Transformers oracle：10 个用例通过；
- vLLM MRoPE YaRN regression：2 个用例通过；
- vLLM 实际 Qwen3.5 cache 为 `(1048576, 64)`；
- inverse frequencies 和 attention scale 与 Transformers 精确相等；
- 位置 `0, 1, 32768, 65536, 98304, 131071, 262143, 524287, 1009999` 的
  cos/sin 最大绝对误差均为 0。

下一验收动作必须重新生成修复后的 rollout evidence，再用相同 token 做 training replay。
在这组新证据完成前，不进入约 270K 的首次跨界阶段。

### 8.8 第 5 步实测结果：NPU CP=1 最小 forward/backward

2026-09-14 使用 `scripts/run_qwen36_yarn_step5_cp1.sh` 完成最小训练回放：

- 运行：`qwen36_yarn_step5_cp1_20260914_175832`；CP=1、TP=1、PP=1、EP=8，8 个 NPU
  rank 使用物理卡 4--11；fused RoPE 明确关闭（`apply_rope_fusion=False`）。
- 训练侧解析到的 YaRN fingerprint 为 `rope_type=yarn`、`rope_theta=10000000`、
  `partial_rotary_factor=0.25`、`factor=4`、`original_max_position_embeddings=262144`、
  `beta_fast=32`、`beta_slow=1`、`mscale=1`、`mscale_all_dim=0`，容量为 262144。
- 一个 microbatch、一个训练 step 的 forward、backward、梯度归约和 optimizer 生命周期均
  正常完成。最终指标为 `train/loss=0.0013503365`、`train/grad_norm=0.8496037`，均为
  finite；日志无 `Traceback`、OOM 或最终错误。
- 落盘 8/8 个 TIS evidence 和 8/8 个 debug train 文件。所有文件可独立读取，包含的
  train/current/rollout logprob 及张量递归 finite 检查全部通过；8 个 rank 的
  `CP/TP/DP=(1,1,8)` 和 YaRN fingerprint 一致。

本轮回放使用 Stage 1 真实样本的 8 条记录，重新编号分组并裁剪为 512 prompt + 1024 loss
tokens，以把测试限定在 CP=1 的最小 NPU forward/backward 路径。它证明第 5 步的功能链路和
数值有限性通过，但不是长上下文外推、CP=8 或训推 logprob 一致性的证据；后者分别留给后续
步骤。裁剪后的 rollout logprob 与当前 actor 的差异不作为本步精度门槛。

### 8.9 第 6 步实测结果：生产 TP=2 下的 CP=8 对拍

2026-09-15 使用与生产训练一致的 TP=2、EP=8 拓扑完成四组最小训练回放。测试固定使用
同一批 5145 个有效 token，fused RoPE 关闭；CP=1 和 CP=8 均完成 forward、backward、
梯度归约和一个 optimizer step：

| 位置编码 | CP | 运行 | loss | grad norm |
| --- | ---: | --- | ---: | ---: |
| YaRN | 1 | `qwen36_yarn_step6_yarn_cp1_tp2_20260915_0202` | 0.0015846272 | 0.8406990 |
| YaRN | 8 | `qwen36_yarn_step6_cp8_20260915_015647` | 0.0008356203 | 0.9483836 |
| standard RoPE | 1 | `qwen36_yarn_step6_rope_cp1_tp2_20260915_0207` | 0.0012155212 | 0.8952698 |
| standard RoPE | 8 | `qwen36_yarn_step6_rope_cp8_tp2_20260915_0211` | 0.0007533017 | 0.9516352 |

四轮日志均无 `Traceback`、OOM 或最终错误；所有 TIS evidence/debug train 文件可读取且
递归 finite，packed position 覆盖完整，落盘 current logprob 与训练时 logprob 的逐 token
差异为 0。CP=1 与 CP=8 的 token 集合也逐项一致。

逐 token training logprob 的 CP=1/CP=8 差异如下：

| 位置编码 | mean | p50 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| YaRN | 0.313387 | 0.057761 | 1.362051 | 2.265476 | 5.545326 |
| standard RoPE | 0.296389 | 0.054225 | 1.315181 | 2.232287 | 5.374091 |
| YaRN 相对基线增量 | 0.016998 | 0.003535 | 0.046870 | 0.033189 | 0.171235 |

已有同拓扑重复 replay 的运行间波动为 mean `0.166--0.191`、p95 `1.028--1.166`；因此
YaRN 相对 standard RoPE 新增的 mean `0.0170`、p95 `0.0469` 明显小于当前 NPU 运行间
波动，未观察到 YaRN 引入额外 CP 精度退化。第 6 步据此判定通过。

本步采用生产必需的 TP=2/CP=8 作为目标拓扑；一次早期 TP=1/CP=8 启动在模型初始化阶段
主动终止，不计入结果。当前样本最大 position 为 1534，因此本结论只覆盖 CP 分片接线、
packed position 和短序列数值稳定性，不覆盖长上下文、完整 rollout 或训推一致性；这些由
第 7 步及后续阶段验收。

### 8.10 第 7 步实测结果：262144 homo 完整 rollout + train

修正 vLLM MRoPE YaRN correction range 后，运行
`qwen36_yarn_s1_vllmfix_20260914_0924` 在旧 Polar 和生产训练拓扑 TP=2、CP=8、EP=8 下
完成两轮同步 rollout 和两个训练 step。运行时 traceback 路径确认加载的是
`/workspace/vllm-023`，其补丁把 YaRN 原始上下文 262144 与 MRoPE 的 4 倍 cache 长度分离；
training 与 rollout 的 resolved fingerprint 完全一致，四个容量配置均为 262144。

- 两轮 rollout 的成功率均为 1.0，每轮 2/2 group 接受，rejected/top-up/abort/requeue 均为
  0；两份 debug rollout 正常落盘。
- step 0：`loss=0.0001740046`、`grad_norm=5.31228`；step 1：
  `loss=0.0046456531`、`grad_norm=51.39125`。两步均完成权重更新，iteration 1 checkpoint
  保存成功。
- 40 份 TIS evidence 和 32 份 debug train 文件正常落盘。重算覆盖 12 个样本、353315 个
  有效 token，所有 logprob/importance ratio finite，CP=8 packed position coverage 完整，
  current/train logprob 逐 token 差异为 0。
- 同 checkpoint 训推指标为 TIS mean `0.995992`、p50 `0.999996`、p95 `1.000934`；
  train/rollout logprob 绝对差 mean `0.175132`、p50 `1.08e-5`、p95 `0.982345`。该差异不按
  逐 token 完全相等验收；结合第 6 步的关闭 YaRN 基线及已测 NPU 重复运行波动，未观察到
  YaRN 特有的训推退化。

本轮最大实际 total length 为 156055、最大 logit position 为 156053。因此第 7 步证明的是
“在原生 262144 容量配置下，只改变位置编码后，完整 rollout、训练、权重同步和落盘链路可
执行且数值稳定”，不证明已经越过原始上下文边界。第一次出现 position >= 262144 的硬证据
仍由第 8 步约 270K 测试提供。

### 8.11 第 8 步实测结果：首次越过 262144 边界

完整 VIME 运行 `qwen36_yarn_step8_270k_20260915_023428` 把 training、rollout 和 vLLM
容量统一为 `270336`，并在生产 TP=2、CP=8、EP=8 拓扑完成 2/2 rollout group 和一个训练
step。rollout success rate 为 1.0，rejected/top-up/abort/requeue 和 truncated ratio 均为 0；
训练 `loss=0.0030776951`、`grad_norm=13.6021341`，均 finite。探针介入前的 16 份 evidence
覆盖 87803 个有效 token，packed position 覆盖完整，current/train 差异为 0；该批自然生成
样本最大 position 为 93502，不能单独证明已跨界。

随后使用不注册 Polar、不进入 VIME sleep/weight-update 生命周期的独立 TP=2 vLLM-023
进程，发起 262145 prompt token + 1 generation 的精确边界请求。服务端返回 HTTP 200，
usage 为 `prompt=262145/completion=1/total=262146`，无截断或 position/index 错误。因此 rollout
侧首次跨界通过。此前在 VIME 权重生命周期中直接插入 probe 导致 sleeping KV engine 退出并
使后续 update_weights 超时，那份 500 结果只记录为测试生命周期污染，不计作 YaRN 失败。

### 8.12 300K 外推精度最终验收

2026-09-15 将最终范围收敛到 `300000` 后，完成以下六类互相独立的证据：

1. `output/qwen36_yarn_math_oracle_300k.json`：在位置 `0, 262143, 262144, 270335,
   299999` 上，Megatron 与 vLLM-023 相对 Transformers oracle 的 inverse frequency、
   attention scale、cos/sin 和 vLLM Q/K apply 最大绝对误差均为 0。该结果直接验证 YaRN
   公式和 vLLM MRoPE 修复，不依赖整模 loss。
2. `output/qwen36_yarn_npu_oracle_300k.json`：同一批位置在 Ascend NPU 上，普通 RoPE 的
   FP32 大相位 cos/sin 最大误差为 `0.007517/0.007805`；YaRN 为
   `0.008559/0.008887`，恰好由普通 RoPE 基线乘 attention scale `1.138629` 覆盖，没有
   额外 YaRN 漂移。BF16 Q/K apply 最大误差为 `0.0234375`，低于 Megatron 现有 BF16
   容差 `0.05`；非 rotary 192 维逐值完全不变。
3. 独立 vLLM-023 TP=2 实算使用来自真实 rollout 的确定性多样 token 序列，299999 prompt
   token 含 6424 个不同 token，再生成 1 token。HTTP 200，usage 精确为
   `299999 + 1 = 300000`，无静默截断，耗时 41.00 秒。原始结果保存在
   `output/qwen36_yarn_300k_boundary_probe_299999_diverse.json`。
4. 训练侧运行 `qwen36_yarn_300k_cp8_20260915_0346` 使用 TP=2、CP=8、EP=8 和两个严格
   300000-token 样本；响应段分别含 3439/4260 个不同 token。两个 microbatch 的完整
   forward/backward、梯度归约和 optimizer step 均完成，`loss=0.0271764472`、
   `grad_norm=7.3405066`，无 OOM/Traceback。16 份 evidence 覆盖全部 598976 个有效响应
   token，最大 logit position 为 299998，CP8 position coverage 无缺失、重复或错位，所有
   train/current/rollout logprob 和 importance ratio finite，current/train 逐 token 差异为 0。
5. 先做了一个错误口径的诊断实验：独立 vLLM-023 对两条人工周期拼接的完整
   300000-token 序列执行 `echo + prompt_logprobs=0`，再用这些 prefill logprob 做训练回放。
   这不是正常 rollout 的 on-policy decode 分布：人工 response 的平均 vLLM logprob 约为
   `-2.71`，其中 15.88% token 的 logprob 小于 `-8`。该实验的 train/rollout abs diff mean
   为 `0.626264`，明显高于无 YaRN 正常训练约 `0.02` 的量级和本项目 `<=0.05` 门槛，不能
   判为精度通过。它只保留为低概率 off-policy token 压力诊断，产物位于
   `output/qwen36_yarn_300k_tis_cp8_20260915_0538/`。
6. 为建立正确口径，先在 32768-token prompt 上让 vLLM 实际 decode 512 token，再将完全
   相同的 prompt 和生成 token 用 `prompt_logprobs` 重算。token 对齐完全一致，decode 与
   prefill-rescore 的 abs diff mean 为 `0.0175703`，回到正常训练约 `0.02` 的量级；结果在
   `output/qwen36_yarn_decode_prefill_ab_32k_512.json`。这证明旧实验没有 token 错位，但
   `0.626` 由非正常 response 分布放大，不能替代真实 rollout 验收。
7. 最终使用 `/workspace/vllm-023` TP=2，以两条 262144-token prompt 分别真实 decode
   37856 token。两次请求都以 `finish_reason=length` 精确得到 `262144 + 37856 = 300000`
   token，耗时 `600.63/600.30s`，所有 75712 个 decode logprob finite；两条 response 的
   logprob mean 分别为 `-0.004733/-0.001554`。replay 与摘要分别保存在
   `output/validation_inputs/qwen36_yarn_300000_2samples_vllm_decode.pt` 和
   `output/qwen36_yarn_300k_vllm_decode_summary.json`。
8. 使用真实 decode replay 运行 `qwen36_yarn_300k_true_decode_tis_cp8_20260915_0700`。
   TP=2、CP=8、EP=8 的两个 microbatch 完成 forward/backward 和 optimizer step，在线
   `train/loss=-7.69796e-5`、`grad_norm=0.233202`，无 OOM/Traceback。16 份 evidence 覆盖
   75712 个 response token，最大 total length 300000、最大 logit position 299998；CP8
   position coverage 完整，current/train logprob 逐 token差异为 0，全部数值 finite。

第 4 项原始 300K training replay 的 rollout logprob 是把较短真实 response 的 token/logprob
周期扩展到 300K；第二周期之后的 logprob 不再对应新上下文，因此那一轮的 train/rollout
diff 和 TIS 不能作为训推精度证据。第 5 项虽然重新计算了对应上下文的 logprob，但目标
token 仍是人工拼接的极低概率 off-policy token，也不能代表正常 rollout 分布。只有第 7、8
项的真实 decode replay 与训练回放是最终精度验收口径。

真实 decode 对照覆盖的 75712 个 token 几乎全部处于外推区：目标 token position 为
`262144..299999`，对应 logit position 为 `262143..299998`。全局 train/rollout logprob
绝对差为 mean `0.000958958`、p50 `2.38e-7`、p95 `6.77e-5`、p99 `0.000763`；低于
项目 `<=0.05` 门槛，也低于无 YaRN 正常训练约 `0.02` 的参考量级。importance ratio mean
为 `1.0002768`、p50 `1.0`、p95 `1.0000056`、p99 `1.0001599`，TIS clip fraction 为
`0.0001189`。少数 outlier 使 abs diff max 达到 `3.6911`，但没有影响均值和高分位门槛，
应在后续长跑中继续监控而不是据此否定总体一致性。

所以本轮的精确结论是：在已测 Qwen3.6 checkpoint、静态 YaRN fingerprint 和生产
TP2/CP8 拓扑下，YaRN 从 262144 外推到 300000 的数学、容量、训练执行和真实 decode
训推精度均通过现有验收门槛。该结论不等同于对所有 prompt/checkpoint/硬件路径作绝对证明；
上线长跑仍需持续观察分位数与 outlier。最终逐 token 汇总保存在
`output/qwen36_yarn_300k_true_decode_tis_cp8_20260915_0700/evidence_summary.json`，外推位置
分桶保存在同目录的 `tis_boundary_analysis.json`。

vLLM 的 OpenAI completion 接口即使 `max_tokens=0` 也会内部保留一个 generation token，
因此评分服务容量设为 `300001`，被评分 prompt 仍严格为 300000 token。默认 16384-token
prefill chunk 的全词表 FP32 log-softmax 峰值过高，评分时将测试 launcher 的
`max_num_batched_tokens` 降到 4096；该调整只改变分块粒度，不改变上下文、位置或 logprob
数学定义。

范围变更前做过一次 524K 容量观察：独立 vLLM 请求成功，但单 token 重复训练样本在 MoE
backward 触发 OOM。它既有非真实路由偏置，又超过当前 300K 范围，只保留为非门槛容量
观察，不计作 YaRN 精度失败，也不再继续 524K/1.01M 测试。

## 9. 上线验收门槛

以下条件全部满足后才进入长跑 RL：

- 已拿到 SFT 的实际 YaRN 配置/启动日志，且 fingerprint 与 training、rollout 完全一致；
- Megatron 参数映射测试通过；
- rotary dimension 明确为 64；
- Transformers 数值 oracle 测试通过；
- Q/K 非 rotary tail 保持不变；
- checkpoint 加载无 missing/unexpected keys；
- vLLM engine 真实环境包含 long-max-model-len 开关；
- CP=1/8 结果符合改动前建立的并行误差基线；
- 若启用 MTP，MTP on/off 的对应验证全部通过；
- 训推 logprob/TIS mismatch 不劣于改动前基线；
- 262K、270K 和 300K 外推精度阶段通过；524K/1.01M 已从本轮范围取消；
- 5-10 步短 RL run 无 KL、entropy、loss 或 grad norm 异常。

## 10. 发布与回滚

推荐 feature 默认关闭，由目标 Qwen3.6 RL launcher 显式开启。发布时记录：

```text
VIME commit SHA
Megatron-LM commit SHA
MindSpeed commit SHA（即使无改动也记录）
checkpoint path/revision
SFT YaRN fingerprint
training/rollout resolved fingerprint
```

回滚顺序：

1. 停止新任务；
2. 在 VIME launcher 关闭 `FEAT_YARN` 和 vLLM long-context 开关，恢复原有 HF override/max length；
3. Megatron 的上游能力回移与 dimension fix 可以保留，因为 feature 关闭时不进入 YaRN 路径；
4. 如果发现通用回归，再回滚对应 Megatron commit；
5. 已用错误位置编码产生的 RL checkpoint 不与正确配置续训混用。

该结构使运行侧可以先回滚，而不要求两个仓库同时做破坏性 reset。

## 11. 风险与待确认项

### 上线前阻塞项

- 获取实际 SFT checkpoint 的 YaRN 参数和启动日志；若 `factor` 或 beta/mscale 与官方示例不同，必须以 SFT 为准。
- 确认生产 RL 是否启用 MTP；这决定是否需要第 3 个功能 commit。
- 确认 1.01M 是单样本 token 上限、prompt 上限还是 prompt + generation 总上限，并统一四处 capacity 配置。

### 已知风险

- 上游 `YarnRotaryEmbedding` 的 partial rotary 缺口如果不修，会正常运行但产生错误位置编码，是最高精度风险。
- 只在 shell export vLLM 环境变量可能无法到达 Ray actor/engine 子进程，会在超长请求时才暴露。
- 静态 YaRN 会改变短上下文行为，因此“131K 小于 262K，不需要启用”的判断不成立。
- CP=8 的风险来自 position/packing 接线，不来自 YaRN 算法本身。
- vLLM MRoPE 的缓存扩容长度不能复用为 YaRN correction range 的原始上下文长度；旧
  vLLM 0.23 路径会静默产生错误频率而不报错。
- MTP + YaRN 尚缺上游现成验收结论，必须保持独立 commit 和独立开关。
- 多模态 MRoPE 不在本方案保证范围内。

## 12. 参考源

### Qwen 与 Transformers

- Qwen3.6-35B-A3B 官方模型页，Ultra-long texts 与 vLLM HF overrides 示例：
  <https://huggingface.co/Qwen/Qwen3.6-35B-A3B#processing-ultra-long-texts>
- Transformers YaRN 参数计算实现，正式测试时使用目标环境已安装版本作为 oracle：
  <https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py>

### NVIDIA Megatron-LM

- 当前上游通用 YaRN embedding：
  <https://github.com/NVIDIA/Megatron-LM/blob/7613fd7a9508833d5895dd35d2847a1b8571fd0d/megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py>
- YaRN HybridModel 接入及 parser `yarn` 选项：
  <https://github.com/NVIDIA/Megatron-LM/commit/4d6cdd52f2b8ec29ad6d761cceb60f1196583f2e>
- 当前上游 YaRN 参数到 config 映射的位置迁移，可据此确认 mapper 的完整字段：
  <https://github.com/NVIDIA/Megatron-LM/commit/522a9ddd102a35c8ad67d46a055c5a8385b87fa1>

### Ascend MindSpeed / MindSpeed-MM

- MindSpeed 通用 RoPE/YaRN patch，用于确认不采用的维度计算路径：
  <https://github.com/Ascend/MindSpeed/blob/f71a16eb37df13af5177dd637900a2ead781935e/mindspeed/core/models/common/embeddings/rotary_pos_embedding.py>
- MindSpeed-MM Qwen3.6 示例目录：
  <https://github.com/Ascend/MindSpeed-MM/tree/6044eb8cca150648659e1b336d9775e46279486b/examples/qwen3_6>
- MindSpeed-MM FSDP Qwen3.5 的 Transformers RoPE 初始化参考，可作为额外数值 oracle，不作为 VIME 迁移目标：
  <https://github.com/Ascend/MindSpeed-MM/blob/6044eb8cca150648659e1b336d9775e46279486b/mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py>

## 13. 最终实施原则

本次适配的边界应保持清晰：

- 数学公式尽量完全使用 Megatron/Transformers 上游；
- 只修复 Qwen partial rotary 所需的最小维度问题；
- VIME 负责配置一致性和进程环境可靠传递；
- MindSpeed 不新增一套重复 YaRN 路径；
- CP=8、MTP 和 1.01M 分别作为组合验证、条件能力和容量扩展处理；
- 在任何长跑前，用独立 oracle 和训推 logprob 对齐证明“配置相同且计算相同”。
