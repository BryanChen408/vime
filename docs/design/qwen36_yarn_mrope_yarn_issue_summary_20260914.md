# MRoPE + YaRN 数值问题调查总结

更新时间：2026-09-14

本文用于和 VIME、Polar、vLLM、Megatron 相关开发者讨论 Qwen3.5/Qwen3.6 类模型在 `MRoPE + YaRN` 配置下的数值一致性问题。

本仓库保存的可应用 vLLM 修复及其运行前置见
[Qwen3.6 YaRN external patch bundle](patches/qwen36_yarn_20260915/README.md)。

## 一句话结论

旧版 vLLM 的 `MRotaryEmbedding + YaRN` 组合存在确定性的数值错误：同一个 `max_position_embeddings` 字段同时被当作 YaRN 的“原始预训练上下文长度”和 MRoPE 的“cos/sin 缓存长度”，导致 correction range 算错，并且缓存长度又被 YaRN 的 scaling factor 重复放大。

这不是 VIME 训练逻辑、Polar 配置或随机训练波动导致的问题。纯推理和 rollout 只要走 `MRoPE + YaRN`，同样会受影响。

## 问题配置

本次以 Qwen3.5 的实际 YaRN 配置为例：

```text
rope_type = yarn
rope_theta = 10_000_000
partial_rotary_factor = 0.25
head_size = 256
rotary_dim = 64
factor = 4
original_max_position_embeddings = 262144
mrope_section = [11, 11, 10]
mrope_interleaved = true
beta_fast = 32
beta_slow = 1
truncate = true
```

## 已确认的错误链路

### 1. YaRN 的两个长度语义

YaRN 中至少有两个不同的长度：

```text
original_max_position_embeddings = 262144
    YaRN correction range 使用的原始预训练长度

cache length = original length * factor = 1048576
    推理时 cos/sin cache 需要覆盖的长度
```

YaRN 作者的参考实现明确将两者分开：[jquesnelle/yarn](https://github.com/jquesnelle/yarn/blob/master/scaled_rope/modeling_llama_together_yarn.py)。

### 2. vLLM 旧实现把两个语义混在一起

vLLM 的 YaRN 入口先创建 `MRotaryEmbedding`，并传入原始长度 `262144`：

```text
vllm/model_executor/layers/rotary_embedding/__init__.py
```

但旧版 `MRotaryEmbedding` 随后做了：

```python
self.cache_max_position_num = max_position_embeddings * 4
super().__init__(..., self.cache_max_position_num, ...)
```

因此父类中的：

```python
self.max_position_embeddings
```

已经变成 `1048576`，不再是 YaRN 所需的原始长度。

随后 MRoPE 直接复用了 YaRN 的两个方法：

```python
YaRNScalingRotaryEmbedding._compute_inv_freq(self, ...)
YaRNScalingRotaryEmbedding._compute_cos_sin_cache(self)
```

而 YaRN 方法会把 `self.max_position_embeddings` 同时用于 correction range 和 cache 长度计算，于是出现两次错误：

1. correction range 使用 `1048576`，而正确值应为 `262144`；
2. YaRN 再乘一次 `factor=4`，得到错误的 `4194304` cache 行数，而不是 `1048576`。

当前 vLLM 主干仍可看到这条实现路径：[mrope.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding/mrope.py)、[yarn_scaling_rope.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.py)。

## 数值证据

对上述实际配置直接计算：

| 检查项 | 正确值 | 旧 vLLM 值 |
|---|---:|---:|
| YaRN correction range | `(14, 22)` | `(16, 24)` |
| cos/sin cache 行数 | `1,048,576` | `4,194,304` |
| 不同的 inverse-frequency 分量 | `0 / 32` | `9 / 32` |
| inverse-frequency 最大绝对误差 | `0` | `5.92927e-05` |
| position 131071 的 cos 最大误差 | `0` | `2.23576` |
| position 262143 的 sin 最大误差 | `0` | `2.25498` |

这不是浮点容差问题。cos/sin 的误差已经达到接近完整数值范围，必然改变 attention 的输入。

另外，attention scale 本身仍为正确值 `1.138629436111989`；问题集中在 correction range、inverse frequency 和 cos/sin cache。

## 参考实现的可信边界

不能简单地说“Transformers/Megatron 就是绝对 golden”。应区分如下：

| 来源 | 在本调查中的定位 |
|---|---|
| YaRN 论文和作者参考实现 | 算法语义的最高层依据 |
| Hugging Face Transformers | 独立、成熟的模型配置实现，可作为数值 oracle |
| NVIDIA Megatron | 训练侧实现，是被测实现，不是数学意义上的 golden |
| vLLM | rollout/推理侧实现，是被测实现 |

YaRN 论文说明了 RoPE 扩展方法和 correction range 的含义：[YaRN 论文](https://arxiv.org/abs/2309.00071)。

Transformers 明确用 `original_max_position_embeddings` 计算 correction range：[Transformers YaRN 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py)。

Megatron 当前实现也保留了独立的 `original_max_position_embeddings` 字段，并将其用于 correction range，见：

```text
Megatron-LM/megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py
```

因此，本次“旧 vLLM 错误”的判定不依赖盲信某一个库：

1. YaRN 作者参考实现使用原始长度；
2. Transformers 使用原始长度；
3. Megatron 使用原始长度；
4. 旧 vLLM 的 MRoPE 路径使用了扩容后的 cache 长度；
5. 代入同一公式后得到可复现的不同数值。

这是足以判定旧 vLLM 逻辑错误的交叉证据。

但这不等于已经证明整条 Ascend 推理链或整模型输出“绝对正确”。仍需单独验证 NPU fused kernel 的 Q/K 输出和端到端 logits。

## 上游调查结果

### MRoPE + YaRN 的引入

该功能由 [vLLM PR #25384](https://github.com/vllm-project/vllm/pull/25384) 引入，并于 2025-09-23 合入主干。

该 PR 的实现方式就是让 MRoPE 复用 YaRN 的 `_compute_inv_freq` 和 `_compute_cos_sin_cache`。PR 页面上曾出现自动审查警告，指出 MRoPE 与 YaRN 的交互可能产生 incorrect rotary embeddings；但详细评论目前无法正常加载，不能断言该评论是否完整指出了本次长度混用问题。

### 后续的 `truncate` 修复

2026-02，Qwen3.5 长上下文启动时暴露了 `MRotaryEmbedding` 缺少 `truncate` 属性：[issue #35056](https://github.com/vllm-project/vllm/issues/35056)。

上游通过 [PR #35080](https://github.com/vllm-project/vllm/pull/35080) 补齐了 `truncate` 参数和属性。该修复解决了启动崩溃，但验收重点是服务启动、KV cache 容量和长上下文可加载性，没有验证 Transformers/Megatron 的 inverse frequency 或 cos/sin 数值一致性，因此没有解决本次静默数值错误。

### 是否已有同一个公开 issue

截至 2026-09-14，没有找到专门描述以下完整问题的公开 issue/PR：

```text
MRoPE 的 cache_max_position_num 被错误地用于 YaRN correction range，
并且 factor 被重复应用到 cache 长度。
```

因此不能说上游已经修复，也不能说绝对没人发现。可以确认的是：

1. 上游主干当前仍保留该实现结构；
2. 上游曾有过“可能产生错误 rotary embedding”的审查警告；
3. 已公开合入的 #35080 只修复了 `truncate` 崩溃，不是本次数值修复。

## 为什么此前纯推没有暴露明显能力问题

这与数值错误并不矛盾，原因有四个。

### 1. 之前的纯推确实走过 YaRN，但参数更温和

历史纯推日志 `train_qwen36_rollout9_eval_20260910-205747.log` 显示：

```text
rope_type = yarn
mrope_section = [11, 11, 10]
factor = 1.5
original_max_position_embeddings = 262144
max_model_len = 389120
```

这已经是 `MRoPE + YaRN` 路径，但不是当前 Qwen3.6 RL 目标使用的 `factor=4.0`。在 `factor=1.5` 下，旧实现的 correction range 仍然是 `(16,24)` 而不是 `(14,22)`，但 inverse-frequency 最大误差约为 `2.64e-05`，小于 `factor=4` 时的 `5.93e-05`。

### 2. `max_model_len` 是容量上限，不等于实际请求长度

历史日志中的 `max_model_len=389120` 只说明服务允许这么长的请求。日志主要记录了：

```text
POST /v1/chat/completions -> 200 OK
throughput / Running / KV cache usage
```

没有记录固定长上下文下的 logits、top-1 token、KL 或能力分数。因此它证明了“服务能运行”，没有证明“长上下文位置编码与参考实现一致”。

### 3. 短上下文的相位误差可能不明显

对 `factor=1.5` 的旧实现和正确实现直接比较，cos/sin 最大绝对误差约为：

| position | cos 最大误差 | sin 最大误差 |
|---:|---:|---:|
| 1 | `0` | `2.74e-05` |
| 1000 | `0.011` | `0.026` |
| 8192 | `0.162` | `0.177` |
| 32768 | `0.643` | `0.762` |
| 131071 | `1.910` | `1.019` |

如果人工测试的 prompt 和生成长度主要在几百到几千 token，模型仍可能生成连贯文本；错误会在更长的位置逐渐积累，而不是一启动就产生乱码。

### 4. 这是系统性频率偏移，不是随机噪声

旧实现不是把输入张量随机破坏，而是把 32 个频率分量中的 9 个按错误的 ramp 边界混合。结果更像“使用了另一套近似位置编码”：

- 生成接口仍然返回 200；
- 常规短题可能仍然答对；
- 人工样例可能看不出明显异常；
- 但长上下文位置、logits、token 选择和 rollout 分布会发生偏移。

所以此前纯推的正确结论只能是“服务可用、输出未明显崩坏”，不能是“YaRN 数值正确”或“精度已通过”。

## 对当前系统的影响

### 会受影响的路径

- vLLM 纯推理：启用 `MRoPE + YaRN` 时受影响；
- Polar rollout：如果使用该 vLLM core，同样受影响；
- VIME YaRN RL 训练：rollout 侧权重/采样分布会受影响，不能只看训练进程是否跑通；
- 长上下文位置越高，误差越明显。

### 不属于本问题的路径

- 普通标准 RoPE；
- 不含 `mrope_section` 的普通 YaRN；
- Polar 的 enforcement、session 派发和 VIME gateway 协议问题；
- VIME 的权重边界 pause/drain/resume 纪律。

Ascend 侧的 `AscendMRotaryEmbedding` 继承 vLLM core 的 `MRotaryEmbedding`，见：

```text
vllm-ascend/vllm_ascend/ops/rotary_embedding.py
```

所以核心长度语义修复应在 vLLM core 完成，不应在 Polar 本地配置中绕过或修改。

## 当前本地修复

当前本地修复位于 `/workspace/vllm-023`，尚未提交：

```text
vllm/model_executor/layers/rotary_embedding/mrope.py
vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.py
tests/kernels/core/test_mrope_yarn.py
```

修复原则：

1. 单独保存 `original_max_position_embeddings`；
2. 单独计算并保存 `cache_max_position_num`；
3. correction range 只使用 original length；
4. cos/sin cache 只扩展一次；
5. 保留原有非 YaRN MRoPE 行为。

已完成的本地验证：

```text
vLLM MRoPE+YaRN targeted tests: 2 passed
Megatron YaRN/Transformers parity tests: 10 passed
VIME YaRN validation tests: 18 passed
ruff/check/diff check: passed
```

修复后的 Qwen3.5 配置实际结果：

```text
cache shape = (1048576, 64)
inverse frequency 与 Transformers 完全一致
attention scale 与 Transformers 完全一致
多个位置的 cos/sin 最大绝对误差 = 0
```

## 当前验证结论边界

### 已经可以下的结论

1. 旧 vLLM MRoPE+YaRN 存在确定性数值错误；
2. 旧 Stage1 的“rollout+train 跑通”不能作为 YaRN 精度通过；
3. 修复后的 CPU/reference 数值已和 YaRN 作者语义、Transformers、Megatron 对齐；
4. 纯推理和 rollout 都需要使用修复后的 vLLM core；
5. Polar 不需要通过修改本地 YAML 来修复这个问题。

### 还不能下的结论

1. 还不能仅凭 CPU 数值测试宣布 Ascend fused rotary kernel 端到端正确；
2. 还不能宣布修复后的整模型 logits、loss、采样分布与训练侧完全一致；
3. 还不能把旧 Stage1/Stage2 结果作为修复后的精度证据。

## 建议的后续验收顺序

1. 在实际 Polar 运行环境部署修复后的 vLLM core，并记录实际 commit/镜像来源；
2. 运行 CPU reference 与 Ascend fused kernel 的固定 Q/K 对比，覆盖 `position=0, 1, 32768, 131071, 262143, 524287, 1009999`；
3. 重新运行 Stage1，保存 rollout 请求数、完成数、token 数、loss 和 checkpoint；
4. 运行 Stage2，比较修复前后的 logits/loss/rollout 统计，不再只依赖 health/metrics 日志；
5. 如要对上游提交 issue/PR，应附上最小复现、`(14,22)` vs `(16,24)` correction range、cache 行数和 Transformers oracle 数值。

## 讨论时需要明确的问题

1. 上游是否接受“original length 与 cache length 必须分字段”的修复方向；
2. MRoPE 的 4x cache 是否应保留为兼容行为，还是统一改成 `original * max(4, factor)`；
3. `mrope_section`、interleaved、partial rotary 和 Ascend fused kernel 是否需要独立 CI parity test；
4. vLLM 后续 YaRN 相关 PR 是否应强制加入 Transformers/作者公式的数值 oracle，而不是只验证服务启动。
