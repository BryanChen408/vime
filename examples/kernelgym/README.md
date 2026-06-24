# KernelGym 算子生成 RL（vime 适配）

用 RL 训练大模型生成高性能 **Triton 算子**：模型读 PyTorch 参考实现 → 产出 `ModelNew` Triton 内核 → 由 [kernelGym](https://github.com/...) 服务编译/校验正确性/测加速比并给出奖励 → 多轮把验证反馈喂回模型迭代改进。

本目录是从 `rllm-lilac` 的 KernelGym 适配迁移到 vime（slime/Megatron + vLLM）的实现。

## 与 vime 的集成方式

vime 不需要单独的 agent 框架；本例通过 vime 的自定义接口接入（见 `docs/en/get_started/customization.md`）：

- `--custom-generate-function-path examples.kernelgym.rollout.generate` —— 每个样本一个多轮生成-验证回合。
- `--custom-config-path examples/kernelgym/kernelgym_config.yaml` —— 注入 `max_turns`、`rollout_interaction_env_path`、`kernelgym.*` 配置。
- **奖励由 rollout 直接写入 `sample.reward`**，利用 vime「custom generate 已填 reward 则不覆盖」的机制（`vime/rollout/vllm_rollout.py`），因此**无需** `--rm-type` 或 `--custom-rm-path`。

## 文件结构

| 文件 | 作用 | 对应 rllm 来源 |
| --- | --- | --- |
| `reward.py` | 三种奖励策略（纯函数，可单测） | `KernelGymEnv.calculate_reward_*` |
| `kernelgym_client.py` | 异步 HTTP 客户端 + payload 构造 + preflight | `_HybridHttpWorker` |
| `env_kernelgym.py` | 多轮交互 env + prompt 模板 + 代码提取 | `KernelGymEnv` + `KernelAgent` |
| `rollout.py` | 多轮 rollout（纯文本，改自 geo3k 多轮例子） | rllm trainer 的 agent-env loop |
| `prepare_data.py` | rllm jsonl → vime prompt-data | `prepare_kernelbench_data.py` |
| `kernelgym_config.yaml` | 运行/奖励配置 | `reward_model.*` 配置块 |
| `run_kernelgym_grpo_npu.sh` | 8 卡 NPU 训练脚本 | `train_kernelgym_megatron_*.sh` |
| `tests/test_kernelgym.py` | CPU 单测（19 例） | — |

## 快速开始

### 1. 启动 kernelGym 验证服务

在本节点（或可达的远程节点，如 npu2）启动：

```bash
cd /path/to/kernelGym-NPU && ./start_all_with_monitor.sh
curl -s http://127.0.0.1:8002/health   # 确认 {"status":"healthy",...}
```

把 `kernelgym_config.yaml` 里的 `kernelgym.server_url` 设为该服务的 `host:port`（默认 settings 是 `10907`，实际部署常用 `8002`，以 `.env` 的 `API_PORT` 为准）。

### 2. 准备数据

```bash
python -m examples.kernelgym.prepare_data \
  --input /home/robomaster/Research/rllm-lilac/data/drkernel_rl_data.jsonl \
  --output examples/kernelgym/data/kernelgym_train.jsonl
# 验证集
python -m examples.kernelgym.prepare_data \
  --input /home/robomaster/Research/rllm-lilac/data/kernelbench_val.jsonl \
  --output examples/kernelgym/data/kernelgym_val.jsonl
```

每条输出形如：

```json
{"prompt": "```python\n<reference_code>\n```",
 "label": "<problem_id>",
 "metadata": {"reference_code": "...", "entry_point": "Model",
              "backend": "triton", "problem_id": "..."}}
```

`prompt` 携带参考代码（供 vime 估算 prompt 长度）；env 在 rollout 时从 `metadata` 重建 system+user 对话并发起验证（system 提示词在代码里，不在数据里）。

### 3. 启动训练

```bash
bash examples/kernelgym/run_kernelgym_grpo_npu.sh
```

## GPU 拓扑：8 卡训练 + 4 推 4 验证

单台 8 卡 Ascend NPU 节点：

- **vime** 在 8 卡上 **colocate** 训练 + rollout 推理（`--vllm-enable-sleep-mode` 在训练/生成阶段相互 offload）。
- **kernelGym worker** 在**同一节点常驻**。算子验证显存占用很小，可与 vLLM 推理在同卡共存，无需时分。
  - 若要「4 推 4 验证」的物理隔离：kernelGym 的 `.env` 设 `GPU_DEVICES=[4,5,6,7]`，vLLM 仍用全部 8 卡（或限定 0-3）。因 kernelGym 占显存少，二者共存即可。
- 两者只通过 **HTTP** 解耦（`server_url`），所以验证服务放本机或远程（如 npu2）都行。

训练脚本默认值（均可用环境变量覆盖）：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `ACTOR_GPUS` / `ROLLOUT_GPUS` | 8 / 8 | 相等即 colocate |
| `TRAIN_TP` | 4 | 训练张量并行 |
| `ROLLOUT_TP` | 4 | 推理张量并行（每个 vLLM 引擎） |
| `MODEL_NAME` | `Qwen3-8B` | `scripts/models/<name>.sh` 提供架构参数 |
| `--vllm-gpu-memory-utilization` | 0.55 | 留余量给同机 kernelGym 的 kernel 运行 |

> 换 30B-A3B（MoE）：`MODEL_NAME=Qwen3-30B-A3B` 并参考 `scripts/run-qwen3-30B-A3B-npu.sh` 调整 EP/ETP 等参数。

## 配置项（`kernelgym_config.yaml`）

- `max_turns`：多轮上限（rollout 与 env 共用）。
- `kernelgym.server_url`：kernelGym 服务地址。
- `kernelgym.stop_on_correct`：产出正确内核即结束回合（省验证开销）。
- `kernelgym.reward_aggregation`：`best`（取多轮最优）或 `last`。
- `kernelgym.reward.reward_func_name`：奖励策略，三选一：
  - `calculate_reward_weighted`（默认）：`w_c·correct + w_p·(speedup>1+eps)`，可加 coverage。
  - `calculate_reward_speedup`：`w_c·correct + w_p·clamp(speedup)`。
  - `calculate_reward_like_kernel`：按 speedup 分档（0.2~1.0）+ 编译/正确性惩罚。
- 超时不变量：`client_timeout >= server_timeout`（代码会自动纠正并告警）。

## 与 rllm 版本的对应关系

| rllm-lilac | 本实现 | 说明 |
| --- | --- | --- |
| `KernelGymEnv.step` / `compute_reward` | `env_kernelgym.KernelGymEnv.step`（async） | 改为异步以适配 vime rollout |
| `KernelAgent` 的 prompt/解析 | `env_kernelgym` 的 `SYSTEM_PROMPT` / `REVISION_USER_TEMPLATE` / `extract_kernel_code` | 提示词与代码提取逻辑保留 |
| `_HybridHttpWorker.submit_and_poll` | `kernelgym_client.submit_and_poll` | `httpx`(sync) → `aiohttp`(async) |
| stepwise advantage（逐轮 reward） | 单轨迹 reward（多轮取 `best`/`last`） | 适配 GRPO 的轨迹级优势；如需逐轮可后续扩展 |

## 测试与验证

```bash
python -m pytest examples/kernelgym/tests/test_kernelgym.py -q
```

- **19 个 CPU 单测**覆盖：三种奖励策略、coverage、preflight、kernel 提取、`ModelNew` 修补、多轮 env step（mock 验证器）、`best` 聚合、preflight 失败惩罚、`max_turns` 截断、数据转换。无需 NPU / kernelGym 服务。
- **API 通路已在 npu2 实测**：`POST /evaluate` 接受本实现的全部 payload 字段；`/results` 返回的 `status/compiled/correctness/speedup/decoy_kernel/metadata/error_message` 与 `reward.py` 解析完全一致。

## 已知限制与未来优化

- **端到端训练需 NPU 环境**：本机无 NPU 时仅做逻辑/接口验证；真实 8 卡训练请在 Ascend 节点跑。
- **暂不支持 routing replay**：多轮 rollout 下 `--use-rollout-routing-replay` 会显式报错（routed-experts 的逐轮累积未实现）。dense 模型与不开 replay 的 MoE 不受影响。
- **train/eval 的 `is_valid` 为全局配置**：如需验证集强制开启 decoy 检测，单独跑一次 `kernelgym.is_valid=true` 的评估。
- **更丰富的反馈**：kernelGym 还返回 `repair_context` / `npu_diagnosis`（含瓶颈诊断），当前 `format_feedback` 只用 `status/compiled/correctness/speedup/error`；可扩展为把诊断要点也喂回模型以加速修复。
