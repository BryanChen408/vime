# Qwen3.5-0.8B MathRL（DAPO-Math GRPO）NPU 训练调试记录

> 机器:`ssh mynpu2`(主机 `cntrain21`,root)
> 目标:利用**后四卡**,用 **Qwen3.5-0.8B**(用户称 "0.9B",modelscope 上仅有 0.8B)跑通 vime 的 **MathRL**(DAPO-Math GRPO)训练。
> 用户约束:**colocate 共卡 + 无 vLLM sleep-mode**(此配置在 Qwen3-30B-MoE 上已验证可成功运行)。

---

## 1. 环境概况

| 项 | 值 |
|---|---|
| NPU | Ascend **910B3 ×8**,每卡 64 GB HBM。卡 0 已被占用(~58 GB),**卡 4-7 空闲** → 用作"后四卡" |
| Python | 3.11.15(`/usr/local/bin/python`) |
| torch / torch_npu | 2.10.0+cpu / 2.10.0 |
| vLLM | 0.21.0 + `vllm-ascend`(`vime-adapter` 分支,已 `pip -e` 安装) |
| ray | 2.55.1 |
| vime | `/workspace/vime`,分支 `npu`(已 `pip -e` 安装) |
| Megatron-LM | `/workspace/Megatron-LM`(`qwen36_vime_adapt`) |
| Megatron-Bridge | `/workspace/Megatron-Bridge-slime/src`(`megatron.bridge`,`qwen35`) |
| CANN | `/usr/local/Ascend/ascend-toolkit/set_env.sh` + `nnal/atb/set_env.sh` |

> ⚠️ 环境是当天刚搭好的:**无任何模型权重、无数据集**;现成脚本 `run_qwen08b_dapo_test_npu.sh` 是从别的 16 卡机拷来的模板,模型/数据/Megatron 路径全是别人的 home 目录,均需重写。

## 2. 关键决策

1. **模型 = `Qwen/Qwen3.5-0.8B-Base`**。探测 modelscope:`Qwen3.5-0.8B-Base` 存在(200),`Qwen3.5-0.9B` 不存在(404)。0.8B-Base 实际参数量接近 0.9B,即用户所指。它是混合线性注意力 + 门控注意力架构(`vime_plugins.models.qwen3_5` spec,`--use-gated-attention --attention-output-gate`),**不能用普通 Qwen2.5 替代**。
2. **后四卡** → `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7`(避开被占的卡 0)。
3. **colocate + 无 sleep**(用户约束):actor 与 rollout 共用同 4 卡,vLLM 常驻显存。0.8B 很小,`--vllm-gpu-memory-utilization 0.30` 即可,训练侧显存绰绰有余。
4. **数据 = `zhuzilin/dapo-math-17k`**(HF):vime 官方文档(`NPU.md`/`quick_start.md`)指定,已是 `{prompt, label}` 的 jsonl 格式,与 `--input-key prompt --label-key label` 直接匹配。
5. **加载方式**:沿用官方 `NPU.md` 的 `--hf-checkpoint + --load + --megatron-to-hf-mode bridge`(用 Megatron-Bridge 直接从 HF 目录加载到 Megatron),`rm-type math`。

## 3. 准备步骤

```bash
# 模型(modelscope,国内快)
pip install -q modelscope
modelscope download --model Qwen/Qwen3.5-0.8B-Base --local_dir /root/models/Qwen3.5-0.8B-Base   # 1.7G ✓

# 数据(hf-mirror)
HF_ENDPOINT=https://hf-mirror.com hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/datasets/dapo-math-17k    # dapo-math-17k.jsonl 10MB ✓
```

- 网络:无代理,但 modelscope(~50ms)、hf-mirror、pypi 均直连可达。注意 `curl -I`(HEAD)会卡死,加 `--connect-timeout` 即正常。

## 4. 训练脚本

`scripts/run_qwen35_08b_mathrl_npu_4card.sh`(融合官方 `NPU.md` Qwen3-4B 流程 + `run_qwen08b` 的 0.8B 超参)。相对官方 Qwen3-4B 脚本的改动:

- `source scripts/models/qwen3.5-0.8B.sh` 取 `MODEL_ARGS`(替代 qwen3-4B)
- `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7`(后四卡,替代 0-7)
- **加 `--colocate`**;actor 4 卡 + rollout 4 卡共用
- **去 `--vllm-enable-sleep-mode`**
- `--tensor-model-parallel-size 1`(0.8B 单卡即可,4 卡 DP);`--rollout-num-gpus-per-engine 1`(4 个 tp1 vLLM engine)
- PYTHONPATH 指向本机 `/workspace/Megatron-Bridge-slime/src` 与 `/workspace/Megatron-LM`
- 补 `source` CANN env、`QWEN36_CAUSAL_CONV1D_IMPL=triton`、`PYTORCH_NPU_ALLOC_CONF`

## 5. 调试记录

（按时间顺序记录每次报错与解决方案）

### 轮次 0 — `ModuleNotFoundError: No module named 'vllm_router'`
- 现象:`train.py` import 阶段崩溃,`vime/utils/arguments.py:9` 顶层 import `vllm_router.launch_router.RouterArgs`。
- 原因:`vllm-router` 是 vime 标准依赖(`requirements.txt:23` → `vllm-router>=0.1.14`,PyPI 包,来自 `github.com/vllm-project/router`),用于调度多个 vLLM engine;本机基础环境漏装。
- 解决:`pip install "vllm-router>=0.1.14" --no-deps -i https://pypi.org/simple`(清华镜像未同步该包,改用官方 PyPI;`--no-deps` 防止连带改动 vllm/torch)。装上 `0.1.14`(aarch64 wheel)。

### 轮次 1 — 后台启动"假失败"(ssh + pkill 自杀)
- 现象:重启命令返回空、日志 mtime 不更新,训练实际没起来。
- 原因:外层 ssh 命令里写了 `pkill -9 -f "vllm"`,该正则**匹配到承载本命令的 bash shell 自身命令行**(其中就含字符串 "vllm"),把自己杀了,后续启动未执行。
- 解决:外层启动命令绝不含会自我匹配的 pkill(清理交给训练脚本内部的 nohup 子进程,命令行不含 pattern,安全);用 `setsid bash script >log 2>&1 </dev/null &` 完全脱离 ssh 会话启动,避免 ray/vllm 子进程持有 ssh fd 导致 ssh 挂起、输出丢失。

### 轮次 2 — `transformers 不认识 qwen3_5`
- 现象:`AutoConfig.from_pretrained` 报 `model type 'qwen3_5' not recognized`(`megatron_parse_args` 验证 HF config 时,`arguments.py:185`)。
- 分析:Qwen3.5-0.8B 是**混合线性注意力 + 多模态**新架构(`text_config.layer_types` = `linear_attention`×3 + `full_attention` 每 4 层交替,带 MTP / 门控注意力 / mrope / vision_config),config 由 `transformers 4.57.0.dev0` 保存;本机 release **4.57.0** 不含 qwen3_5。
- 解决:`pip install transformers==5.12.1 --no-deps` + `pip install -U huggingface_hub`(→ 1.21.0,transformers 5.x 需要 `is_offline_mode`)。验证:`AutoConfig` 返回 `Qwen3_5Config`(24 层),vllm 0.21 / vime 仍可 import。vime 的 `mbridge/qwen3_5.py` 已通过 `_get_text_config()` 适配该嵌套 config。

### 轮次 3 — 初始化全程顺利(transformers 修复后)
启动后依次通过(无致命错误,仅 apex/dynamo 等无害 warning):
- **vllm-router 就绪**:`Router ready`,server on `:4077`(多 engine 调度层)
- 4 个 vLLM engine 解析出 `Qwen3_5ForConditionalGeneration`,成功加载 safetensors(含 vision 部分)
- **混合线性注意力架构被正确支持**:`patch_mamba_config` 对齐 mamba/attention page size;vllm 编译 `splitting_ops` 含 `linear_attention`/`gdn_attention_core`/`mamba_mixer`
- **colocate 权重同步就位**:`vLLMColocateWorkerExtension` 注入 NPUWorker,提供 `init_weight_transfer_engine`/`start_weight_update`/`update_weights_chunk`/`finish_weight_update` RPC(`weight_transfer_config backend='ipc'`)
- 之后进入 vllm-ascend cudagraph + triton kernel 编译(0.8B 混合架构编译耗时数分钟)

> 关于 sleep mode:日志中 `enable_sleep_mode: True`。vime 在 `vllm_engine.py:427` 当 `offload_rollout=True 且未显式关 sleep` 时会强制 `--enable-sleep-mode`;colocate 路径默认带 rollout offload。这是 colocate 共卡的标准做法(训练时让 vLLM 释放显存)。先以默认配置跑通,再评估是否需严格"无 sleep"。

### 轮次 4 — `ModuleNotFoundError: modelopt` / `transformer_engine`(megatron.bridge 不兼容 NPU)
- 现象:编译完成、vLLM sleep 成功(`Sleep mode freed 8.47 GiB`)、4 个 `MegatronTrainRayActor` 启动并到 `initialize_model_and_optimizer`,随后崩溃(`ERR99999`)。真正异常是 `MegatronTrainRayActor.init` → `from megatron.bridge import AutoBridge` 链式 import 失败:先缺 `modelopt`(NVIDIA TensorRT Model Optimizer),装上后又缺 `transformer_engine`(NVIDIA CUDA 专用,**NPU 无法安装**)。
- 根因:我从官方 **Qwen3-4B(标准 dense)** 示例照搬了 `--megatron-to-hf-mode bridge`,它走 `megatron.bridge.AutoBridge`,硬依赖 TE/modelopt 等 CUDA 生态。
- 关键发现:`--megatron-to-hf-mode` 默认是 **`raw`**(`arguments.py:138`,`choices=["raw","bridge"]`);**run_qwen08b 正是用默认 raw**(仅 `--hf-checkpoint + --ref-load`)。raw 模式用 `--spec`(`vime_plugins.models.qwen3_5`)构建 megatron 模型 + vime 自己的 mbridge 做权重转换,**绕开 megatron.bridge / TE / modelopt**。
- 解决:脚本删除 `--load` 与 `--megatron-to-hf-mode bridge`,改用 `--ref-load ${HF_CKPT}`,对齐 run_qwen08b。(过程中已装 `nvidia-modelopt 0.44 + pulp`,raw 模式不再需要,保留无害。)

### ⚠️ 关键转折 — 用户纠正:不改代码,核对 35B 脚本 + 确认库分支
此前我一度想改 `vime_plugins/models/qwen3_5.py` 的 gdn_backend 默认值 —— 错误方向。用户指出:**代码理论上不需要改**,应先核对已验证的 `run_qwen36_35b_a3b_dapo_math_npu.sh`,并确认依赖库在团队适配分支/提交上。

**库分支核对(全部正确,团队最近 4-5 天适配):** vime `npu`、Megatron-LM `qwen36_vime_adapt`(fix: NPU colocate)、MindSpeed `migrate-qwen36-gdn`(triton GDN kernels)、vllm-ascend `vime-adapter`(RL sleep-mode fix)、Megatron-Bridge-slime `qwen35`。

**35B 脚本核对 → 发现我漏的关键配置:** ① `--qwen-gdn-backend npu`(我漏了→默认 fla) ② `--ref-load` 指向 **torch_dist**(我错用 HF) ③ `--rm-type deepscaler` ④ `--vllm-compilation-config`。

### 轮次 5 — fla backend(缺 `--qwen-gdn-backend npu`)
- actor 构建 GDN 层报 `ImportError: Qwen GDN backend 'fla' requires flash-linear-attention`。
- 根因:vime 默认 `qwen_gdn_backend=fla`(GPU 版),本机只有 `fla_npu`;NPU 应走 `npu` backend(`mindspeed.ops.chunk_gated_delta_rule`/`causal_conv1d`,已验证可用)。
- 解决:脚本加 `--qwen-gdn-backend npu`(35B 脚本本就有,不改代码)。

### 轮次 6 — actor 需 torch_dist 格式(raw 模式 mcore load)
- 35B 注释:"actor loaded from converted torch_dist via native mcore load_checkpoint";`checkpoint.py:130` 断言加载 HF 仅 bridge 模式支持。raw 模式 `--ref-load` 必须是 torch_dist。
- 解决:用 `tools/convert_hf_to_torch_dist.py` 把 HF 转 torch_dist。

### 轮次 7 — convert 缺 `mbridge`
- `from mbridge import AutoBridge` → ModuleNotFound。`docker/npu_patch/README.md`(团队完整环境清单)指定 `pip install git+ISEEKYAN/mbridge.git@89eb108 --no-deps`。装上。本机手搭环境漏装的依赖都在这个 README 里。

### 轮次 8 — mbridge 不认识 qwen3_5(lazy import)
- `ValueError: Unregistered model type: qwen3_5`。vime `vime_plugins/mbridge/__init__.py` 用 lazy `__getattr__`,`import vime_plugins.mbridge` 不触发 `qwen3_5.py:29` 的 `@register_model`。
- 解决(不改 vime 代码):转换脚本用 `python -c "import vime_plugins.mbridge.qwen3_5; runpy.run_path(...)"` 显式触发注册。

### 轮次 9 — `NameError: TENorm`(wrapper 破坏 import 顺序)
- vime spec 硬编码 `use_transformer_engine=True`,megatron 需 TENorm;NPU 上 TENorm 由 mindspeed patch 成 `PTNorm`,但需 `import mindspeed.megatron_adaptor` **先于任何 megatron import**(convert 脚本头部就是这么做的)。我的 wrapper 先 import qwen3_5 → 提前触发 megatron import,mindspeed patch 未生效 → TENorm 未定义。
- 解决:wrapper 改为 `import torch_npu; import mindspeed.megatron_adaptor; import vime_plugins.mbridge.qwen3_5; runpy...`(mindspeed-first)。
- ✅ **转换成功**:`/root/models/Qwen3.5-0.8B-Base_torch_dist/`(release/*.distcp,~1.5GB)。

### 轮次 10 — 对齐 35B 重启 → 推进到权重同步,暴露 split 维度报错
训练脚本完整对齐 35B(`--qwen-gdn-backend npu` + `--ref-load torch_dist` + `--rm-type deepscaler` + `--vllm-compilation-config` + sleep mode)。这次顺利通过 actor 初始化 + vLLM ready + CUDA graph capture,**首次推进到权重同步**,然后崩:
`RuntimeError: split_with_sizes expects split_sizes to sum exactly to 6144, but got [2048, 2048, 4096]`(`gdn_param_mapping.py:279 deinterleave_gdn_conv1d`)。

### 轮次 11 — split 根因:GDN `linear_num_value_heads` 默认 32 vs 实际 16(双子智能体交叉验证)
> 此处起按用户要求**大胆并行用子智能体多角度分析**,两个 Agent 独立得出完全一致、证据严密的根因。
- raw 模式权重同步器 `megatron_to_hf/qwen3_5.py::_gdn_cfg(args)` 从 **Megatron args** 读 GDN 维度,但 `qwen3.5-0.8B.sh` 没传 `--linear-*` 参数 → `linear_num_value_heads` 落到 **Megatron 默认 32**,而 0.8B 实际 **16**。代码算 v_dim=128×32=4096 → split `[2048,2048,4096]`=8192;实际 conv_dim=128×16×... =6144 → 不匹配。
- **35B 能跑是巧合**:它真就是 32 个 value head=Megatron 默认,掩盖了这个 bug;0.8B 是 16:16 才暴露。属 model args 与 config.json 不一致,非代码 bug。
- 解法(不改 vime 代码):训练脚本 `${MODEL_ARGS[@]}` 后补 `--linear-num-value-heads 16`(连同 key-head-dim/value-head-dim/num-key-heads/conv-kernel-dim 一并钉死,对齐 config.json)。✅ **split 报错消失**。

### 轮次 12 — base 模型 + deepscaler reward 会"假 0"(子智能体实测)
- 子智能体实测:`deepscaler.py:5-10` 要求 **response** 含 `</think>`,否则静默 return 0;而 Qwen3.5-0.8B-Base 的 chat template 默认把 `<think>\n\n</think>\n\n` 放进 **prompt**(prefill),response 里没有 → 即使模型答对(`\boxed{4}`)reward 仍=0 → GRPO advantage 全 0 → 训练无信号。
- 解法(不改 vime 代码,一行):`--rm-type math` —— 判分内核与 deepscaler 相同(extract `\boxed{}` + grade_mathd/sympy),但去掉 `</think>` 门控。也正是官方 NPU.md Qwen3-4B 用的 reward。

### 轮次 13 — NPU 跨进程 IPC 共享显存导入失败(子智能体机制级定位)
- split 修复后推进更远,崩在 vLLM 接收权重:`RuntimeError: AclrtMemImportFromShareableHandle error code 507899`(`update_weight_from_tensor.py:614` → `torch_npu rebuild_npu_tensor`)。
- 子智能体机制级根因:torch_npu 有两套 IPC——原生(`AclrtIpcMemGetExportKey`)和 **VMM/expandable**(`AclrtMemImportFromShareableHandle`)。报错是后者,只在生产者显存位于 expandable 段时触发。vime 只给 vLLM 子进程 `env.pop` 掉 `PYTORCH_NPU_ALLOC_CONF`(`vllm_engine.py:360-367`,sleep 模式需要),**但没给 trainer 关** → trainer=VMM/vLLM=native,句柄不兼容;叠加后四卡(4-7)逻辑/物理设备号错位(句柄属 device_id=6,导入传 deviceId=0)。
- 解法(不改 vime 代码):训练脚本注释掉 `export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`,让 trainer 也走原生 IPC,与 vLLM 对齐。

### 轮次 14 — 关 expandable 后 IPC 换了个错(native 路径也失败)
关 expandable 让 IPC 从 VMM 路径(507899)切到原生路径,但仍崩在权重同步:
`RuntimeError: devptr INTERNAL ASSERT FAILED ... entry in cache has missing shared_ptr`(`torch_npu rebuild_npu_tensor`)。两条 IPC 路径都堵,且都伴随 `POST /wake_up?tags=weights`(sleep→wake)。

### 轮次 15 — 决定性统一机制:分配器类型必须一致 + sleep 硬绑定 vLLM 分配器(子智能体)
> fallback 子智能体给出 torch_npu IPC 的完整机制,一举解释所有现象 + 用户"无 sleep 可成功"。
- torch_npu 两条**互斥** IPC 通道:native(`AclrtIpcMemGetExportKey/ImportByKey`)vs VMM(`AclrtMem*ShareableHandle`)。**两端分配器类型必须一致**,句柄才能互通。
- **sleep mode 硬绑定 vLLM 分配器**:`vllm-ascend/platform.py:700-714` 当 `enable_sleep_mode=False` 主动加回 `expandable_segments:True`(→VMM);`camem.py:150-155` sleep 模式断言 expandable 必须关(→native)。

| 配置 | trainer | vLLM | 结果 |
|---|---|---|---|
| 轮次10-13(sleep开 + trainer expandable开) | VMM | native | ❌ 507899 |
| 轮次14(sleep开 + trainer expandable关) | native | native | ❌ missing shared_ptr |
| **用户"无 sleep 成功"(sleep关 + trainer expandable开)** | VMM | VMM | ✅ |

- 设备号偏移(`device_id=6` vs `deviceId=0`)是**症状不是根因**:无 sleep 在相同后四卡偏移下能成,vime 已用 `update_weight_from_tensor` 的设备重映射 + UUID 路由消化了偏移。
- **解法 = 复刻用户的无 sleep 路径(Fallback B,三步缺一不可)**:① 恢复 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`;② 去掉 `--vllm-enable-sleep-mode`;③ **加 `--no-offload-rollout`** —— 因为 colocate 默认 `offload_rollout=True`(`arguments.py:1764`)会在 `vllm_engine.py:427` **强制加回 sleep**,只删②无效,必须③显式关掉。
  > A/B 不可混用:trainer 与 vLLM 一端 native 一端 VMM 必崩。
- 副作用:无 sleep → vLLM 训练阶段仍占 ~18GB(util 0.3),0.8B + 60GB 卡放得下(用户跑成过 no-sleep colocate)。

### 轮次 16 — 无 sleep(Fallback B)也失败,仍 507899
确认 `offload_rollout=False` 生效(megatron args dump)、两端都应为 VMM,但权重同步仍 `507899`:`shareableHandle[...], deviceId[0]`,句柄在物理 `device_id=4/5/7`。一度判断是后四卡设备号错位。

### 轮次 17 — 🔬 决定性根因:驱动/固件版本不匹配(子智能体最小复现实验)
> 子智能体在空闲卡上写了最小两进程 NPU-IPC 复现(`reduce_tensor` 导出→另进程导入,完全复刻 vime 路径),跑 6 组对照,**推翻了设备号偏移假说**:

| 实验 | 配置 | 结果 |
|---|---|---|
| **T0** | VMM,同卡同命名空间、**零偏移、无 override** | ❌ 507899 `get device uuid failed` |
| T1 | VMM,模拟 NOSET=0(都 dev=0) | ❌ 507899 |
| T2 | VMM,复刻后四卡 colocate(producer多卡视图/consumer单卡) | ❌ 507899 |
| T5/T6 | **native**(Fallback A 路径) | ❌ `driver and firmware packages do not match` + 硬崩 |

**连 T0(完美对齐、零偏移)都崩** → 不是偏移/sleep/expandable/选卡问题。两个铁证:
1. `torch.npu.get_device_properties().uuid` = **全零** `00000000-...` → VMM IPC 依赖的 device UUID 驱动给不出(报错 "get device uuid failed deviceId=0 result=207000")。
2. native 路径报 "Feature is not supported ... **driver and firmware packages do not match**"。

**版本核对(决定性)**:驱动 `24.1.rc2` 的 `compatible_version_fw=[7.0.0, 7.2.99]`,本机固件 **`7.3.0.1.231`** → 超出范围。**驱动固件不兼容,同时废掉 NPU 跨进程 IPC(VMM+native 两路)+ device UUID 查询。这是基础设施层问题,非任何 vime 配置/环境变量/选卡可解。**

验证脚本留在 `mynpu2:/tmp/ipc_test/ipc_harness.py`。修复后重跑 T0 应 PASS、UUID 应非全零。

### 轮次 18 — ✅ 升级驱动后 IPC 彻底通过(根因解除)
用户把驱动升级到 **`26.0.rc1`**(原 24.1.rc2)。验证:device UUID 变为**非全零**(`00cc0800-d000-...`),卡 0-7 全空闲。重跑训练:**IPC 权重同步彻底通过**(无 507899 / missing shared_ptr),完整推进到 actor 第一个 train step(rollout 也已完成)。

### 轮次 19 — train step 的 torch 编译报错(驱动升级副作用)
actor 首个 train step 前向崩:`cannot pickle 'PyCapsule'`(**红鲱鱼**)。子智能体定论:真因是 megatron `bias_dropout`/`bias_swiglu` fusion 的 `@jit_fuser=torch.compile` + vime `ppo_utils.py` 的 `@torch.compile`,在 **torch_npu 2.10(随 26.0.rc1 一起升级)的 inductor `get_gpu_type()` 断言**下失败 → InductorError → Ray 序列化该异常时报 PyCapsule。属驱动升级带来的 torch 行为变化,非 vime 回归。
- 解法(单行、不改 vime 代码、有仓库先例 `scripts/run_cg_consistency_10step.sh:66`):`export TORCHDYNAMO_DISABLE=1` → actor 走纯 eager,一次覆盖全部 3 个 compile 点;vLLM 子进程由 vime `env.pop` 掉它,cudagraph + GDN triton kernel 均不受影响。

### 轮次 20 — ✅✅ 完整跑通(RL 闭环持续迭代)
加 `TORCHDYNAMO_DISABLE=1` 重跑,**完整 RL 闭环跑通并持续迭代**:
- 权重同步(IPC,1.3~2.4s)→ rollout 生成 → math reward → actor train step(`Timer train end 38.9s`)→ 再同步,已稳定完成 **3+ 个 step**(`perf 0/1/2`),`train.py` 持续 ALIVE。
- 性能:`actor_train_tflops≈2.1`、`actor_train_tok_per_s≈1697`、`step_time≈129s`、`update_weights_time≈1.3~2.4s`。
- rollout 生成正常(实测样本:模型在解 2020 堆硬币竞赛题,输出 `<think>…</think>` 推理)。
- **reward 暂为 0**:`rollout/rewards: 0.0`,但 `rollout/truncated: 0.81~0.94` —— 94% 的 response 撞 `--rollout-max-response-len 4096` 被截断,模型竞赛数学推理太长、没写到 `\boxed{答案}`,math reward 提取不到 → 0。**非 bug(流程全通),是 response 预算太小**。增大 `--rollout-max-response-len`(如 8192/16384,对齐 35B 的 16384)+ 相应 `--vllm-max-model-len` 即可让 reward 出信号。

## 6. 结论与状态(✅ 已完整跑通 RL 训练)

**软件层全部打通(全程未改 vime 核心代码)**:依赖补齐(vllm-router/mbridge/modelopt/transformers 5.12.1)→ raw 加载模式 → `--qwen-gdn-backend npu` → 补 GDN 维度参数 → `--rm-type math` → triton 缓存到 /tmp。训练**完整推进到 actor 初始化 + torch_dist 加载 + 4×vLLM ready + 首次权重同步**。

**唯一阻塞 = 硬件/驱动层**:colocate 的 NPU-IPC 权重同步被**驱动 24.1.rc2 与固件 7.3.0.1.231 不兼容**挡死,经最小复现实验证明,非软件可解。

**可行解(按推荐度)**:
1. **运维对齐固件/驱动**(根因修复,保后四卡 colocate,零代码):固件降到 7.2.x 或驱动升级到覆盖 7.3.x。修完现有 IPC 路径原样可用。
2. 换一台驱动固件匹配的机器(用户"无 sleep 成功"的那台很可能就是)。
3. 绕开 IPC 走非 colocate + HCCL 权重同步(需先把缺失的 `vllm_ascend.distributed.weight_transfer.hccl_engine` 编进 vllm-ascend,且 actor/rollout 分卡)或磁盘中转(`update_weights_from_disk` 存在但 colocate 未接线,需改代码)。

## 7. 运维修复指引(已选方案:固件/驱动对齐)

**目标**:让驱动与固件落入同一兼容区间。当前驱动 `24.1.rc2` 支持固件 `[7.0.0, 7.2.99]`,实测固件 `7.3.0.1.231` 超范围。二选一:
- **A. 固件降级到 7.2.x**(落入驱动兼容范围)— 通常更快
- **B. 驱动升级**到 `compatible_version_fw` 覆盖 7.3.x 的版本

**步骤(运维,需 root + 独占整机)**:
1. 协调停掉**所有** NPU 任务(含卡 0 他人进程 `313153`)— 固件/驱动操作需独占。
2. 用 Ascend 官方包降固件(`npu-smi -i <id> -f`)或升驱动(驱动 `.run` 包),按官方流程。
3. 重启或重载驱动。

**修复后自检(两步)**:
1. device UUID 非全零:
   `ASCEND_RT_VISIBLE_DEVICES=3 python3 -c "import torch,torch_npu; print(torch.npu.get_device_properties(0).uuid)"` → 应输出**非全零** UUID(当前是全零)。
2. IPC 复现转 PASS:`bash /tmp/ipc_test/run_ipc_test.sh` → T0 应 **PASS**(当前 6 组全 FAIL,见 `/tmp/ipc_test/RESULTS.log`)。

**修复后一键重跑(软件全就绪,零改动)**:
```bash
cd /workspace/vime
bash scripts/run_qwen35_08b_mathrl_npu_4card.sh
```
脚本当前是「无 sleep / 两端 VMM」配置(复刻用户验证过的共卡无 sleep)。若修复后无 sleep 显存偏紧,可切回「sleep + 两端 native」:加回 `--vllm-enable-sleep-mode`、删 `--no-offload-rollout`、注释 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`(三者保持分配器一致即可)。

## 8. 软件就绪清单(修固件后即可跑,均已落位)
- 模型:`/root/models/Qwen3.5-0.8B-Base`(HF, vLLM rollout 用)
- torch_dist:`/root/models/Qwen3.5-0.8B-Base_torch_dist`(actor `--ref-load` 用,已转换)
- 数据:`/root/datasets/dapo-math-17k/dapo-math-17k.jsonl`
- 依赖:vllm-router / mbridge@89eb108 / nvidia-modelopt+pulp / transformers 5.12.1 + huggingface_hub 1.21
- 脚本:`/workspace/vime/scripts/run_qwen35_08b_mathrl_npu_4card.sh`(GDN backend npu + GDN 维度参数 + rm-type math + triton 缓存 /tmp + 后四卡)
- 转换脚本:`/workspace/vime/scripts/convert_qwen35_08b_torch_dist.sh`

## 9. rollout 吞吐(throughput)分析与优化
**现象**:初配置 rollout 生成 throughput 偏低(稳态单 engine ~260 tok/s,瞬时见过 14.8)。
**子智能体隔离 benchmark 定论(910B3 / vllm-ascend 0.21,含标准模型对照)**:

| 配置 | 单请求 | batch64 单 engine |
|---|---|---|
| GDN 0.8B eager(关图) | 12.6 tok/s | — |
| **GDN 0.8B aclgraph(当前)** | **66 tok/s** | **2798 tok/s** |
| GDN 0.8B + static_kernel | 73(+10%) | — |
| 标准 Qwen2.5-0.5B aclgraph(对照) | 96 tok/s | 4247 tok/s |

- **单请求是平台单流地板**:标准 0.5B 也才 96 tok/s,GDN 66 = 平台地板 + GDN ~50% 开销;单请求"数千"对这套 910B/vllm-ascend **任何**小模型都不可达(host 调度 bound)。
- **图模式全开**:GDN eager→aclgraph **5.2×**(12.6→66),无 eager fallback,`FULL_DECODE_ONLY` 已最优。
- **聚合数千可达且已验证**:提并发 16→128 sample 后实测单 engine **1634 tok/s**、4 engine 聚合 **~6500 tok/s**。根因:初配置照搬 35B 小批 test(16 sample → 4 req/engine,摊不开 GDN ~15ms/step 固定开销)。
- **优化(纯脚本,不改 vime)**:① 提并发(`--rollout-batch-size`×`--n-samples-per-prompt` → 64-128);② `--vllm-max-num-seqs 48`;③ `--vllm-additional-config '{"ascend_compilation_config":{"enable_static_kernel":true}}'`(+10%);④ 图模式保持 `FULL_DECODE_ONLY`。

## 10. sleep + offload 全开测试(为大模型训练准备)
固件修复后测试 vime 标准的省显存特性集(对齐 35B 验证脚本),为之后 35B 等大模型训练铺路。子智能体核查(file:line 依据)后的**正确配置**:

| 特性 | 参数 | 说明 |
|---|---|---|
| sleep | **加** `--vllm-enable-sleep-mode` + **删** `--no-offload-rollout` | 只加 sleep flag 而保留 `--no-offload-rollout` → sleep 永不被调用 → OOM。`offload_rollout`(colocate 默认 True)驱动运行期 sleep/wake(`placement_group.py:214`/`rollout.py:1073`) |
| trainer expandable | **保留** `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` | **不要关!** vime 在 `vllm_engine.py:360-366` 只对 vLLM 子进程 `env.pop`,trainer 自留。两端"trainer expandable + vllm native"即 35B 验证组合 |
| NPU 权重 offload | `offload_train`(colocate 默认 True,已开) | `actor.py:85` 初始化 `NPUWeightOffloader`,rollout 阶段 offload/onload 模型权重 |
| 优化器 CPU offload | **加** `--optimizer-cpu-offload --use-precision-aware-optimizer --overlap-cpu-optimizer-d2h-h2d` | Adam 状态卸 CPU。`optimizer-cpu-offload` 硬依赖 `use-precision-aware-optimizer`(megatron assert) |
| 激活省显存 | `--recompute-granularity full`(已开) | NPU 无 TE,真正的 activation CPU offload(`--cpu-offloading-num-layers`/FGA)会 assert 崩;recompute 是唯一可用手段 |
| KV offload | **不加** | 0.8B KV 极小 + sleep 已释放;vllm V1 删了 `--swap-space`,需 kv-connector,大模型再单独验证 |

**✅ 测试结果(完整跑通 + 持续迭代)**:
- `Sleep mode freed 8.96 GiB`(vLLM 训练阶段让出近 9GB HBM)
- NPU 权重 offloader:`before/after wake_up model` 显存随 offload/onload 变化(rollout 阶段 offload、train 阶段 onload)
- optimizer CPU offload(precision-aware)+ GDN **兼容,无报错**
- 完整 **2+ 个 train step**(`Timer train end 40s`,`perf 0/1`),持续迭代
- perf:sleep 2.2-3.1s / wake_up 1.4s / update_weights 1.3-2.2s / actor_train 33-40s / step 141-158s
- actor 显存仅用 **~7-9 GB**(总 61GB)→ sleep + offload 大幅省显存
- **结论:这套「sleep + NPU 权重 offload + optimizer CPU offload + 激活重计算」组合已验证可用,可直接用于 35B 等大模型训练**(对齐 `run_qwen36_35b_a3b_dapo_math_npu.sh`)。

## 11. 显存中间层(torch_memory_saver)调研 + rollout NPU 残留诊断
**Q:rollout 阶段 actor 是否卸载?** `actor.sleep()` 调 `NPUWeightOffloader.offload()`(`NPU weight offload: 230 params, 1435 MiB`),把 `param.data` 指针换成 CPU + `grad=None`,actor `allocated` 7.04→4.5GB。**但这是「假卸载」**:`param.data` 本是 DDP flat buffer 的 view,换指针并未释放底层 flat buffer,省下的 ~2.5GB 实为 `empty_cache` 收的激活(详见下方 v4 实测铁证)。优化器 state 由 `--optimizer-cpu-offload`(`fraction=1.0`)常驻 CPU。

**Q:能否像 torch_memory_saver 提供 VMM 中间层?(子智能体实测 + 三方源码印证)**
1. **507899 = 驱动问题,已被 26.0.rc1 修复(实测铁证)**:在新驱动下重跑 IPC harness(`/tmp/ipc_test/RESULTS_postupgrade.log`),**T0-T6 全 PASS**(旧驱动全 FAIL/段错误)。VMM + native 两条 IPC 路径、含 colocate 卡偏移场景都通。"分配器类型必须一致"是旧驱动下的伪结论。
2. **但 trainer 仍不能用 torch_memory_saver(VMM region)** —— 另一个独立根因(非驱动):region 物理内存 `handleType=ACL_MEM_HANDLE_TYPE_NONE` **不可导出 IPC handle**(三方印证:tms `csrc/utils.h` + vllm-ascend `camem_allocator.cpp:54,135` + torch_npu `AclInterface.h` 的可导出路径),且 `resume()` 重建句柄使已导出句柄失效。colocate 权重同步要 trainer **导出** actor 权重给 vLLM → region 导不出。**vLLM 能用 VMM(接收方,从不导出),trainer 不能(导出方)** —— 这就是为何 vLLM sleep(CaMemAllocator)能用、trainer 只能用 offloader。
3. **NPUWeightOffloader 是 trainer 唯一可行(能导出 IPC handle)**;但 v4 探针实测发现它对 Megatron distributed-optimizer 是**「假卸载」** —— 只把 `param.data` 换成 CPU 指针,底层 DDP flat buffer 没释放,rollout 阶段 HBM 实际没省(详见下方实测)。
4. **ref model 不加载**:`with_ref=(kl_coef≠0 or use_kl_loss)`,脚本 `--kl-loss-coef 0` → ref 不上 NPU、不参与前向(`--ref-load` 只是 actor 初始权重源)。rollout 残留 **不含 ref**;残留主体是 **Megatron DDP 的 fp32 `grad_data` + bf16 `param_data` flat buffer**(v4 铁证见下,既非 framework 也非优化器 state —— 此前"优化器 fp32 main"是 v3 误判)。
5. **真正优化方向 = host 内存去重(35B 实际瓶颈)**:actor 权重在 host 存**两份**(`TensorBackuper` pinned 喂 vLLM + `NPUWeightOffloader.cpu()`),`wake_up` 还重复还原两次。35B bf16 多占 ~70GB pinned host + 每轮一次 70GB 冗余 CPU→NPU 拷贝。最小改法(提案,需 review):offloader 不再 `.cpu()` 整存,仅释放 NPU 存储 + 记 shape/dtype,数据交给已有的 `_switch_model("actor")` 从 backuper 填。

**诊断方法(hook/patch 探针,不改 vime)**:写 `npu_mem_probe.py` + `sitecustomize.py`(放 `PYTHONPATH` 最前,每个 ray actor 进程自动 import 注入,零改 vime 源码)。演进了三版才落地:
- **patch 目标的坑**:最初 monkey-patch `MegatronTrainRayActor.sleep/wake_up` —— `cls.sleep=wrapper` 确实执行了(`patched` 行落盘为证),但 4 个 actor 进程**无一触发 wrapper 入口的 `ENTER` 标记**:ray actor 的 `self.sleep()` 方法分发不走被替换的类属性(cloudpickle 类重建 / 实例方法绑定的玄学)。**改 patch `NPUWeightOffloader.offload(model)/onload(model)`**(actor 内部持有的普通工具对象,非 ray actor,`type(inst).offload` 必命中;且 `npu_weight_offloader.py:194` 日志铁证它被调用)后一次成功。
- **dump 内容**:`torch.npu.memory_allocated()` 总量(权威) + 扫 `model` 的 `named_parameters()+named_buffers()` 统计仍在 NPU 的字节 + 安全 gc 全扫描(每对象 try 包裹,列出进程内所有 NPU tensor 句柄)。写 `/root/npu_probe_<pid>.log`(绕 ray stdout 缓冲)。首版裸 `gc.get_objects()`(`numel()/shape` 在 per-obj try 外抛异常→整段静默中断、err 也被 stdout 缓冲)不可靠,改为「先写 allocated+model 归类主结果、再附加安全 gc」后稳定落盘。

**实测结果(0.8B,4 actor 一致;三轮逼近,v4 加 DDP buffer + optimizer 扫描钉死):**

| 阶段 | allocated | model 在 NPU | DDP `grad_data` | DDP `param_data` |
|---|---|---|---|---|
| AFTER_OFFLOAD(rollout) | **4.514 GB** | **0.000 GB**(0/230) | **3.010 GB fp32 @npu** | **1.505 GB bf16 @npu** |
| AFTER_ONLOAD(train) | 6.019 GB | 1.505 GB(230/230) | 3.010 GB fp32 @npu | 1.505 GB bf16 @npu |

- **铁证:offload 后 4.514GB ≈ `grad_data` 3.010(fp32) + `param_data` 1.505(bf16),两个 Megatron DDP flat buffer。** optimizer state 实测为空(`fraction=1.0` 已全卸 CPU),framework 近 0。(v3 曾把 gc 看到的 `752M f32` 误判为"优化器 fp32 main",v4 直接扫 `_ParamAndGradBuffer` 才钉死是 DDP `grad_data`。)
- **「假卸载」**:offloader 把 `param.data` 换成 CPU(named scan=0、`onload−offload=1.505GB` 看似卸了 model),但 `param.data` 本是 DDP `param_data` flat buffer 的 view → **底层 flat buffer 没释放,rollout 阶段 model 权重 HBM 一点没省**(省的 2.5GB 只是 `empty_cache` 收的激活)。
- **根因 = `_release_ddp_buffers` 失效**:它本为释放 flat buffer 而写,但用 `dir(ddp)` 找单个有 `grad_data` 属性的对象;而 Megatron 把 `_ParamAndGradBuffer` 放在 **`ddp.buffers`(list)+ `expert_parallel_buffers`(MoE)** 里 → `hasattr(list,"grad_data")=False` → **整段空操作**。
- **修检测后仍释放不了(架构限制,已回退改动)**:遍历 list 后 `Released 4305 MiB` 日志出现、`buf.grad_data/param_data` 设 empty、训练正确(perf 跑通),**但 allocated 纹丝不动 4.514GB**。因 storage 被**多重引用**:每个 `bucket.grad_data`(view)、`bucket_group.cached_*_shard_list`、distributed-optimizer main grad/param —— 断一个引用不释放;且它们是 reduce-scatter/all-gather **热路径常驻**,不能在 rollout 释放。
- **VMM 两难(闭环)**:能绕过 Python 多重引用、在虚拟内存层 unmap 物理页的只有 `torch_memory_saver`(VMM),但 NPU VMM region 不可导出 IPC(权重同步崩);Python offload 能导 IPC 但释放不了 flat buffer。**NPU+colocate 下这 4.5GB 架构上无法在 rollout 释放**。
- **可省方向**:① grad buffer fp32→bf16(`accumulate_allreduce_grads_in_fp32=False`,省 1.5GB,精度 tradeoff);② non-colocate 分卡(可上 VMM,牺牲卡利用率);③ 接受(大模型按 per-rank 评估)。
