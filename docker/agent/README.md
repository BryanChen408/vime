# vime + Polar 部署总指引(docker/agent)

vime 算子 RL 全栈的容器镜像与部署脚本,目标平台 **Ascend A3(910_93 类)/ aarch64,
CANN 9.0.0**。这是**统一入口**——下面先讲三块怎么拼、外部按什么顺序用,再给文档索引。

## 三块组成

| 目录 | 镜像 | 是什么 |
|-----|------|--------|
| [`training/`](training/) | `vime:a3-release` | RL **训练镜像**:vllm / vllm-ascend / Megatron / MindSpeed / fla_npu,各自从**官方上游** pin 版本 + 小 patch。NPU 自定义算子首次进容器再编(build 期无 NPU)。 |
| [`sandbox/`](sandbox/) | `ascendc-tilelang:v1`(profile 里现名 `sandbox:v1`) | Polar agent 跑任务的**沙箱镜像**:TileLang(源码编)+ torch/torch_npu + cmake<4 + Claude Code CLI。 |
| [`polar/`](polar/) | — | **Polar**(`ProRL-Agent-Server`,agentic RL server)。独立私仓拉取,分支 `feat/ascendc-rl-t2a`;经 slime_bridge 驱动训练,在沙箱镜像里跑任务 rollout。 |

```
  training/  vime:a3-release   (RL 训练器: actor + vLLM rollout)
        │  slime_bridge(权重同步 / 任务 I/O)
  polar/    ProRL-Agent-Server (agent server: triton / ascendc 算子任务)
        │  拉起任务子容器
  sandbox/  ascendc-tilelang   (每任务执行环境: tilelang / AscendC)
```

## 外部使用流程(按此顺序)

1. **构建训练镜像** —— 按 [`training/README.md`](training/README.md):把 triton-ascend
   wheel 放进 `training/wheels/`,`docker build -f Dockerfile.release -t vime:a3-release .`。
   (本地测试可用 `Dockerfile.release.cann900`。)
2. **首次编 NPU 算子** —— 在 A3 机器上进训练容器跑一次
   `bash /workspace/build_npu_kernels.sh`(编 vllm-ascend 自定义算子 + fla_npu),
   可选 `docker commit` 成 `vime:a3-final`。
3. **构建沙箱镜像** —— 按 [`sandbox/README.md`](sandbox/README.md):
   `bash build_ascendc_tilelang.sh` 出 `ascendc-tilelang:v1`。
4. **拉起 Polar** —— 按 [`polar/README.md`](polar/README.md):在宿主机 clone
   `ProRL-Agent-Server` checkout `feat/ascendc-rl-t2a`、`pip install -e`。
5. **启动训练** —— 按 [`RUNBOOK.md`](RUNBOOK.md):单机/双机拉起 vime+polar,
   profile 按场景选(triton→`profile.vime.yaml`,ascendc→`profile.t2a.yaml`)。

## 文档索引

- [`RUNBOOK.md`](RUNBOOK.md) —— **主启动参考**(镜像加载、Polar profile、resource
  layout、单机/双机启动命令、调参、验证、常见问题)。
- [`training/README.md`](training/README.md) —— 训练镜像构建 + 首次算子编译 + 依赖基线表。
- [`sandbox/README.md`](sandbox/README.md) —— 沙箱镜像构建 + A3/aarch64 说明。
- [`polar/README.md`](polar/README.md) —— Polar 拉取 + 场景(triton/ascendc)profile 对照 + 部署。

## 提交进 vime 仓的约定

- `training/wheels/*.whl`(triton-ascend wheel,~188MB)**已 .gitignore**,不提交;按
  `training/README.md` 从 gitcode releases 下载。
- `training/patches/*.patch`(小文本)与所有 Dockerfile/脚本正常提交。
- Polar 是独立仓(不 vendor 进来),`polar/` 只写怎么拉+部署。
