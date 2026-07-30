# vime + Polar 启动参考

面向已经把训练镜像 load 成容器后的使用者,目标是把单机/双机 **vime + Polar**
算子 RL 训练稳定拉起来。改编自内部的 slime+polar 启动参考;**关键区别:vime 用
vLLM/vllm-ascend 推理(不是 SGLang)**,并有一批 NPU 专属环境变量。

当前边界:

- **vime 训练容器**负责 Ray、Megatron 训练、**vLLM 推理**、样本数据集读取。
- **Polar 仓库放在宿主机**,负责 rollout/gateway/observer/watcher,以及启动 agent
  runtime 子容器(即 sandbox 镜像)。
- vime 只需要知道 Polar rollout URL、数据集路径、resource layout。
- 运行产物:vime 侧写到 `output/polar_bridge/`;polar 侧写到仓内
  `output/ascend_operator/`。

---

## 1. 目录约定

```
宿主机:
/home/docker/polar_debug/ProRL-Agent-Server/     # Polar 仓(feat/ascendc-rl-t2a)
  deploy/ascend_operator/
  operator_runtime/
  output/ascend_operator/

训练容器内:
/workspace/vime/
  scripts/run-qwen36-35b-polar-minimal.sh
  scripts/resource_layout.*.yaml
  vime_bridge/                                    # vime↔polar 桥接
  output/polar_bridge/
```

Polar 的 profile 里相对路径按 Polar 仓根目录解析;不要把路径写死进配置。

---

## 2. 加载镜像并启动训练容器

每台参与 vime/Ray 的机器都要起训练容器。

```
docker load -i <vime-a3-release-image>.tar

docker run -it \
  --name <vime-container-name> \
  --net=host --ipc=host --shm-size=768g --privileged \
  -v /dev:/dev \
  -v /home/docker:/home/docker \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  vime:a3-release \
  bash
```

**首次进容器:编译 NPU 自定义算子**(镜像里 vllm-ascend/fla_npu 只 clone 未编译,
见 `training/README.md`):

```
bash /workspace/build_npu_kernels.sh
# 可选:固化成已编译镜像,后续直接用
#   docker commit <vime-container-name> vime:a3-final
```

Polar agent runtime 用的 **sandbox 镜像**也要在 Polar 宿主机可见。profile 里默认是
`sandbox:v1`(即 `ascendc-tilelang` 沙箱,见 `sandbox/`;如已按新名构建可改成
`ascendc-tilelang:v1`)。先确认存在:

```
docker images | grep -E 'sandbox.*v1|ascendc-tilelang'
```

---

## 3. Polar 宿主机准备

Polar 必须在**能访问 Docker daemon 的宿主机**启动(它的 DockerRuntime 要拉起 agent
runtime 子容器),**不要在训练容器里启动**。宿主机需要 Polar 仓 + Python 3.11。

```
cd /home/docker/polar_debug/ProRL-Agent-Server
git checkout feat/ascendc-rl-t2a          # 暂定分支

# 已有 venv 直接激活:
source /root/polar-venv/bin/activate
# 没有则创建:
#   python3.11 -m venv /root/polar-venv
#   source /root/polar-venv/bin/activate
#   pip install -e .
```

---

## 4. Polar Profile 配置

Polar 宿主机只读一个 profile,**按算子任务场景选**:

| 场景 | profile |
|---|---|
| **triton** 算子(生成/优化 Triton kernel) | `deploy/ascend_operator/profile.vime.yaml` |
| **ascendc** 算子(生成 AscendC kernel,t2a) | `deploy/ascend_operator/profile.t2a.yaml` |

下文命令统一用 `${POLAR_PROFILE}`,按场景设成上面之一。两份 profile 的字段结构一致
(下面以 `profile.vime.yaml` 为例);主要差别在 `operator.runtime.image`
(triton 沙箱 vs ascendc-tilelang 沙箱)和 npu_pool。

关键字段:

```yaml
service:
  bind_host: 0.0.0.0
  rollout_url:       http://<train-host>:8080     # vime 访问 polar rollout server
  gateway_url:       http://<train-host>:8100     # polar gateway 对 agent runtime 暴露
  sglang_router_url: http://<train-host>:8001     # ⚠️ 字段名是遗留;实际指向 vime vLLM
                                                   #    router 端口(VLLM_ROUTER_PORT=8001)
  inference_engine:  vllm                          # 选 vllm 后端(不是 sglang)

paths:
  output_dir:            output/ascend_operator
  operator_runtime_dir:  operator_runtime

pipeline_budget:
  generation_max:   3      # Claude Code 生成轮数上限
  optimization_max: 1      # 优化轮数上限

gateway:
  max_init_workers:    16
  max_run_workers:     64   # agent runtime 并发上限(想 64 个 Claude Code 并发就开到 64)
  max_postrun_workers: 64

operator:
  runtime:
    image:    sandbox:v1                        # = ascendc-tilelang 沙箱;按需改 ascendc-tilelang:v1
    network:  host
    npu_pool: "0,1,2,3"                          # polar agent/eval 可租用的 NPU(避开 rollout/train)
```

**与 slime 的唯一实质差异**:推理端点。slime 的 `sglang_router_url` 指向 SGLang
router(:4077);vime 里这个字段(名字没改)指向 **vime vLLM 的 router 端口 :8001**,
并靠 `inference_engine: vllm` 选后端。vime 侧要开 `FEAT_LB_PROXY=1` 把这个端口做成
Python 透传 LB proxy(见 §10)。

---

## 5. 启动 Polar 宿主机服务

```
cd /home/docker/polar_debug/ProRL-Agent-Server
source /root/polar-venv/bin/activate

# 按场景选:triton→profile.vime.yaml,ascendc→profile.t2a.yaml
POLAR_PROFILE=deploy/ascend_operator/profile.vime.yaml \
POLAR_RUN_ID=polar_$(date +%Y%m%d_%H%M%S) \
NO_PROXY=127.0.0.1,localhost,<train-host>,<infer-host> \
no_proxy=127.0.0.1,localhost,<train-host>,<infer-host> \
  bash deploy/ascend_operator/restart_polar_host.sh
```

检查:

```
curl -s http://<train-host>:8080/health      # rollout
curl -s http://<train-host>:8100/health      # gateway
# observer: http://<train-host>:18088
```

---

## 6. Resource Layout 配置

vime 训练容器通过 `RESOURCE_LAYOUT` 指定训练/推理卡。vime 格式:

```yaml
roles:
  actor:                                    # 训练 actor 的 NPU
    - {node: <train-host>, devices: "8-15"}
  rollout:                                  # vLLM 推理的 NPU
    - {node: <infer-host>, devices: "8-15"}
  polar_reserved:                           # 给 polar agent/eval 预留(不进 Ray placement)
    - {node: <train-host>, devices: "0-3"}
rollout:
  num_gpus_per_engine: 4                     # 每个 vLLM engine 的 TP 卡数
```

现有样例(`scripts/`):

```
resource_layout.dual88train64infer.yaml                   # 双机:88 训练 / 64 推理
resource_layout.dual88train52infer.yaml
resource_layout.single_agent0-3_rollout4-7_train8-15.yaml # 单机:polar0-3/rollout4-7/train8-15
```

> 说明:vime 的 `rollout.num_gpus_per_engine` 对应 slime 的同名字段;vime 用
> `--rollout-num-gpus` / `--rollout-num-gpus-per-engine` 传引擎数与 TP,没有 slime 的
> `sglang_dp_size`(DP 由 `FEAT_DP_EXTERNAL_LB` 控制,见 §10)。

---

## 7. 数据集路径

vime 训练容器需能读到:

```
<DATA_ROOT>/operator_tasks.jsonl
<DATA_ROOT>/op_tasks/
```

vime 默认(可覆盖):

```
OPERATOR_DATA_ROOT=<DATA_ROOT>
# 或显式:
OPERATOR_TASK_JSONL=<DATA_ROOT>/operator_tasks.jsonl
OPERATOR_TASKS_DIR=<DATA_ROOT>/op_tasks
```

---

## 8. 双机启动:train-host 训练 + infer-host 推理

拓扑:train-host 0-7 训练、8-15 给 polar;infer-host 8-15 推理;Polar 跑在 train-host。

先起 Polar(profile 里 `rollout_url/gateway_url` 指 train-host,`sglang_router_url` 指
train-host:8001,`npu_pool: "8-15"`)。

然后 train-host 训练容器起 Ray head(脚本会等 infer-host worker 加入后自动提交):

```
cd /workspace/vime
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_ADDR=<train-host> CURRENT_IP=<train-host> SOCKET_IFNAME=<train-host-iface> \
NNODES=2 NPUS_PER_NODE=8 \
RUN_ID=qwen36_polar_dual_$(date +%Y%m%d_%H%M%S) \
RESOURCE_LAYOUT=scripts/resource_layout.dual88train64infer.yaml \
POLAR_ROLLOUT_URL=http://<train-host>:8080 \
VLLM_ROUTER_PORT=8001 \
OPERATOR_DATA_ROOT=<DATA_ROOT> \
NUM_ROLLOUT=2 ROLLOUT_BATCH_SIZE=2 N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=8 \
  bash scripts/run-qwen36-35b-polar-minimal.sh
```

infer-host 训练容器加入 Ray:

```
cd /workspace/vime
ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
MASTER_ADDR=<train-host> CURRENT_IP=<infer-host> SOCKET_IFNAME=<infer-host-iface> \
NNODES=2 NPUS_PER_NODE=8 \
  bash scripts/run-qwen36-35b-polar-minimal.sh
```

train-host 只把 0-7 给 Ray;8-15 留给宿主机 polar 子容器,不入 Ray。infer-host 只把
8-15 给 Ray 做 vLLM 推理。

---

## 9. 单机启动:一台机同时训练、推理、Polar

拓扑:0-3 polar agent/eval | 4-7 vLLM 推理 | 8-15 训练。

Polar(单机 profile,`npu_pool: "0,1,2,3"`):

```
cd /home/docker/polar_debug/ProRL-Agent-Server
source /root/polar-venv/bin/activate
# 按场景选:triton→profile.vime.yaml,ascendc→profile.t2a.yaml
POLAR_PROFILE=deploy/ascend_operator/profile.vime.yaml \
POLAR_RUN_ID=polar_single_$(date +%Y%m%d_%H%M%S) \
NO_PROXY=127.0.0.1,localhost,<host> no_proxy=127.0.0.1,localhost,<host> \
  bash deploy/ascend_operator/restart_polar_host.sh
```

vime 训练容器(只暴露 4-15 给 Ray):

```
cd /workspace/vime
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7,8,9,10,11,12,13,14,15 \
MASTER_ADDR=<host> CURRENT_IP=<host> SOCKET_IFNAME=<host-iface> \
NNODES=1 NPUS_PER_NODE=12 \
RUN_ID=qwen36_polar_single_$(date +%Y%m%d_%H%M%S) \
RESOURCE_LAYOUT=scripts/resource_layout.single_agent0-3_rollout4-7_train8-15.yaml \
POLAR_ROLLOUT_URL=http://<host>:8080 VLLM_ROUTER_PORT=8001 \
OPERATOR_DATA_ROOT=<DATA_ROOT> \
NUM_ROLLOUT=2 ROLLOUT_BATCH_SIZE=2 N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=8 \
  bash scripts/run-qwen36-35b-polar-minimal.sh
```

---

## 10. 常用调参

**训练规模**(脚本 env):

```
NUM_ROLLOUT=4 ROLLOUT_BATCH_SIZE=16 N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=64
POLAR_MAX_ACTIVE_SESSIONS=64
```

**Claude Code pipeline 上限**(Polar profile 的 `pipeline_budget`,不是脚本 NUM_ROLLOUT):

```yaml
pipeline_budget: { generation_max: 5, optimization_max: 3 }
```

**vLLM 推理并行**(取代 slime 的 sglang DP):

```
ROLLOUT_NUM_GPUS=8                # rollout 总卡数
ROLLOUT_NUM_GPUS_PER_ENGINE=4    # 每 engine TP —— 8 卡 = 2 engine × TP4
FEAT_DP_EXTERNAL_LB=1            # vLLM 原生 external-LB 分布式 DP(每 engine=1 DP rank)
FEAT_LB_PROXY=1                 # 前置 Python LB proxy 替 Rust router(保 return_token_ids
                                #  + 会话亲和);polar 的 sglang_router_url 就指这个 :8001
```

> `FEAT_DP_EXTERNAL_LB=1` 必须同开 `FEAT_LB_PROXY=1`。可选 `FEAT_BALANCE_SCHED=1`
> 防跨 DP batch 拖尾。

---

## 11. 验证

```
ray status                                        # Ray
curl -s http://<train-host>:8001/health           # vLLM router (LB proxy)
curl -s http://<train-host>:8080/health           # polar rollout
curl -s http://<train-host>:8100/health           # polar gateway

# vime 训练日志:
tail -f /workspace/vime/output/polar_bridge/... 或 logs/
# polar 日志:
tail -f .../output/ascend_operator/runs/${POLAR_RUN_ID}/logs/rollout.log
tail -f .../output/ascend_operator/runs/${POLAR_RUN_ID}/logs/gateway.log
# observer: http://<polar-host>:18088
```

---

## 12. 常见问题

**vime NPU 专属(slime 没有,但 vime 必须)** —— 脚本已内置,排障时确认没被覆盖:

- `TASK_QUEUE_ENABLE=0` —— **=1 会让 GDN/ring-attn 训练出 NaN**,必须 0。
- `VLLM_ASCEND_ENABLE_NZ=0` —— NZ 格式与 RL 每步换权重冲突(精度崩),必须 0。
- `VLLM_VERSION=0.21.0` —— 本机 vllm 自报 0.21.1.dev2,不钉会走 vllm-main-only 导入报
  ModuleNotFound。
- `QWEN36_CP_MODE=ulysses` / `QWEN36_CAUSAL_CONV1D_IMPL=triton` —— GDN/CP 路径。
- `TORCHDYNAMO_DISABLE=1` —— 昇腾 inductor 断言,走 eager。
- `source /usr/local/Ascend/nnal/atb/set_env.sh` —— 除 CANN set_env 外还要 source atb。

**代理**:Polar health 返回代理 HTML 或 504 → 检查 `NO_PROXY/no_proxy` 是否绕过
train-host/infer-host。

**网卡**:Gloo 报找不到网卡 → 容器内 `ip -o -4 addr show` 查真实网卡名,设
`SOCKET_IFNAME=<真实网卡名>`。

**Ray 资源数不对**:`ray status` 对照 —— `RESOURCE_LAYOUT` 里 `node` 必须与 Ray 看到
的节点 IP 一致,`devices` 必须与该节点暴露的设备编号一致。

**短 debug run 尾部 session 异常落盘**:session_pool 预取导致的收尾现象;训练侧只消费
`NUM_ROLLOUT * ROLLOUT_BATCH_SIZE` 个 group,prefix merge 后训练 sample 数可能大于原始
session 数,正常。长稳训练不会表现成这个问题。
