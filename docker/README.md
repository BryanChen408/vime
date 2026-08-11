# vLLM v0.23.0 PD-Mooncake Docker Image

Docker 镜像，包含：
- vLLM v0.23.0 (editable)
- vLLM-Ascend v0.23.0 (editable)
- Mooncake v0.3.9 (ASCEND_DIRECT=ON)
- Go 1.23.8 (清华镜像源)
- MPICH 4.2.3
- 所有必要的依赖和补丁

## 文件清单

```
/workspace/
├── Dockerfile.vllm-pd-mooncake    # Dockerfile
├── requirements-vllm-pd.txt       # Python 依赖
├── build-docker.sh                # 构建脚本
├── README-docker.md               # 本文档
└── patches/
    └── 01-vllm-ascend-v023-compat.patch  # vllm-ascend 兼容性补丁
```

## 前置要求

1. **基础镜像**：需要包含 CANN 9.0.0、Python 3.11、torch-npu
2. **Docker 版本**：>= 19.03
3. **磁盘空间**：至少 20GB
4. **网络**：需要访问 GitHub、清华镜像源

## 构建镜像

### 方法 1: 使用构建脚本（推荐）

```bash
cd /workspace
chmod +x build-docker.sh
./build-docker.sh
```

### 方法 2: 手动构建

```bash
cd /workspace

# 准备补丁目录
mkdir -p patches
cp /workspace/vime-pd-patches-20260808/01-vllm-ascend-v023-compat.patch patches/

# 构建镜像
docker build \
  --build-arg BASE_IMAGE=your-base-image:tag \
  -t vllm-pd-mooncake:v0.23.0-ascend \
  -f Dockerfile.vllm-pd-mooncake \
  .
```

### 自定义构建参数

```bash
docker build \
  --build-arg BASE_IMAGE=your-base-image:tag \
  --build-arg GO_VERSION=1.23.8 \
  --build-arg MOONCAKE_VERSION=v0.3.9 \
  --build-arg VLLM_ASCEND_VERSION=v0.23.0 \
  -t vllm-pd-mooncake:custom \
  -f Dockerfile.vllm-pd-mooncake \
  .
```

## 运行容器

### 基本运行

```bash
docker run -it --rm \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  vllm-pd-mooncake:v0.23.0-ascend
```

### 挂载代码和数据

```bash
docker run -it --rm \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /path/to/vime:/workspace/vime \
  -v /path/to/models:/workspace/models \
  -v /path/to/data:/workspace/data \
  -p 8001:8001 \
  -p 28290:28290 \
  vllm-pd-mooncake:v0.23.0-ascend
```

## 验证环境

### 容器内验证

```bash
# 验证 Python 包
python3 -c "import vllm; print('vllm:', vllm.__version__)"
python3 -c "import vllm_ascend; print('vllm_ascend: OK')"
python3 -c "import mooncake; print('mooncake: OK')"

# 验证 Go
go version

# 验证 CANN
source /usr/local/Ascend/cann-9.0.0/set_env.sh
npu-smi info
```

### 快速测试

```bash
docker run --rm vllm-pd-mooncake:v0.23.0-ascend \
  python3 -c "import vllm, vllm_ascend, mooncake; print('✅ All OK')"
```

## 镜像内容

### 安装路径

- **vllm**: `/workspace/vllm` (editable)
- **vllm-ascend**: `/workspace/vllm-ascend` (editable)
- **Mooncake**: `/workspace/Mooncake` + `/usr/local/Ascend/cann-9.0.0/python/site-packages/mooncake`
- **Go**: `/usr/local/go`
- **yalantinglibs**: `/usr/local/include/ylt`

### 环境变量

```bash
VLLM_VERSION=0.23.0
USE_WANDB=0
ASCEND_ROOT=/usr/local/Ascend
CANN_ROOT=/usr/local/Ascend/cann
PATH=/usr/local/go/bin:...
PYTHONPATH=/usr/local/lib/python3.11/site-packages:...
GOPROXY=https://goproxy.cn,direct
```

### 已应用的补丁

- `01-vllm-ascend-v023-compat.patch`: 禁用 v0.24+ 的 patch_dp_device_ids

## 运行 PD 脚本

```bash
# 进入容器
docker exec -it <container-id> bash

# 清理残留进程（如果有）
pkill -9 -f 'vllm|ray|train_async'
ray stop --force

# 运行 PD 脚本
cd /workspace/vime
bash scripts/run-qwen36-35b-polar-minimal-single-rollout-only-pd.sh
```

## 故障排查

### 1. CANN 环境未加载

```bash
# 容器内手动 source
source /usr/local/Ascend/cann-9.0.0/set_env.sh
```

### 2. 端口冲突

```bash
# 清理残留进程
pkill -9 -f 'vllm|ray'
ray stop --force
```

### 3. 验证 NPU 可见性

```bash
npu-smi info
python3 -c "import torch_npu; print(torch_npu.npu.device_count())"
```

### 4. Go 依赖问题

```bash
# 验证 Go 环境
go env
echo $GOPROXY  # 应显示 https://goproxy.cn,direct
```

## 镜像大小优化（可选）

构建后的镜像可能较大（10-15GB）。优化建议：

1. **多阶段构建**：将编译和运行时分离
2. **清理缓存**：`docker build --no-cache`
3. **压缩层**：使用 `docker-squash` 或 multi-stage build

## 相关链接

- [vLLM](https://github.com/vllm-project/vllm)
- [vLLM-Ascend](https://github.com/Ascend/vllm-ascend)
- [Mooncake](https://github.com/kvcache-ai/Mooncake)
- [补丁包](./vime-pd-patches-20260808/)

## 许可证

本 Dockerfile 遵循相关项目的开源许可证。
