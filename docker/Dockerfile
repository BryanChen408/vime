# syntax=docker/dockerfile:1.7
# =============================================================================
# vime 训练框架镜像 — Ascend A3 (ascend910_93) / aarch64 / CANN 9.0.0
#
# 输入:一个「只有基础 CANN 环境」的镜像(CANN toolkit + kernels + python,
#       没有 torch / torch_npu / vllm / Megatron / MindSpeed)。
# 输出:可直接跑 vime RL 训练(Megatron 训练侧 + vLLM-Ascend rollout 侧)的镜像。
#
# 本文件由三份 Dockerfile 合并整理而来:
#   - rllm-env/dev-base/Dockerfile   → 系统依赖 / CANN env / torch 栈 / MPICH / HCCL
#   - rllm-env/dev-vime/Dockerfile   → python 依赖分层 / 各依赖仓 clone+editable 安装 / fla_npu
#   - vime/docker/Dockerfile(旧)   → 仅 PD-Mooncake 推理链路,现降级为可选阶段
#     (旧文件已备份为 docker/Dockerfile.pd-mooncake)
#
# 构建(不需要 build context,所有源码都是 clone 的):
#   docker build -t vime:a3-cann900 -f vime/docker/Dockerfile \
#     --build-arg HTTP_PROXY=... --build-arg HTTPS_PROXY=... \
#     --build-arg PIP_INDEX_URL=<内网源> --build-arg PIP_TRUSTED_HOST=<host> \
#     --build-arg BUILD_NPU_KERNELS=0 \
#     vime/docker
#
# 两段式构建(默认):BUILD_NPU_KERNELS=0
#   vllm-ascend 的 AscendC 自定义算子 与 fla_npu 的 GDN 算子 依赖 bisheng / npu-smi /
#   真实 NPU 环境,在 docker build 阶段经常不可用。默认只 clone + 装 python 部分,
#   首次在 A3 机器上起容器后跑一次镜像内的 /workspace/build_npu_kernels.sh 完成编译,
#   再 docker commit 固化。若你的构建机就是 A3 且能跑 npu-smi,可 --build-arg BUILD_NPU_KERNELS=1
#   一步到位。
#
# 已验证的坑(都写进了对应 RUN 的注释,编号与下面一致):
#   [1] CANN env 脚本路径随 base 镜像而异(cann-9.0.0/ vs ascend-toolkit/),
#       统一生成 /etc/profile.d/ascend-cann.sh 做探测,后续所有步骤只 source 它。
#   [2] torch-npu 必须用 PyPI stock 2.10.0,不能用内源版:torch_npu 里打包了
#       torchnpugen(fla_npu 的 gen.sh 用它做 codegen),新版内源 torchnpugen 会生成
#       op_plugin/ops/dvm/lazy_fusion_kernel.h 的 include,fla v26.1.0 没有这个头 → 编译失败。
#   [3] triton-ascend==3.2.1 不在公网 PyPI(公网只有 3.2.0),必须走内网 PIP_INDEX_URL
#       或 https://gitcode.com/Ascend/triton-ascend/releases 下 aarch64 wheel。
#   [4] numpy 要在 vllm 之后重新 pin 回 1.26.0:vllm 不 pin numpy 会拉 2.3.5,
#       numpy-2 ABI 直接打断 triton-ascend;MindSpeed 硬上限 numpy<=1.26.0。
#   [5] CANN 算子编译的 CMake 按绝对路径调 /usr/local/bin/ninja,pip 装的 ninja 不在那儿 → 建软链。
#   [6] latex2sympy2 必须 --no-deps 先装:它硬 pin antlr4-python3-runtime==4.7.2,
#       与 omegaconf 需要的 4.9.* 冲突;让 4.9.3 赢,latex2sympy2 在 4.9.3 上解析正常。
#   [7] vllm-ascend 的 setup.py 会调 npu-smi 探芯片型号,构建机无 NPU 时失败 →
#       显式 ENV SOC_VERSION=ascend910_9391(Atlas A3)让它跳过探测。
#   [8] bisheng 装在 ${CANN}/tools/bisheng_compiler/bin 但 set_env.sh 不加进 PATH,
#       vllm-ascend csrc/build.sh 与 fla build.sh 都 `which bisheng` → 显式加 PATH。
#   [9] cmake 4.4.0 会把 RUNPATH 写成字面量 `$$ORIGIN`,导致运行期
#       libvllm_ascend_kernels.so 找不到(sleep-mode CaMem init_module=None 崩) → patchelf 修回。
#  [10] Megatron-LM 不 pip install,按 slime/vime 的用法走 PYTHONPATH 就地使用。
#  [11] base 镜像若不是 A3 变体,会缺 ascend910_93 算子包,任何 NPU 算子都报
#       561103 Parse dynamic kernel config fail(连 arange 都跑不了)。见文末说明。
#  [12] torch_memory_saver 是 CUDA-only,NPU 上 import 直接 ModuleNotFoundError;
#       rollout 显存 offload 走 vime scripts/npu_mem_offload/ 的 storage-resize hook。
# =============================================================================

# 基础镜像:只需要 CANN 9.0.0(aarch64)。两个已验证选项:
#   - 公网:quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.12   ← -a3- tag 自带 ascend910_93 算子
#   - 内网:ascend-cann900-aarch64-huawei:base(py3.11,当前落地环境用的就是它)
# 下面的版本 pin 在 py3.11 上完整验证过,py3.12 兼容。
ARG BASE_IMAGE=quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.12
FROM ${BASE_IMAGE}

LABEL maintainer="vime"
LABEL description="vime RL training stack (Megatron-LM + MindSpeed + vLLM-Ascend + fla_npu GDN) on Ascend A3 / CANN 9.0.0"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /workspace

# ---------------------------------------------------------------------------
# 构建参数:代理 / pip 源 / 各依赖仓 ref / 开关
# ---------------------------------------------------------------------------
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy
ARG NO_PROXY="localhost,127.0.0.1,::1"
ARG no_proxy
ARG PIP_INDEX_URL
ARG PIP_TRUSTED_HOST
ARG GIT_SSL_NO_VERIFY=1

# 依赖仓(当前 vime 训练栈实际在用的组合:vllm 0.23.0 线)
ARG VIME_REPO=https://github.com/BryanChen408/vime.git
ARG VIME_REF=a3-pd
ARG VLLM_REPO=https://github.com/ljyrj/vllm.git
ARG VLLM_REF=releases/v0.23.0
ARG VLLM_ASCEND_REPO=https://github.com/ljyrj/vllm-ascend-023.git
ARG VLLM_ASCEND_REF=vime-adapter-v023
ARG MEGATRON_LM_REPO=https://github.com/MarkJoson/Megatron-LM.git
ARG MEGATRON_LM_REF=qwen36_vime_adapt
ARG MINDSPEED_REPO=https://gitcode.com/bitszh3271/MindSpeed.git
ARG MINDSPEED_REF=migrate-qwen36-gdn
ARG FLA_REPO=https://github.com/flashserve/flash-linear-attention-npu.git
ARG FLA_REF=v26.1.0

# A3 相关:SoC 型号 + fla_npu 需要落地的 GDN 算子集合
ARG FLA_SOC=ascend910_93
ARG FLA_OPS=chunk_bwd_dv_local,chunk_bwd_dqkwg,chunk_gated_delta_rule_bwd_dhu,prepare_wy_repr_bwd_da,prepare_wy_repr_bwd_full,chunk_fwd_o,chunk_gated_delta_rule_fwd_h,recurrent_gated_delta_rule,recompute_wu_fwd,causal_conv1d
ARG SOC_VERSION=ascend910_9391

# 版本 pin
ARG TORCH_VERSION=2.10.0
ARG TORCH_NPU_VERSION=2.10.0
ARG TRITON_ASCEND_VERSION=3.2.1
# MPICH:dev-base 用的是 3.2.1;这里取 4.2.3(新版 gcc 上更稳),hccl_test 两者皆可
ARG MPICH_VERSION=4.2.3

# 开关:1 = 构建期就编译 NPU 算子(要求构建机是 A3 且 CANN 完整);0 = 两段式(默认)
ARG BUILD_NPU_KERNELS=0
# 开关:1 = 额外编译 Mooncake(仅 PD 分离式 rollout 需要);训练主链路不需要
ARG WITH_MOONCAKE=0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_ROOT_USER_ACTION=ignore \
    HTTP_PROXY=${HTTP_PROXY:-} \
    HTTPS_PROXY=${HTTPS_PROXY:-} \
    http_proxy=${http_proxy:-} \
    https_proxy=${https_proxy:-} \
    NO_PROXY="${NO_PROXY}" \
    no_proxy="${no_proxy}" \
    PIP_INDEX_URL=${PIP_INDEX_URL:-} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST:-} \
    GIT_SSL_NO_VERIFY=${GIT_SSL_NO_VERIFY}

# ===========================================================================
# 1. 系统依赖
#    基础工具 + 编译链 + fla_npu 编译依赖(gawk/pigz/dos2unix/gtest/pkg-config,
#    fla 自带的 install_deps.sh 会卡在 apt 交互提示,所以这里直接装齐)
#    + 训练/排障常用网络工具(HCCL 建链问题全靠这些排)
# ===========================================================================
RUN set -ex; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential gcc g++ make cmake ninja-build pkg-config patchelf \
        git git-lfs curl wget ca-certificates unzip zip tar patch dos2unix rsync \
        gawk pigz bison flex \
        libgtest-dev libgmock-dev \
        libssl-dev libffi-dev zlib1g-dev libbz2-dev libsqlite3-dev libxslt1-dev \
        libnuma-dev libopenblas-dev liblapack-dev libjemalloc2 \
        numactl kmod sshpass openssh-client \
        procps psmisc lsof strace gdb file jq vim less tree tmux htop \
        net-tools iproute2 iputils-ping netcat-openbsd pciutils dnsutils iperf3 \
    ; rm -rf /var/lib/apt/lists/*

# ===========================================================================
# 2. [1][8] CANN 环境统一入口
#    不同 base 镜像 CANN 落点不同(cann-9.0.0 / cann / ascend-toolkit),
#    生成一个探测脚本,后面所有编译步骤和运行期 bashrc 都只 source 它。
#    [8] 顺带把 bisheng 编译器加进 PATH:set_env.sh 不会加,而
#        vllm-ascend csrc/build.sh 和 fla build.sh 都靠 `which bisheng` 判定。
# ===========================================================================
RUN <<'SETUP_CANN_ENV'
set -ex
mkdir -p /etc/profile.d
cat > /etc/profile.d/ascend-cann.sh <<'PROFILE'
# CANN 环境探测(按落点优先级)
for _cann in /usr/local/Ascend/cann-9.0.0 /usr/local/Ascend/cann \
             /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/ascend-toolkit; do
  if [ -f "${_cann}/set_env.sh" ]; then
    . "${_cann}/set_env.sh"
    export CANN_HOME="${_cann}"
    break
  fi
done
unset _cann

# NNAL/ATB(存在才 source)
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  . /usr/local/Ascend/nnal/atb/set_env.sh
fi

# [8] bisheng 不在 set_env.sh 的 PATH 里
for _bi in "${CANN_HOME}/tools/bisheng_compiler/bin" \
           /usr/local/Ascend/ascend-toolkit/latest/tools/bisheng_compiler/bin; do
  # 注意用 if/fi 而不是 `[ -d x ] && export`:后者在 `set -e` 的 RUN 里
  # 一旦目录不存在会让整条 source 失败,直接崩构建。
  if [ -d "${_bi}" ]; then
    export PATH="${_bi}:${PATH}"
  fi
done
unset _bi

# fla_npu 自定义算子 vendor 库(两种 CANN 落点都挂上,不存在的路径无害)
export LD_LIBRARY_PATH="/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib:/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
PROFILE
source /etc/profile.d/ascend-cann.sh
echo "CANN_HOME=${CANN_HOME}"
test -n "${CANN_HOME}"
grep -q ascend-cann /root/.bashrc || echo '. /etc/profile.d/ascend-cann.sh' >> /root/.bashrc
SETUP_CANN_ENV

# ===========================================================================
# 3. pip 源
# ===========================================================================
RUN set -ex; \
    if [ -n "${PIP_INDEX_URL}" ]; then \
        printf '%s\n' '[global]' \
            "index-url = ${PIP_INDEX_URL}" \
            "trusted-host = ${PIP_TRUSTED_HOST}" \
            'timeout = 120' > /etc/pip.conf; \
        cat /etc/pip.conf; \
    fi; \
    pip install --no-cache-dir -U pip setuptools wheel setuptools_rust setuptools_scm nanobind

# ===========================================================================
# 4. [2] PyTorch 栈 — aarch64 原生 wheel(公网 PyPI 就有 aarch64 版本)
#    torch 解析为 2.10.0(+cpu),这正是 torch_npu 绑定的构建。
#    ！！torch-npu 保持 PyPI stock 版本,严禁换内源包(见文件头 [2])。
# ===========================================================================
RUN pip install --no-cache-dir \
        torch==${TORCH_VERSION} \
        torch-npu==${TORCH_NPU_VERSION} \
        torchvision==0.25.0 \
        torchaudio==2.10.0

# [3] triton-ascend:公网 PyPI 只有 3.2.0,3.2.1 需内网源 / 手动 wheel。
#     失败时给出明确指引,不要静默降级(GDN triton kernel 依赖 3.2.1)。
RUN pip install --no-cache-dir "triton-ascend==${TRITON_ASCEND_VERSION}" || { \
        echo "!! triton-ascend==${TRITON_ASCEND_VERSION} 安装失败。"; \
        echo "!! 公网 PyPI 无此版本,请用 --build-arg PIP_INDEX_URL=<内网源>,"; \
        echo "!! 或从 https://gitcode.com/Ascend/triton-ascend/releases 下 aarch64 wheel 手动装。"; \
        exit 1; }

# ===========================================================================
# 5. Python 依赖(按变更频率分层,便于 layer cache)
# ===========================================================================
# 5.1 训练核心
RUN pip install --no-cache-dir \
        accelerate==1.13.0 \
        datasets==4.8.5 \
        einops==0.8.2 \
        gpytorch==1.15.2 \
        jsonargparse==4.49.0 \
        mbridge==0.15.1 \
        safetensors==0.7.0 \
        tokenizers==0.22.2 \
        transformers==5.12.1 \
        huggingface_hub==1.21.0 \
        sentencepiece==0.2.1 \
        peft==0.19.1 \
        timm==1.0.27 \
        scipy==1.13.1 \
        scikit-learn==1.9.0 \
        pandas==3.0.3 \
        numpy==1.26.4 \
        numba \
        ninja==1.13.0 \
        pybind11==3.0.4 \
        protobuf==6.33.6
# 注:transformers 最终会被 vllm-ascend/requirements.txt 拉到 5.5.4(它最后装),属预期。

# 5.2 分布式 / RL / 编排
# [12] torch_memory_saver 是 CUDA-only,NPU 不装;rollout 显存 offload 用
#      vime scripts/npu_mem_offload/ 的 storage-resize hook(verl 式 resize_(0)+empty_cache)。
RUN pip install --no-cache-dir \
        ray[default]==2.48.0 \
        vllm-router==0.1.14 \
        blobfile==3.2.0 \
        wandb==0.27.2 \
        tensorboard \
        memray==1.19.3 \
        tabulate==0.10.0 \
        ring-flash-attn==0.1.8 \
        cloudpickle

# 5.3 工具 / 数据 / reward
# [6] latex2sympy2 先单独 --no-deps 装(它硬 pin antlr 4.7.2,会踩坏 omegaconf)
RUN pip install --no-cache-dir --no-deps latex2sympy2==1.9.1
RUN pip install --no-cache-dir \
        pylatexenc==2.10 \
        omegaconf==2.3.1 \
        antlr4-python3-runtime==4.9.3 \
        openai==2.43.0 \
        pyarrow \
        av==17.0.1 \
        hf_xet==1.5.0 \
        modelscope==1.37.1 \
        aiohttp==3.14.0 \
        fastapi==0.123.10 \
        uvicorn==0.49.0 \
        "httpx[http2]==0.28.1" \
        tiktoken==0.13.0 \
        pyzmq==27.1.0 \
        GitPython==3.1.50 \
        xgrammar==0.2.2 \
        partial-json-parser==0.2.1.1.post7 \
        llguidance==1.3.0 \
        outlines-core==0.2.14 \
        pillow \
        pyyaml

# 5.4 vime agent / SWE 任务侧(vime/requirements.txt 里的 agent 依赖)
RUN pip install --no-cache-dir \
        anthropic \
        openai-agents \
        "mcp[cli]" \
        e2b \
        qwen_vl_utils

# [5] CANN 算子编译的 CMake 按绝对路径调用 /usr/local/bin/ninja
RUN ln -sf "$(command -v ninja)" /usr/local/bin/ninja && /usr/local/bin/ninja --version

# ===========================================================================
# 6. MPICH(hccl_test / 多机连通性诊断用)
# ===========================================================================
RUN set -ex; \
    cd /tmp; \
    wget --no-check-certificate "https://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz"; \
    tar -zxf "mpich-${MPICH_VERSION}.tar.gz"; \
    cd "mpich-${MPICH_VERSION}"; \
    ./configure --disable-fortran --prefix=/usr/local/mpich; \
    make -j"$(nproc)"; \
    make install; \
    rm -rf "/tmp/mpich-${MPICH_VERSION}" "/tmp/mpich-${MPICH_VERSION}.tar.gz"
ENV PATH=/usr/local/mpich/bin:${PATH}
# 注意用 ${VAR:+:${VAR}} 拼接:变量未设时直接跟 `:${VAR}` 会留下尾部冒号,
# 而 LD_LIBRARY_PATH/PYTHONPATH 里的空条目 = 当前目录,会造成难查的加载/导入串味。
ENV LD_LIBRARY_PATH=/usr/local/mpich/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# hccl_test(跨机 HCCL 带宽/建链自测,排 ADXL/RoCE 问题时必用)
RUN set -ex; source /etc/profile.d/ascend-cann.sh; \
    if [ -d "${CANN_HOME}/tools/hccl_test" ]; then \
        cd "${CANN_HOME}/tools/hccl_test" && \
        make MPI_HOME=/usr/local/mpich ASCEND_DIR="${CANN_HOME}" || \
        echo "warn: hccl_test 编译失败(不影响训练),跳过"; \
    fi

# ===========================================================================
# 7. vime 主仓(editable)
#    本地开发建议运行时把工作树挂进来覆盖:-v /path/to/vime:/workspace/vime
#    因为是 editable 安装,挂载后无需重装。
# ===========================================================================
RUN set -ex; source /etc/profile.d/ascend-cann.sh; \
    git clone "${VIME_REPO}" /workspace/vime; \
    cd /workspace/vime; \
    (git checkout --detach "${VIME_REF}" || git checkout --detach "origin/${VIME_REF}"); \
    pip install --no-cache-dir -e /workspace/vime --no-build-isolation

# ===========================================================================
# 8. vLLM(rollout 引擎前端)— VLLM_TARGET_DEVICE=empty,NPU 后端在 vllm-ascend 里
# ===========================================================================
RUN set -ex; source /etc/profile.d/ascend-cann.sh; \
    git clone "${VLLM_REPO}" /workspace/vllm; \
    cd /workspace/vllm; \
    (git checkout --detach "${VLLM_REF}" || git checkout --detach "origin/${VLLM_REF}"); \
    VLLM_TARGET_DEVICE=empty pip install --no-cache-dir -v -e /workspace/vllm --no-build-isolation

# [4] vllm 不 pin numpy,上一步会把 numpy 拉到 2.x,numpy-2 ABI 打断 triton-ascend;
#     MindSpeed 又硬 cap <=1.26.0 → 统一回落 1.26.0(triton 只留一条无害的 pip-check 告警)。
RUN pip install --no-cache-dir "numpy==1.26.0"

# ===========================================================================
# 9. vLLM-Ascend(NPU 后端 + 自定义 AscendC 算子)
#    [7] SOC_VERSION 让 setup.py 跳过 npu-smi 探测(构建机通常没有 NPU)
#    [8] bisheng PATH 由 ascend-cann.sh 提供
#    [9] 编完 patchelf 修 RUNPATH(cmake 4.4.0 会写成字面量 $$ORIGIN)
#    BUILD_NPU_KERNELS=0 时只装 python 部分,算子留给 build_npu_kernels.sh
# ===========================================================================
ENV SOC_VERSION=${SOC_VERSION}
RUN set -ex; source /etc/profile.d/ascend-cann.sh; \
    git clone "${VLLM_ASCEND_REPO}" /workspace/vllm-ascend; \
    cd /workspace/vllm-ascend; \
    (git checkout --detach "${VLLM_ASCEND_REF}" || git checkout --detach "origin/${VLLM_ASCEND_REF}"); \
    git submodule update --init --recursive; \
    rm -rf /workspace/vllm-ascend/csrc/build; \
    find /workspace/vllm-ascend -name CMakeCache.txt -delete; \
    COMPILE_CUSTOM_KERNELS=${BUILD_NPU_KERNELS} \
        pip install -v --no-cache-dir -e /workspace/vllm-ascend --no-build-isolation; \
    if [ "${BUILD_NPU_KERNELS}" = "1" ]; then \
        patchelf --set-rpath '$ORIGIN:$ORIGIN/lib:$ORIGIN/_cann_ops_custom/vendors/custom_transformer/op_api/lib' \
            /workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.*.so; \
    fi

# ===========================================================================
# 10. MindSpeed(Ascend 训练加速库,GDN 分支)— 最后装,会把 numpy cap 在 1.26.0
# ===========================================================================
RUN set -ex; source /etc/profile.d/ascend-cann.sh; \
    git clone "${MINDSPEED_REPO}" /workspace/MindSpeed; \
    cd /workspace/MindSpeed; \
    (git checkout --detach "${MINDSPEED_REF}" || git checkout --detach "origin/${MINDSPEED_REF}"); \
    find /workspace/MindSpeed -name CMakeCache.txt -delete 2>/dev/null || true; \
    pip install -v --no-cache-dir -e /workspace/MindSpeed --no-build-isolation

# ===========================================================================
# 11. [10] Megatron-LM — 不 pip install,按 slime/vime 用法走 PYTHONPATH 就地使用
# ===========================================================================
RUN set -ex; \
    git clone "${MEGATRON_LM_REPO}" /workspace/Megatron-LM; \
    cd /workspace/Megatron-LM; \
    (git checkout --detach "${MEGATRON_LM_REF}" || git checkout --detach "origin/${MEGATRON_LM_REF}")

# ===========================================================================
# 12. fla_npu(flash-linear-attention-npu)— Qwen3.6 GDN 的 AscendC 算子
#     MindSpeed 的 mindspeed/ops/chunk_gated_delta_rule.py 硬 import fla_npu,必须有。
#     [2] codegen(gen.sh)用的是 torch_npu 里打包的 torchnpugen,所以上面必须是 stock torch-npu。
#     BUILD_NPU_KERNELS=0 时只 clone 源码,编译留给 build_npu_kernels.sh。
# ===========================================================================
RUN set -ex; source /etc/profile.d/ascend-cann.sh; \
    git clone "${FLA_REPO}" /workspace/flash-linear-attention-npu; \
    cd /workspace/flash-linear-attention-npu; \
    (git checkout --detach "${FLA_REF}" || git checkout --detach "origin/${FLA_REF}"); \
    if [ "${BUILD_NPU_KERNELS}" = "1" ]; then \
        bash build.sh --soc="${FLA_SOC}" --pkg --ops="${FLA_OPS}" </dev/null; \
        set -- build_out/*.run; bash "$1" --quiet --install-for-all </dev/null; \
        cd torch_custom/fla_npu; \
        bash gen.sh npu_custom.yaml; \
        python setup.py bdist_wheel; \
        pip install dist/fla_npu*.whl --force-reinstall --no-deps; \
    fi

# ===========================================================================
# 13. 第二阶段脚本:在 A3 机器上首次起容器时跑一次,补齐两个需要真实 NPU 环境的编译。
#     跑完 `docker commit <container> vime:a3-cann900-kernels` 固化即可。
# ===========================================================================
RUN cat > /workspace/build_npu_kernels.sh <<'EOF' && chmod +x /workspace/build_npu_kernels.sh
#!/bin/bash
# 在带 NPU 的 A3 机器上、容器内执行一次:编译 vllm-ascend 自定义算子 + fla_npu GDN 算子。
set -ex
source /etc/profile.d/ascend-cann.sh
: "${SOC_VERSION:=ascend910_9391}"
: "${FLA_SOC:=ascend910_93}"
: "${FLA_OPS:=chunk_bwd_dv_local,chunk_bwd_dqkwg,chunk_gated_delta_rule_bwd_dhu,prepare_wy_repr_bwd_da,prepare_wy_repr_bwd_full,chunk_fwd_o,chunk_gated_delta_rule_fwd_h,recurrent_gated_delta_rule,recompute_wu_fwd,causal_conv1d}"
export SOC_VERSION

# [8] bisheng:ascend-cann.sh 已经尽量加过 PATH,这里再兜一次底(find 兜底路径)
if ! command -v bisheng >/dev/null; then
  bi="$(find /usr/local/Ascend -type f -name bisheng -path '*bisheng_compiler/bin/*' 2>/dev/null | head -1)"
  if [ -n "${bi}" ]; then
    export PATH="$(dirname "${bi}"):${PATH}"
  fi
fi
command -v bisheng

# --- vllm-ascend 自定义 AscendC 算子 ---
cd /workspace/vllm-ascend
rm -rf csrc/build
find . -name CMakeCache.txt -delete
COMPILE_CUSTOM_KERNELS=1 pip install -v --no-cache-dir -e . --no-build-isolation
# [9] cmake 4.4.0 会把 RUNPATH 写成字面量 $$ORIGIN → 运行期找不到 libvllm_ascend_kernels.so
patchelf --set-rpath '$ORIGIN:$ORIGIN/lib:$ORIGIN/_cann_ops_custom/vendors/custom_transformer/op_api/lib' \
    /workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.*.so

# --- fla_npu GDN 算子 ---
cd /workspace/flash-linear-attention-npu
bash build.sh --soc="${FLA_SOC}" --pkg --ops="${FLA_OPS}" </dev/null
bash "$(ls build_out/*.run | head -1)" --quiet --install-for-all </dev/null
cd torch_custom/fla_npu
bash gen.sh npu_custom.yaml
python setup.py bdist_wheel
pip install dist/fla_npu*.whl --force-reinstall --no-deps

# --- 自检 ---
cd /workspace
python -c "import torch, torch_npu; print('torch', torch.__version__, '| npu count', torch_npu.npu.device_count())"
python -c "import fla_npu; import mindspeed.ops.chunk_gated_delta_rule; print('fla_npu + mindspeed GDN: OK')"
python -c "import vllm, vllm_ascend; print('vllm', vllm.__version__, '| vllm_ascend OK')"
echo "=== NPU kernels ready. 记得 docker commit 固化 ==="
EOF

# ===========================================================================
# 14. (可选)Mooncake — 只有 PD 分离式 rollout 才需要;训练主链路不需要。
#     --build-arg WITH_MOONCAKE=1 打开。安装到 CANN 的 site-packages 下。
# ===========================================================================
ARG GO_VERSION=1.23.8
ARG YALANTINGLIBS_VERSION=0.5.7
ARG MOONCAKE_VERSION=v0.3.9
ENV GOPROXY=https://goproxy.cn,direct
ENV GO111MODULE=on
RUN if [ "${WITH_MOONCAKE}" = "1" ]; then set -ex; \
        apt-get update; \
        apt-get install -y --no-install-recommends \
            libboost-all-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
            libcurl4-openssl-dev libhiredis-dev liburing-dev libjemalloc-dev libasio-dev \
            libyaml-cpp-dev libjsoncpp-dev libgoogle-glog-dev libibverbs-dev; \
        rm -rf /var/lib/apt/lists/*; \
        cd /tmp && wget --no-check-certificate "https://mirrors.tuna.tsinghua.edu.cn/golang/go${GO_VERSION}.linux-arm64.tar.gz" && \
            tar -C /usr/local -xzf "go${GO_VERSION}.linux-arm64.tar.gz" && rm "go${GO_VERSION}.linux-arm64.tar.gz"; \
        export PATH=/usr/local/go/bin:${PATH}; go version; \
        cd /tmp && git clone --depth 1 --branch "${YALANTINGLIBS_VERSION}" https://github.com/alibaba/yalantinglibs.git && \
            cd yalantinglibs && mkdir build && cd build && \
            cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF && \
            cmake --build . -j"$(nproc)" && cmake --install . && cd /tmp && rm -rf yalantinglibs; \
        source /etc/profile.d/ascend-cann.sh; \
        git clone --depth 1 --branch "${MOONCAKE_VERSION}" https://github.com/kvcache-ai/Mooncake.git /workspace/Mooncake && \
            cd /workspace/Mooncake && git submodule update --init --recursive && \
            mkdir -p build && cd build && cmake .. -DUSE_ASCEND_DIRECT=ON && \
            make -j"$(nproc)" && make install; \
    else \
        echo "skip Mooncake (WITH_MOONCAKE=0);PD 分离式 rollout 才需要,用 --build-arg WITH_MOONCAKE=1 打开"; \
    fi
ENV PATH=/usr/local/go/bin:${PATH}

# ===========================================================================
# 15. 运行期环境变量
#     只放「与硬件/框架强绑定、改了就跑不起来」的;调优类留给 scripts/ 里的启动脚本。
# ===========================================================================
# [10] Megatron-LM 就地使用;vime/MindSpeed 虽是 editable,一并显式挂上更稳
ENV PYTHONPATH=/workspace/Megatron-LM:/workspace/MindSpeed:/workspace/vime${PYTHONPATH:+:${PYTHONPATH}}
# fla_npu vendor 算子库(两种 CANN 落点都挂,缺的那条无害)
ENV LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib:/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# vllm-ascend 的 vllm_version_is() 判据:本地 vllm 比 v0.23.0 tag 多几个提交,
# 版本号自报 0.23.1.devN,不显式指定会走 vllm-main-only 的 import 分支 → ModuleNotFoundError
ENV VLLM_VERSION=0.23.0
# NZ 格式与 RL 每步换权重冲突(wake_up 硬 raise + 精度崩),必须 0
ENV VLLM_ASCEND_ENABLE_NZ=0
# =1 会让 GDN / ring-attention 训练出 NaN
ENV TASK_QUEUE_ENABLE=0
# 昇腾 inductor get_gpu_type() 断言 → 强制 eager
ENV TORCHDYNAMO_DISABLE=1
ENV PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
ENV RAY_DEDUP_LOGS=1
ENV HYDRA_FULL_ERROR=1
ENV CPU_AFFINITY_CONF=1

# HCCL:只固定端口范围和超时,不预设网卡/协议
ENV HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
ENV HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
ENV HCCL_SOCKET_FAMILY=AF_INET
ENV HCCL_CONNECT_TIMEOUT=600
ENV HCCL_EXEC_TIMEOUT=2400
ENV HCCL_WHITELIST_DISABLE=1
ENV HCCL_BUFFSIZE=512
# 注意:故意不设 HCCL_INTRA_ROCE_ENABLE / HCCL_INTRA_PCIE_ENABLE。
#      设成 RoCE 后 PD KV 传输在偶数卡上会建链超时(103902),不设则走 HCCS 全通。
#      需要时在启动脚本里按机器实测结果设,不要焊死在镜像里。
# HCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME 同理:由启动脚本按机器网卡名设置。

RUN mkdir -p /workspace/logs /workspace/output /workspace/wandb_logs

# 清掉构建期代理,避免泄漏进运行容器
ENV http_proxy= https_proxy= no_proxy= HTTP_PROXY= HTTPS_PROXY= NO_PROXY=

# ===========================================================================
# 16. 构建期自检(只查不需要 NPU 设备的部分)
# ===========================================================================
RUN set -ex; source /etc/profile.d/ascend-cann.sh; \
    python -c "import torch; print('torch:', torch.__version__)"; \
    python -c "import transformers, ray, numpy; print('transformers:', transformers.__version__, '| ray:', ray.__version__, '| numpy:', numpy.__version__)"; \
    python -c "import importlib.metadata as m; print('triton-ascend:', m.version('triton-ascend'))"; \
    python -c "import triton; print('triton(ascend) module:', triton.__version__)"; \
    test -d /workspace/Megatron-LM/megatron; \
    test -f /workspace/build_npu_kernels.sh; \
    echo "=== build-time verification OK ==="

WORKDIR /workspace
CMD ["/bin/bash"]

# =============================================================================
# 首次在 A3 机器上使用
# -----------------------------------------------------------------------------
# 1) 起容器(注意挂 driver / dcmi / npu-smi,以及 16 张卡的 device):
#    docker run -it --name vime --network host --ipc host --shm-size 128g --privileged \
#      $(for i in $(seq 0 15); do echo --device=/dev/davinci$i; done) \
#      --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
#      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
#      -v /usr/local/dcmi:/usr/local/dcmi \
#      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
#      -v /path/to/models:/workspace/models -v /path/to/data:/workspace/data \
#      vime:a3-cann900 bash
#
# 2) 补编译 NPU 算子(BUILD_NPU_KERNELS=0 构建时必做,只需一次):
#    bash /workspace/build_npu_kernels.sh
#    docker commit vime vime:a3-cann900-kernels
#
# 3) [11] 若任何 NPU 算子报 561103 "Parse dynamic kernel config fail"(连 arange 都失败),
#    说明 base 镜像缺 A3(ascend910_93)算子包。检查:
#      ls ${CANN_HOME}/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910_93/
#      # 必须有 ops_math / ops_nn / ops_cv,只有 ops_oam 就是缺
#    补装:
#      bash Ascend-cann-A3-ops_9.0.0_linux-aarch64.run --install --install-for-all --quiet \
#        --install-path=/usr/local/Ascend/cann-9.0.0 \
#        --feature=aclnn_ops_infer,aclnn_ops_train,aclnn_math,aclnn_rand
#    用 quay.io/ascend/cann:9.0.0-a3-* 作 base 则无此问题。
#
# 4) 跑训练:
#    cd /workspace/vime && bash scripts/<你的启动脚本>.sh
#    启动脚本自己会设 ASCEND_RT_VISIBLE_DEVICES / HCCL_SOCKET_IFNAME / MASTER_ADDR 等。
#    ⚠ 注意脚本里的 `export VLLM_VERSION=` 必须与镜像里的 vllm 版本线一致:
#      本镜像默认 0.23.0 线(scripts/run-qwen36-35b-polar-*-pd.sh 等已写 0.23.0),
#      而 scripts/run_qwen36_35b_a3b_dapo_math*.sh / run-qwen36-35b-polar-minimal.sh
#      仍写死 0.21.0 —— 用这些脚本前要么改脚本,要么把 VLLM_REF/VLLM_ASCEND_REF
#      改回 0.21.0 线重新构建镜像。
# =============================================================================
