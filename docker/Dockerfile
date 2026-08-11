# Dockerfile for vLLM v0.23.0 + vLLM-Ascend v0.23.0 + Mooncake v0.3.9 PD Architecture
# Base image: Assumes CANN 9.0.0, Python 3.11, and torch-npu are pre-installed

ARG BASE_IMAGE=ascendhub.huawei.com/public-ascendhub/ascend-mindspore:24.0.RC3-centos7.6
FROM ${BASE_IMAGE}

LABEL maintainer="vime-pd-mooncake"
LABEL description="vLLM v0.23.0 PD-Mooncake environment with Ascend NPU support"

# ================================
# Environment Setup
# ================================
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Go mirror for China (Tsinghua)
ENV GOPROXY=https://goproxy.cn,direct
ENV GO111MODULE=on

# Python package mirrors (optional, uncomment if needed)
# ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# ================================
# System Dependencies
# ================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    ca-certificates \
    # Mooncake dependencies
    libboost-all-dev \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler-grpc \
    libcurl4-openssl-dev \
    libhiredis-dev \
    liburing-dev \
    libjemalloc-dev \
    libasio-dev \
    pkg-config \
    patchelf \
    libc6-dev \
    libc-bin \
    libyaml-cpp-dev \
    libjsoncpp-dev \
    libgoogle-glog-dev \
    libibverbs-dev \
    # vLLM dependencies
    libssl-dev \
    # Network tools
    net-tools \
    iproute2 \
    iputils-ping \
    netcat \
    # Utilities
    vim \
    htop \
    tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ================================
# Install Go 1.23.8 (Mooncake requirement)
# ================================
ARG GO_VERSION=1.23.8
RUN echo "Installing Go ${GO_VERSION} from Tsinghua mirror..." && \
    cd /tmp && \
    wget --no-check-certificate https://mirrors.tuna.tsinghua.edu.cn/golang/go${GO_VERSION}.linux-arm64.tar.gz && \
    tar -C /usr/local -xzf go${GO_VERSION}.linux-arm64.tar.gz && \
    rm go${GO_VERSION}.linux-arm64.tar.gz

ENV PATH=/usr/local/go/bin:${PATH}
ENV GOPATH=/root/go

# Verify Go installation
RUN go version

# ================================
# Install MPI (for Mooncake)
# ================================
ARG MPICH_VERSION=4.2.3
RUN cd /tmp && \
    wget https://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz && \
    tar -xzf mpich-${MPICH_VERSION}.tar.gz && \
    cd mpich-${MPICH_VERSION} && \
    ./configure --prefix=/usr/local --disable-fortran && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && rm -rf mpich-${MPICH_VERSION}*

# ================================
# Python Dependencies
# ================================
COPY requirements.txt /tmp/
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# ================================
# Install vLLM v0.23.0 (editable)
# ================================
WORKDIR /workspace
ARG VLLM_REPO=https://github.com/ljyrj/vllm.git
ARG VLLM_BRANCH=releases/v0.23.0
RUN git clone --depth 1 --branch ${VLLM_BRANCH} ${VLLM_REPO} vllm && \
    cd vllm && \
    pip install -e . --no-build-isolation

# ================================
# Install vLLM-Ascend v0.23.0 (editable)
# ================================
ARG VLLM_ASCEND_REPO=https://github.com/ljyrj/vllm-ascend-023.git
ARG VLLM_ASCEND_BRANCH=vime-adapter-v023
RUN git clone --depth 1 --branch ${VLLM_ASCEND_BRANCH} ${VLLM_ASCEND_REPO} vllm-ascend


# Build and install (with CANN environment)
RUN bash -c 'source /usr/local/Ascend/cann-9.0.0/set_env.sh && \
    cd /workspace/vllm-ascend && \
    export SOC_VERSION=ascend910_9391 && \
    pip install -e . --no-build-isolation --no-deps'

# Fix RPATH for cmake 4.4.0 bug (if needed)
RUN patchelf --set-rpath '$ORIGIN:$ORIGIN/lib:$ORIGIN/_cann_ops_custom/vendors/custom_transformer/op_api/lib' \
    /workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.*.so 2>/dev/null || true

# Set VLLM_VERSION for patch compatibility
ENV VLLM_VERSION=0.23.0

# ================================
# Install yalantinglibs (Mooncake dependency)
# ================================
ARG YALANTINGLIBS_VERSION=0.5.7
RUN cd /tmp && \
    git clone --depth 1 --branch ${YALANTINGLIBS_VERSION} https://github.com/alibaba/yalantinglibs.git && \
    cd yalantinglibs && \
    mkdir build && cd build && \
    cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF && \
    cmake --build . -j$(nproc) && \
    cmake --install . && \
    cd /tmp && rm -rf yalantinglibs

# ================================
# Install Mooncake v0.3.9
# ================================
ARG MOONCAKE_VERSION=v0.3.9
RUN cd /workspace && \
    git clone --depth 1 --branch ${MOONCAKE_VERSION} https://github.com/kvcache-ai/Mooncake.git && \
    cd Mooncake && \
    git submodule update --init --recursive

# Build Mooncake with ASCEND_DIRECT
RUN bash -c 'source /usr/local/Ascend/cann-9.0.0/set_env.sh && \
    cd /workspace/Mooncake && \
    mkdir -p build && cd build && \
    cmake .. -DUSE_ASCEND_DIRECT=ON && \
    make -j$(nproc) && \
    make install'

# ================================
# Install vime (if applicable)
# ================================
# Uncomment if vime is needed in the image
# COPY vime /workspace/vime
# RUN cd /workspace/vime && pip install -e .

# ================================
# Setup CANN Environment
# ================================
ENV ASCEND_ROOT=/usr/local/Ascend
ENV CANN_ROOT=${ASCEND_ROOT}/cann
ENV CANN_TOOLKIT_ROOT=${ASCEND_ROOT}/ascend-toolkit/latest
ENV PATH=${CANN_ROOT}/bin:${CANN_TOOLKIT_ROOT}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CANN_ROOT}/lib64:${CANN_TOOLKIT_ROOT}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages:${CANN_ROOT}/python/site-packages:${PYTHONPATH}

# ================================
# Runtime Configuration
# ================================
# Disable wandb by default
ENV USE_WANDB=0

# Set working directory
WORKDIR /workspace

# Create necessary directories
RUN mkdir -p /workspace/logs \
             /workspace/output \
             /home/docker/logs \
             /workspace/wandb_logs

# ================================
# Verification
# ================================
RUN echo "=== Environment Verification ===" && \
    python3 -c "import vllm; print('vllm version:', vllm.__version__)" && \
    python3 -c "import vllm_ascend; print('vllm_ascend: OK')" && \
    python3 -c "import mooncake; print('mooncake: OK')" && \
    go version && \
    which cmake && cmake --version | head -1 && \
    echo "=== Verification Complete ==="

# ================================
# Entrypoint
# ================================
# Source CANN environment on container start
RUN echo 'source /usr/local/Ascend/cann-9.0.0/set_env.sh 2>/dev/null' >> /root/.bashrc

CMD ["/bin/bash"]
