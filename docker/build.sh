#!/bin/bash
# Build script for vLLM PD-Mooncake Docker image

set -e

# ================================
# Configuration
# ================================
IMAGE_NAME="vllm-pd-mooncake"
IMAGE_TAG="v0.23.0-ascend"
BASE_IMAGE="${BASE_IMAGE:-ascendhub.huawei.com/public-ascendhub/ascend-mindspore:24.0.RC3-centos7.6}"

DOCKERFILE="Dockerfile.vllm-pd-mooncake"
BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Building vLLM PD-Mooncake Docker Image"
echo "=========================================="
echo ""
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Base: ${BASE_IMAGE}"
echo "Dockerfile: ${DOCKERFILE}"
echo ""

# ================================
# Prepare build context
# ================================
echo "== Preparing build context =="

# Check if patch exists
PATCH_DIR="${BUILD_DIR}/patches"
mkdir -p "${PATCH_DIR}"

if [ -f "/workspace/vime-pd-patches-20260808/01-vllm-ascend-v023-compat.patch" ]; then
    cp /workspace/vime-pd-patches-20260808/01-vllm-ascend-v023-compat.patch "${PATCH_DIR}/"
    echo "✅ Copied vllm-ascend patch"
else
    echo "⚠️  Warning: vllm-ascend patch not found, will skip"
    touch "${PATCH_DIR}/01-vllm-ascend-v023-compat.patch"
fi

# Check if requirements file exists
if [ ! -f "${BUILD_DIR}/requirements-vllm-pd.txt" ]; then
    echo "❌ Error: requirements-vllm-pd.txt not found"
    exit 1
fi

echo "✅ Build context ready"
echo ""

# ================================
# Build Docker image
# ================================
echo "== Building Docker image =="
echo ""

docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg GO_VERSION=1.23.8 \
    --build-arg MPICH_VERSION=4.2.3 \
    --build-arg YALANTINGLIBS_VERSION=0.5.7 \
    --build-arg MOONCAKE_VERSION=v0.3.9 \
    --build-arg VLLM_ASCEND_VERSION=v0.23.0 \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -t "${IMAGE_NAME}:latest" \
    -f "${DOCKERFILE}" \
    "${BUILD_DIR}"

BUILD_STATUS=$?

echo ""
echo "=========================================="
if [ ${BUILD_STATUS} -eq 0 ]; then
    echo "✅ Build successful!"
    echo "=========================================="
    echo ""
    echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo "Run container:"
    echo "  docker run -it --rm \\"
    echo "    --device=/dev/davinci0 \\"
    echo "    --device=/dev/davinci1 \\"
    echo "    --device=/dev/davinci2 \\"
    echo "    --device=/dev/davinci3 \\"
    echo "    --device=/dev/davinci_manager \\"
    echo "    --device=/dev/devmm_svm \\"
    echo "    --device=/dev/hisi_hdc \\"
    echo "    -v /usr/local/dcmi:/usr/local/dcmi \\"
    echo "    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \\"
    echo "    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \\"
    echo "    ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo "Test environment:"
    echo "  docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} \\"
    echo "    python3 -c 'import vllm, vllm_ascend, mooncake; print(\"OK\")'"
else
    echo "❌ Build failed!"
    echo "=========================================="
    exit 1
fi
