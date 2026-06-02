#!/usr/bin/env bash
# Build the deepstream-yolo custom nvinfer parser (libnvdsinfer_custom_impl_Yolo.so)
# ON THE HOST and drop it into vision-service/deepstream/, where Dockerfile.deepstream
# bakes it into the image.
#
# Why on the host: the DeepStream 7.1 samples base image has no nvcc, and its only
# apt source (l4t-repo.nvidia.com) is unreachable off the Jetson. The host JetPack
# has nvcc + the DS 7.1 SDK at the SAME versions as the container (CUDA 12.6 / DS 7.1
# / aarch64), so the .so is ABI-compatible with the runtime image.
#
# Re-run this whenever the DeepStream version or the parser repo changes.
#
#   vision-service/scripts/build_yolo_parser.sh
set -euo pipefail

REPO="${DS_YOLO_REPO:-https://github.com/marcoslucianops/DeepStream-Yolo}"
# Pin to a known-good commit; override with DS_YOLO_REF=<tag-or-sha> to upgrade.
REF="${DS_YOLO_REF:-68769f3}"
EXPECTED_SYMBOL="${EXPECTED_SYMBOL:-NvDsInferParseYolo}"
CUDA_VER="${CUDA_VER:-$(/usr/local/cuda/bin/nvcc --version | grep -oP 'V\K[0-9]+\.[0-9]+')}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/deepstream"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "Cloning $REPO @ $REF ..."
git clone --depth 1 --branch "$REF" "$REPO" "$WORK/ds-yolo" 2>/dev/null || {
    echo "Branch/tag clone failed; trying shallow clone + fetch ..."
    git clone --depth 1 "$REPO" "$WORK/ds-yolo"
    (cd "$WORK/ds-yolo" && git fetch --depth 1 origin "$REF" && git checkout FETCH_HEAD) || {
        echo "ERROR: Could not checkout pinned ref '$REF'" >&2
        exit 1
    }
}

echo "Building parser (CUDA_VER=$CUDA_VER) ..."
PATH="/usr/local/cuda/bin:$PATH" CUDA_VER="$CUDA_VER" \
    make -C "$WORK/ds-yolo/nvdsinfer_custom_impl_Yolo"

cp "$WORK/ds-yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so" "$DEST/"
echo "Installed -> $DEST/libnvdsinfer_custom_impl_Yolo.so"

nm -D "$DEST/libnvdsinfer_custom_impl_Yolo.so" \
    | grep -qE " T .*${EXPECTED_SYMBOL}$" || {
    echo "ERROR: expected parser symbol '${EXPECTED_SYMBOL}' not found in .so" >&2
    echo "Check DS_YOLO_REF or update parse-bbox-func-name in pgie_yolov8n.txt" >&2
    exit 1
}
echo "Parser symbol '${EXPECTED_SYMBOL}' verified."
