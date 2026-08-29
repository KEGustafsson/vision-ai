#!/usr/bin/env bash
# Build the deepstream-yolo custom nvinfer parser (libnvdsinfer_custom_impl_Yolo.so)
# ON THE HOST and drop it into vision-service/deepstream/, where Dockerfile.deepstream
# bakes it into the image.
#
# JETPACK 6 ONLY. The JetPack 5 image (Dockerfile.deepstream.xavier) compiles its
# own parser inside the build — its DeepStream 6.3 / CUDA 11.4 / TensorRT 8.5 .so
# is a different ABI and must NOT be committed over the one this script produces.
# Both images install the parser at /opt/vision-service/lib/, which is what
# deepstream/pgie_*.txt names in custom-lib-path.
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
# Full 40-char SHA (not a short SHA): upstream rewrote history and dropped the old
# short pin 68769f3 entirely, so pin the immutable full commit. The parser sources
# under nvdsinfer_custom_impl_Yolo/ are byte-unchanged across these commits, so the
# .so this builds matches the ONNX exported at the SAME commit. Keep this in
# lockstep with DS_YOLO_REF in Dockerfile.deepstream (the ONNX export uses it too).
REF="${DS_YOLO_REF:-93aedb656a47b141ecbea99c407b002262287cfe}"
EXPECTED_SYMBOL="${EXPECTED_SYMBOL:-NvDsInferParseYolo}"
CUDA_VER="${CUDA_VER:-$(/usr/local/cuda/bin/nvcc --version | grep -oP 'V\K[0-9]+\.[0-9]+')}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/deepstream"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "Cloning $REPO @ $REF ..."
# Full clone (not shallow): lets us check out any REF — branch, tag, or a short SHA,
# which a shallow `fetch <short-sha>` cannot resolve on GitHub. checkout failures
# abort via `set -e` (no `|| true`).
git clone "$REPO" "$WORK/ds-yolo"
git -C "$WORK/ds-yolo" checkout --quiet "$REF"
# Fail loudly if we did not land exactly on $REF: building the parser from an
# unpinned commit must not be allowed to slip through silently.
if [ "$(git -C "$WORK/ds-yolo" rev-parse HEAD)" \
     != "$(git -C "$WORK/ds-yolo" rev-parse --verify "${REF}^{commit}")" ]; then
    echo "ERROR: HEAD is not the pinned ref '$REF' after checkout" >&2
    exit 1
fi

echo "Building parser (CUDA_VER=$CUDA_VER) ..."
PATH="/usr/local/cuda/bin:$PATH" CUDA_VER="$CUDA_VER" \
    make -C "$WORK/ds-yolo/nvdsinfer_custom_impl_Yolo"

cp "$WORK/ds-yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so" "$DEST/"
echo "Installed -> $DEST/libnvdsinfer_custom_impl_Yolo.so"

# Verify the parser exports the symbol pgie_yolo11n.txt names in parse-bbox-func-name.
# Use `grep -c` (counts ALL input) rather than `grep -q` (exits at the first match):
# under `set -o pipefail`, grep -q closing the pipe early SIGPIPEs nm (exit 141),
# which pipefail would then report as a spurious "symbol not found". `|| true`
# keeps grep's exit-1-on-zero-matches from tripping `set -e`.
found="$(nm -D "$DEST/libnvdsinfer_custom_impl_Yolo.so" \
    | grep -cE " T .*${EXPECTED_SYMBOL}$")" || true
if [ "${found:-0}" -eq 0 ]; then
    echo "ERROR: expected parser symbol '${EXPECTED_SYMBOL}' not found in .so" >&2
    echo "Check DS_YOLO_REF or update parse-bbox-func-name in pgie_yolo11n.txt" >&2
    exit 1
fi
echo "Parser symbol '${EXPECTED_SYMBOL}' verified."
