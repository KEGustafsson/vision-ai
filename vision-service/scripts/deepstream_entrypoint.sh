#!/bin/sh
# Container entrypoint for BOTH DeepStream images (Dockerfile.deepstream on
# JetPack 6 / DeepStream 7.1, Dockerfile.deepstream.xavier on JetPack 5 /
# DeepStream 6.3).
#
# Compose bind-mounts the host's deepstream/ over the image's copy so the built
# TensorRT engine persists across recreates and models can be swapped without a
# rebuild. That mount also SHADOWS the default COCO ONNX baked into the image —
# and the *.onnx is gitignored, so a fresh clone's host deepstream/ has none.
# The image therefore stages a second copy outside the mount (deepstream-default/)
# and this script seeds it in on first run, which is what makes
# `docker compose ... up -d --build` work from a clean clone in ONE command.
#
# It is deliberately a committed file rather than a Dockerfile heredoc: heredocs
# need the BuildKit dockerfile 1.4+ frontend, which the JetPack 5 boxes' older
# Docker does not have.
set -eu
# Paths are overridable only so tests can run this script on a scratch tree.
ds_dir=${VISION_DS_DIR:-/opt/vision-service/deepstream}
image_dir=${VISION_DS_IMAGE_DIR:-/opt/vision-service/deepstream-image}
seed=${VISION_DS_SEED:-/opt/vision-service/deepstream-default/yolo11n_ds.onnx}
dest=$ds_dir/yolo11n_ds.onnx

# Release deployments (deploy/docker-compose.deepstream*.yml) keep deepstream/
# in a named Docker volume, which Docker fills from the image only while it is
# empty. After an image update the volume would still hold the previous image's
# ONNX, pgie configs and labels, and nvinfer would quietly rebuild an engine
# from that old ONNX rather than fail. Those compose files therefore set
# VISION_DS_SYNC_FROM_IMAGE=1: when the image's build identity differs from the
# one recorded in the volume, the image's own files are copied back in and the
# old engines removed, so the new image runs what it shipped with. Files the
# user added (another model's ONNX, its engine) are kept. Source builds, which
# bind-mount the git checkout's deepstream/, never set it: their files belong
# to the checkout.
if [ "${VISION_DS_SYNC_FROM_IMAGE:-0}" = "1" ] && [ -f "$image_dir/.image-id" ]; then
    want=$(cat "$image_dir/.image-id")
    have=$(cat "$ds_dir/.image-id" 2>/dev/null || true)
    if [ "$want" != "$have" ]; then
        # Fail fast (set -e) rather than start on a half-refreshed directory.
        rm -f "$ds_dir"/*.engine
        cp -R "$image_dir"/. "$ds_dir"/
        echo "deepstream-entrypoint: refreshed $ds_dir from the image" \
             "(image ${want}, volume had ${have:-none}); old engines removed"
    fi
fi

if [ ! -f "$dest" ] && [ -f "$seed" ]; then
    if cp "$seed" "$dest" 2>/dev/null; then
        echo "deepstream-entrypoint: seeded default ONNX -> $dest"
    else
        # FAIL FAST, do not warn-and-continue. The seed is only attempted when
        # the destination is missing, so reaching here means the mount has no
        # model AND we cannot write one. Every pgie config resolves its
        # onnx-file inside this directory, so nvinfer would fail at pipeline
        # start and the service would come up with NO DETECTOR — the exact
        # fail-open a safety-relevant sensor must not have. Better to exit
        # non-zero: compose restarts the container and the reason stays at the
        # top of `docker logs` instead of scrolling past as a warning.
        echo "deepstream-entrypoint: ERROR cannot write $dest and no model is" \
             "present there. nvinfer would start without a model." >&2
        echo "deepstream-entrypoint: fix with" \
             "vision-service/scripts/fix_host_permissions.sh (grants UID 10001" \
             "write access to the bind-mounted deepstream/ and models/ dirs)," \
             "or place the ONNX in deepstream/ yourself." >&2
        exit 1
    fi
fi
exec "$@"
