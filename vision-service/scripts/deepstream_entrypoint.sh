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
seed=/opt/vision-service/deepstream-default/yolo11n_ds.onnx
dest=/opt/vision-service/deepstream/yolo11n_ds.onnx
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
