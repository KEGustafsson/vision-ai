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
        echo "deepstream-entrypoint: WARN could not write $dest — make the host" \
             "deepstream/ dir writable by UID 10001, or place the ONNX yourself" >&2
    fi
fi
exec "$@"
