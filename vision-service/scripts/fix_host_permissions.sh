#!/usr/bin/env bash
# Grants the DeepStream container's non-root user (UID 10001) write access to
# the bind-mounted deepstream/ and models/ host dirs, via ACL rather than
# chmod/chown so the owning host user keeps full access too. Needed because
# nvinfer seeds/writes the ONNX and TRT engine into these dirs at container
# startup (see docs/jetson-setup.md, "Non-root").
set -euo pipefail

if ! command -v setfacl >/dev/null 2>&1; then
    echo "error: setfacl not found (install acl package: apt install acl)" >&2
    exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dirs=(
    "$repo_root/vision-service/deepstream"
    "$repo_root/vision-service/models"
)

for dir in "${dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "skip: $dir does not exist" >&2
        continue
    fi
    setfacl -m u:10001:rwx "$dir"
    setfacl -d -m u:10001:rwx "$dir"
    echo "ok: UID 10001 can read/write $dir"
done
