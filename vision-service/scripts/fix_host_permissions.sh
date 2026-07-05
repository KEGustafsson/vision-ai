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
    # -R covers pre-existing files/subdirs too, not just ones created after
    # this runs. setfacl reports "Operation not permitted" per-entry (without
    # aborting the recursion) for anything owned by a different UID we can't
    # re-ACL non-root (e.g. UID 10001's own prior output) -- harmless, since
    # that UID already has owner access to its own files. Don't let those
    # expected per-file warnings (non-zero exit) abort the rest of this script.
    setfacl -R -m u:10001:rwx "$dir" || true
    setfacl -R -d -m u:10001:rwx "$dir" || true
    echo "ok: UID 10001 can read/write $dir (existing + future files)"
done
