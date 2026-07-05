"""Download Ultralytics weights to ``vision-service/models/<model>`` deterministically.

Ultralytics auto-downloads on first use, but it writes into its own private cache
(or the current working directory), NOT the path the rest of the system expects.
The DeepStream image build's export stage looks for ``models/yolo11n.pt``, the
TensorRT engine export (``export_engine.py --weights models/yolov8n.pt``) looks
for ``models/yolov8n.pt``, and the docs key off both — so this writes the weights
to exactly the path the caller asks for rather than leaving them in a cache the
next step can't find.

Usage::

    python3 scripts/download_models.py                 # -> models/yolov8n.pt (jetson/marine backend)
    python3 scripts/download_models.py --model yolo11n.pt   # -> models/yolo11n.pt (deepstream backend)
    python3 scripts/download_models.py --model yolov8s.pt
    python3 scripts/download_models.py --model /path/to/local.pt   # copies it in
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"

# Known-good content hashes for pinned model artifacts, so the offline/download
# flow enforces the same provenance the DeepStream image build does
# (Dockerfile.deepstream ARG YOLO11N_SHA256, left empty/unpinned by default).
# Keep these in sync.
KNOWN_SHA256 = {
    "yolov8n.pt": "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(dest: Path, skip: bool) -> None:
    """Verify dest against its pinned hash (if one is known and not skipped)."""
    expected = KNOWN_SHA256.get(dest.name)
    if skip or expected is None:
        return
    actual = _sha256(dest)
    if actual != expected:
        raise SystemExit(
            f"{dest.name} sha256 mismatch: expected {expected}, got {actual}. "
            f"The upstream artifact changed (or the file is corrupt). If this is an "
            f"intentional model update, update KNOWN_SHA256 and the matching "
            f"pinned SHA256 build-arg in whichever Dockerfile consumes it, or "
            f"pass --no-verify."
        )
    print(f"verified {dest.name} sha256 {actual}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default="yolov8n.pt",
        help="A known Ultralytics asset name (e.g. yolov8n.pt) to download, or a "
             "path to an existing .pt file to copy into models/.",
    )
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the sha256 provenance check (see KNOWN_SHA256).",
    )
    args = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    dest = MODELS / Path(args.model).name

    # Case 1: an existing local file (offline boats hand-place weights) — copy it
    # to the canonical path so downstream steps find it there.
    src_path = Path(args.model)
    if src_path.exists() and src_path.is_file():
        if src_path.resolve() != dest.resolve():
            shutil.copy2(src_path, dest)
        _verify(dest, args.no_verify)
        print(f"copied weights to {dest}")
        return

    # Case 2: a known asset name — download it INTO models/ (not the cache).
    # Ultralytics writes a bare asset name to the current working directory, so
    # run the download from inside models/ and confirm the file landed there.
    name = Path(args.model).name
    prev_cwd = Path.cwd()
    try:
        os.chdir(MODELS)
        from ultralytics import YOLO

        model = YOLO(name)  # downloads ./<name> into models/ if not already present
        _ = model.names     # touch metadata to confirm the weights are usable
    except Exception as exc:
        print(f"failed to download {args.model}: {exc}")
        raise
    finally:
        os.chdir(prev_cwd)

    if not dest.exists():
        raise SystemExit(
            f"expected weights at {dest} after download, but the file is missing — "
            f"check that {args.model!r} is a valid Ultralytics asset name."
        )
    _verify(dest, args.no_verify)
    print(f"downloaded weights to {dest}")


if __name__ == "__main__":
    main()
