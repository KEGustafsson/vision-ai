"""Download YOLOv8 weights to ``vision-service/models/<model>`` deterministically.

Ultralytics auto-downloads on first use, but it writes into its own private cache
(or the current working directory), NOT the path the rest of the system expects.
The DeepStream image build (``COPY models/yolov8n.pt`` in the export stage), the
TensorRT engine export (``export_engine.py --weights models/yolov8n.pt``) and the
docs all key off ``models/yolov8n.pt`` — so this writes the weights to exactly
that path rather than leaving them in a cache the next step can't find.

Usage::

    python3 scripts/download_models.py                 # -> models/yolov8n.pt
    python3 scripts/download_models.py --model yolov8s.pt
    python3 scripts/download_models.py --model /path/to/local.pt   # copies it in
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default="yolov8n.pt",
        help="A known Ultralytics asset name (e.g. yolov8n.pt) to download, or a "
             "path to an existing .pt file to copy into models/.",
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
    print(f"downloaded weights to {dest}")


if __name__ == "__main__":
    main()
