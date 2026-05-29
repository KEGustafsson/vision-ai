"""Download YOLOv8 weights into models/. Ultralytics auto-downloads on first
use, so this is a convenience for pre-baking images / offline boats."""

from __future__ import annotations

import argparse
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    args = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    from ultralytics import YOLO

    YOLO(args.model)  # triggers download into the ultralytics cache
    print(f"ensured weights available: {args.model}")


if __name__ == "__main__":
    main()
