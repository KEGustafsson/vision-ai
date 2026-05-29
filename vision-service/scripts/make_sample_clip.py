"""Render a short synthetic marine clip to vision-service/samples/clip.mp4 so
mock mode can be exercised with the video-file source path.

    python scripts/make_sample_clip.py --seconds 20 --fps 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.camera.synthetic import SyntheticSource  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "samples" / "clip.mp4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    src = SyntheticSource("forward", with_mob=True, fps=args.fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT), fourcc, args.fps, (src.width, src.height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter for {OUT} (codec unavailable?)")
    try:
        for _ in range(args.seconds * args.fps):
            writer.write(src.read().image)
    finally:
        writer.release()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
