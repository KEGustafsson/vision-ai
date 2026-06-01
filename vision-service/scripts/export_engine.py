"""Export a YOLOv8 .pt model to a TensorRT .engine (run ON the Jetson).

    python scripts/export_engine.py --weights models/yolov8n.pt --imgsz 640

Produces models/yolov8n.engine (FP16). INT8 needs a calibration dataset; see
docs/jetson-setup.md.
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    # FP16 is the default on Jetson; pass --no-half to export an FP32 engine.
    ap.add_argument("--no-half", dest="half", action="store_false", default=True,
                    help="export FP32 instead of FP16")
    ap.add_argument("--workspace", type=int, default=3)
    # Max batch for the engine. >1 (with --dynamic) lets both cameras run in one
    # inference (see DetectorConfig.batch_cameras). Default 1 = single-stream.
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dynamic", action="store_true",
                    help="dynamic batch 1..batch (needed to also run a single camera)")
    ap.add_argument("--out", default=None,
                    help="rename the exported .engine to this path")
    args = ap.parse_args()

    import shutil

    from ultralytics import YOLO

    model = YOLO(args.weights)
    try:
        path = model.export(format="engine", half=args.half, imgsz=args.imgsz,
                            workspace=args.workspace, device=0,
                            batch=args.batch, dynamic=args.dynamic)
    except Exception as exc:
        print(f"TensorRT export failed: {exc}")
        print("Ensure this runs on the Jetson with TensorRT installed and CUDA available.")
        raise
    if args.out and str(path) != args.out:
        shutil.move(str(path), args.out)
        path = args.out
    print(f"exported engine: {path}")


if __name__ == "__main__":
    main()
