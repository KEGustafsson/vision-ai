"""Train YOLOv8s on the Roboflow "Marine Surveillance" dataset and export a
parser-ready ONNX for the DeepStream pipeline (detector.model=marine-surveillance).

WHY ON-BOX TRAINING
===================
Roboflow does NOT let you download another user's trained weights — only the
dataset. So we download the dataset, train YOLOv8s here, and export. This needs
torch+CUDA+ultralytics, which the runtime DeepStream image does not have. Run it
inside an Ultralytics Jetson container (it bundles the right aarch64 torch):

  docker run --rm -it --runtime nvidia --network host \
    -v /home/kgustafs/docker/apps/vision-ai:/work -w /work \
    ultralytics/ultralytics:latest-jetson-jetpack6 \
    bash -lc "pip install -q roboflow onnx onnxslim onnxruntime && \
      python3 training/train_marine_surveillance.py \
        --api-key \$ROBOFLOW_KEY --workspace WS --project PROJ --version N"

⚠ The Orin is RAM-tight. Stop the GPU co-tenants first (vision-service +
gstreamer_in/out_overlay) so training does not OOM or stall the live pipeline,
and use a modest --batch (default 8). Restart them when done.

PIPELINE
========
download dataset → assert class order matches classmap → train YOLOv8s →
stock ONNX export → vision-service/scripts/convert_to_deepstream.py (the proven
[N,6] rewrite, same one used for forward-watch) →
vision-service/deepstream/marine-surveillance.onnx, and regenerate
vision-service/deepstream/labels_marine_surveillance.txt from data.yaml.

After it finishes: rebuild the image so COPY deepstream bakes the new ONNX, set
detector.model=marine-surveillance, delete any stale
marine-surveillance.onnx_b*_gpu0_fp16.engine, and recreate the container.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
REPO_DIR = TRAINING_DIR.parent
VISION_DIR = REPO_DIR / "vision-service"
SCRIPTS_DIR = VISION_DIR / "scripts"  # convert_to_deepstream.py lives here (shared)
DEEPSTREAM_DIR = VISION_DIR / "deepstream"

# Must match _MS_LABEL order in app/detector/classmap.py. Roboflow orders classes
# alphabetically; if the dataset's data.yaml differs, we STOP rather than silently
# train a model whose class ids don't line up with the class map.
EXPECTED_NAMES = ["boat", "buoy", "kayak", "sail-boat", "speed-boat", "vessel", "war-ship"]


def _norm(s: str) -> str:
    return s.strip().lower().replace("_", "-").replace(" ", "-")


def _check_class_order(data_yaml: Path) -> list[str]:
    import yaml
    with open(data_yaml) as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):  # {0: 'boat', ...}
        names = [names[k] for k in sorted(names)]
    got = [_norm(n) for n in names]
    want = [_norm(n) for n in EXPECTED_NAMES]
    if got != want:
        raise SystemExit(
            "Dataset class order does NOT match the class map.\n"
            f"  data.yaml names : {names}\n"
            f"  expected (classmap _MS_LABEL): {EXPECTED_NAMES}\n"
            "Either you have a different dataset/version, or Roboflow reordered "
            "the classes. Update _MS_LABEL/_MS_CLS_REMAP in app/detector/classmap.py "
            "(and labels_marine_surveillance.txt) to this exact order, then re-run."
        )
    return names


def _resolve_local_dataset(src: Path, workdir: Path) -> Path:
    """Accept a .zip export, a dataset dir, or a data.yaml; return a data.yaml with
    ABSOLUTE train/val/test paths (Roboflow ships ``../train/images`` which only
    resolves from a specific cwd — rewrite it so ultralytics always finds the data)."""
    import zipfile
    import yaml

    if src.is_file() and src.suffix == ".zip":
        dest = workdir / "dataset"
        dest.mkdir(parents=True, exist_ok=True)
        print(f"extracting {src.name} → {dest}")
        with zipfile.ZipFile(src) as z:
            z.extractall(dest)
        root = dest
    elif src.is_dir():
        root = src
    elif src.is_file() and src.name.endswith(".yaml"):
        root = src.parent
    else:
        raise SystemExit(f"--data {src} is not a .zip, a dir, or a data.yaml")

    dy = root / "data.yaml"
    if not dy.is_file():
        cands = list(root.rglob("data.yaml"))
        if not cands:
            raise SystemExit(f"no data.yaml found under {root}")
        dy = cands[0]
    root = dy.parent

    with open(dy) as f:
        cfg = yaml.safe_load(f)
    for split, sub in (("train", "train/images"), ("val", "valid/images"), ("test", "test/images")):
        p = root / sub
        if p.is_dir():
            cfg[split] = str(p.resolve())
    cfg.pop("path", None)
    norm = root / "data.normalized.yaml"
    with open(norm, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"  normalized data.yaml → {norm}")
    return norm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path,
                    help="local YOLOv8 dataset (a .zip export, a dataset dir, or a "
                         "data.yaml). If given, skips the Roboflow download — no API key needed.")
    ap.add_argument("--api-key", help="Roboflow API key (only if --data is not given)")
    ap.add_argument("--workspace", help="Roboflow workspace slug")
    ap.add_argument("--project", help="Roboflow project slug")
    ap.add_argument("--version", type=int, help="dataset version number")
    ap.add_argument("--base", default="yolov8s.pt", help="base weights to fine-tune")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8, help="keep small on the Orin")
    ap.add_argument("--workers", type=int, default=8,
                    help="dataloader workers; LOWER on the 8GB Orin (unified mem) to "
                         "leave room for the model — 2 is a good start")
    ap.add_argument("--cache", default="False",
                    help="ultralytics image cache: False / ram / disk (keep False on Orin)")
    ap.add_argument("--device", default="0", help="'0' for GPU, 'cpu' to force CPU")
    ap.add_argument("--out", type=Path, default=DEEPSTREAM_DIR / "marine-surveillance.onnx")
    ap.add_argument("--workdir", type=Path, default=TRAINING_DIR / "_train_marine_surveillance")
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse an existing best.pt in --workdir (export/convert only)")
    args = ap.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)

    # 1. dataset ------------------------------------------------------------
    if args.data:
        data_yaml = _resolve_local_dataset(args.data, args.workdir)
    else:
        if not all([args.api_key, args.workspace, args.project, args.version]):
            raise SystemExit("provide --data, OR all of --api-key/--workspace/--project/--version")
        from roboflow import Roboflow
        rf = Roboflow(api_key=args.api_key)
        proj = rf.workspace(args.workspace).project(args.project)
        print(f"downloading {args.workspace}/{args.project} v{args.version} (yolov8)")
        ds = proj.version(args.version).download(
            "yolov8", location=str(args.workdir / "dataset"))
        data_yaml = Path(ds.location) / "data.yaml"
    names = _check_class_order(data_yaml)
    print(f"  class order OK: {names}")

    # 2. train --------------------------------------------------------------
    from ultralytics import YOLO
    runs = args.workdir / "runs"
    best = runs / "ms" / "weights" / "best.pt"
    if args.skip_train and best.is_file():
        print(f"--skip-train: reusing {best}")
    else:
        print(f"training {args.base} on {data_yaml} "
              f"({args.epochs} epochs, imgsz {args.imgsz}, batch {args.batch}, device {args.device})")
        cache = {"false": False, "true": True}.get(str(args.cache).lower(), args.cache)
        YOLO(args.base).train(
            data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, device=args.device, workers=args.workers, cache=cache,
            project=str(runs), name="ms", exist_ok=True)
    if not best.is_file():
        raise SystemExit(f"training did not produce {best}")

    # 3. stock ONNX export (dynamic batch so the engine can build at batch=2) -
    print("exporting stock ONNX (dynamic batch)")
    stock = Path(YOLO(str(best)).export(
        format="onnx", opset=12, dynamic=True, simplify=True, imgsz=args.imgsz))
    print(f"  stock onnx: {stock}")

    # 4. convert to the parser's [N,6] layout (proven converter) ------------
    sys.path.insert(0, str(SCRIPTS_DIR))
    import convert_to_deepstream as conv
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n, nc = conv.convert(stock, args.out, force=False, imgsz=args.imgsz)
    print(f"  converted → {args.out}: {n} anchors, {nc} classes → [batch, {n}, 6]")
    if nc != len(EXPECTED_NAMES):
        raise SystemExit(f"exported model has {nc} classes, expected {len(EXPECTED_NAMES)}")
    try:
        conv.validate(stock, args.out)
    except ImportError:
        print("  (onnxruntime missing — skipped numeric cross-check)")

    # 5. regenerate the label file from the dataset's exact names -----------
    labels = DEEPSTREAM_DIR / "labels_marine_surveillance.txt"
    labels.write_text("\n".join(names) + "\n")
    print(f"  wrote {labels}")

    print("\nDONE. Next:")
    print("  1) rebuild the image so COPY deepstream bakes marine-surveillance.onnx")
    print("  2) set detector.model: marine-surveillance in config/deepstream.yaml")
    print("  3) delete any stale marine-surveillance.onnx_b*_gpu0_fp16.engine")
    print("  4) recreate vision-service (overlay-stop dance to avoid NVMM OOM)")


if __name__ == "__main__":
    main()
