"""Download the forward-watch marine obstacle ONNX into deepstream/ AND convert
it to the DeepStream-Yolo layout the custom parser requires.

The model (YOLOv8n, 6 classes: ship/boat/debris/buoy/kayak/log) is published at
https://github.com/SkipperDon/signalk-forward-watch and is NOT vendored in this
repo. Run this once to pre-bake the image / provision an offline boat when you
select ``detector.model: forward-watch`` in config/deepstream.yaml.

IMPORTANT: the published ONNX is a STOCK Ultralytics export (output
``[1, 4+nc, 8400]``). The ``NvDsInferParseYolo`` custom parser this project builds
cannot read that layout — fed as-is it yields ZERO detections. So after
downloading we run scripts/convert_to_deepstream.py to rewrite the head into the
parser's ``[1, 8400, 6]`` ``[x1,y1,x2,y2,score,class]`` layout (the same thing
DeepStream-Yolo's export_yoloV8.py bakes in). Conversion needs the ``onnx``
package (and ``onnxruntime`` for the optional cross-check); if it is missing the
raw file is kept and the exact convert command is printed.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
DEEPSTREAM_DIR = SCRIPTS_DIR.parent / "deepstream"
DEFAULT_URL = (
    "https://raw.githubusercontent.com/SkipperDon/signalk-forward-watch/"
    "main/models/forward-watch.onnx"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL, help="source ONNX URL")
    ap.add_argument(
        "--out",
        default=str(DEEPSTREAM_DIR / "forward-watch.onnx"),
        help="destination path (must match onnx-file in pgie_forward_watch.txt)",
    )
    ap.add_argument("--no-convert", action="store_true",
                    help="download only; do NOT rewrite to the parser layout")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {args.url}\n  -> {out}")
    urllib.request.urlretrieve(args.url, out)
    size = out.stat().st_size
    if size < 1_000_000:
        raise SystemExit(
            f"downloaded file is only {size} bytes — expected ~12 MB. "
            "Check the URL (it may be an HTML error page or a Git LFS pointer)."
        )
    print(f"saved {size / 1e6:.1f} MB to {out}")

    convert_cmd = f"python3 {SCRIPTS_DIR / 'convert_to_deepstream.py'} {out} --inplace"
    if args.no_convert:
        print(f"\n--no-convert set: the file is STOCK layout and will produce ZERO\n"
              f"detections until converted. Run:\n  {convert_cmd}")
        return

    # Convert in place to the parser-compatible layout.
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import convert_to_deepstream as conv
    except ImportError as e:
        raise SystemExit(
            f"\nDownloaded OK, but conversion needs the 'onnx' package ({e}).\n"
            f"Install it (pip install onnx onnxruntime) and run:\n  {convert_cmd}\n"
            "The raw file as-is will produce ZERO detections."
        )

    tmp = out.with_suffix(".onnx.tmp")
    print("\nconverting to DeepStream-Yolo layout (parser needs [N,6])")
    n, nc = conv.convert(out, tmp, force=False)
    print(f"  rewrote head: {n} anchors, {nc} classes -> output [batch, {n}, 6]")
    try:
        conv.validate(out, tmp)
    except ImportError:
        print("  (onnxruntime not installed — skipped numeric cross-check)")
    tmp.replace(out)
    print(f"converted in place: {out}")
    print("  reminder: delete any cached *_gpu0_fp16.engine so nvinfer rebuilds it.")


if __name__ == "__main__":
    main()
