"""Download the forward-watch marine obstacle ONNX into deepstream/.

The model (YOLOv8n, 6 classes: ship/boat/debris/buoy/kayak/log) is published at
https://github.com/SkipperDon/signalk-forward-watch and is NOT vendored in this
repo. Run this once to pre-bake the image / provision an offline boat when you
select ``detector.model: forward-watch`` in config/deepstream.yaml.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

DEEPSTREAM_DIR = Path(__file__).resolve().parent.parent / "deepstream"
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


if __name__ == "__main__":
    main()
