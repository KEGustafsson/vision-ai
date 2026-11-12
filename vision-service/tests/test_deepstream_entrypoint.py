"""The DeepStream images' entrypoint (scripts/deepstream_entrypoint.sh), run for
real against a scratch tree.

The release compose files keep deepstream/ in a named volume, which Docker
fills from the image only while it is empty. After an image update the volume
would still hold the previous image's ONNX, pgie configs and labels, and
nvinfer would quietly rebuild an engine from the old ONNX instead of failing.
With VISION_DS_SYNC_FROM_IMAGE=1 the entrypoint refreshes the volume from the
image whenever the image's build identity changes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "deepstream_entrypoint.sh"

pytestmark = pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX sh")


def _tree(tmp_path: Path, image_id: str = "image-2") -> dict:
    image = tmp_path / "deepstream-image"
    image.mkdir()
    (image / "pgie_yolo11n.txt").write_text("new pgie\n")
    (image / "labels_coco.txt").write_text("new labels\n")
    (image / "yolo11n_ds.onnx").write_text("new onnx\n")
    (image / ".image-id").write_text(image_id + "\n")
    seed = tmp_path / "deepstream-default"
    seed.mkdir()
    (seed / "yolo11n_ds.onnx").write_text("seed onnx\n")
    vol = tmp_path / "deepstream"
    vol.mkdir()
    return {"image": image, "seed": seed / "yolo11n_ds.onnx", "vol": vol}


def _run(t: dict, sync: bool) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(
        VISION_DS_DIR=str(t["vol"]),
        VISION_DS_IMAGE_DIR=str(t["image"]),
        VISION_DS_SEED=str(t["seed"]),
    )
    if sync:
        env["VISION_DS_SYNC_FROM_IMAGE"] = "1"
    else:
        env.pop("VISION_DS_SYNC_FROM_IMAGE", None)
    return subprocess.run(
        ["sh", str(SCRIPT), "sh", "-c", "echo started"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _old_volume(vol: Path, image_id: str = "image-1") -> None:
    (vol / "pgie_yolo11n.txt").write_text("old pgie\n")
    (vol / "labels_coco.txt").write_text("old labels\n")
    (vol / "yolo11n_ds.onnx").write_text("old onnx\n")
    (vol / "yolo11n_ds.onnx_b2_gpu0_fp16.engine").write_text("old engine\n")
    (vol / "forward-watch.onnx").write_text("user's own model\n")
    (vol / ".image-id").write_text(image_id + "\n")


def test_refreshes_a_volume_left_by_another_image(tmp_path):
    t = _tree(tmp_path)
    _old_volume(t["vol"])
    r = _run(t, sync=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("started")
    vol = t["vol"]
    assert (vol / "pgie_yolo11n.txt").read_text() == "new pgie\n"
    assert (vol / "labels_coco.txt").read_text() == "new labels\n"
    assert (vol / "yolo11n_ds.onnx").read_text() == "new onnx\n"
    assert not list(vol.glob("*.engine")), "an engine built from the old ONNX must not survive"
    assert (vol / "forward-watch.onnx").read_text() == "user's own model\n"
    assert (vol / ".image-id").read_text().strip() == "image-2"


def test_refreshes_a_volume_with_no_identity(tmp_path):
    # A volume filled before images carried an identity counts as another image's.
    t = _tree(tmp_path)
    _old_volume(t["vol"])
    (t["vol"] / ".image-id").unlink()
    assert _run(t, sync=True).returncode == 0
    assert (t["vol"] / "pgie_yolo11n.txt").read_text() == "new pgie\n"
    assert not list(t["vol"].glob("*.engine"))


def test_keeps_the_engine_when_the_image_is_unchanged(tmp_path):
    t = _tree(tmp_path, image_id="image-1")
    _old_volume(t["vol"], image_id="image-1")
    r = _run(t, sync=True)
    assert r.returncode == 0, r.stderr
    assert "refreshed" not in r.stdout
    assert (t["vol"] / "yolo11n_ds.onnx_b2_gpu0_fp16.engine").exists()
    assert (t["vol"] / "pgie_yolo11n.txt").read_text() == "old pgie\n"


def test_never_touches_a_source_checkout_without_the_opt_in(tmp_path):
    # Source builds bind-mount the git checkout's deepstream/: its files are
    # the checkout's, so nothing is refreshed or removed.
    t = _tree(tmp_path)
    _old_volume(t["vol"])
    assert _run(t, sync=False).returncode == 0
    assert (t["vol"] / "pgie_yolo11n.txt").read_text() == "old pgie\n"
    assert (t["vol"] / "yolo11n_ds.onnx_b2_gpu0_fp16.engine").exists()


def test_still_seeds_a_missing_onnx(tmp_path):
    t = _tree(tmp_path)
    r = _run(t, sync=False)
    assert r.returncode == 0, r.stderr
    assert (t["vol"] / "yolo11n_ds.onnx").read_text() == "seed onnx\n"
