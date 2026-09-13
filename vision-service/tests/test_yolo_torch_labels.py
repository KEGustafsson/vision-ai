"""The torch/tensorrt parse path must decode raw class ids through the ACTIVE
model's class map (like the DeepStream pipeline), not assume COCO: raw ids
collide across models (forward-watch 0 = ship, COCO 0 = person), so decoding
with the wrong table would e.g. turn every ship into a phantom "person" —
a man-overboard candidate."""

from __future__ import annotations

import numpy as np

from app.camera.base import Frame
from app.detector.tracker import VelocityTracker
from app.detector.yolo_torch import YoloTorchDetector


class _FakeBoxes:
    """Shaped like ultralytics' Boxes: one array per attribute covering every
    detection (that is what lets the parser pull the result off the GPU in four
    transfers instead of four per box)."""

    def __init__(self, rows):
        self.xyxy = np.array([r["xyxy"] for r in rows], dtype=np.float32)
        self.conf = np.array([r["conf"] for r in rows], dtype=np.float32)
        self.cls = np.array([r["cls"] for r in rows], dtype=np.float32)
        ids = [r.get("tid") for r in rows]
        self.id = None if any(i is None for i in ids) else np.array(ids, dtype=np.float32)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def _detector(model_name: str) -> YoloTorchDetector:
    # Bypass __init__ (which loads ultralytics weights); _parse only needs the
    # model name.
    det = YoloTorchDetector.__new__(YoloTorchDetector)
    det._model_name = model_name
    return det


def _parse_one(model_name: str, raw_cls: int):
    det = _detector(model_name)
    frame = Frame(image=np.zeros((4, 4, 3), dtype=np.uint8), seq=1)
    results = [_FakeResult(_FakeBoxes(
        [{"cls": raw_cls, "conf": 0.9, "xyxy": [10.0, 10.0, 50.0, 40.0], "tid": 1}]))]
    tracks = det._parse(results, frame, VelocityTracker())
    assert len(tracks) == 1
    return tracks[0]

def test_coco_ids_pass_through_unchanged():
    t = _parse_one("coco", 0)
    assert (t.label, t.cls) == ("person", 0)
    t = _parse_one("coco", 8)
    assert (t.label, t.cls) == ("vessel", 8)


def test_forward_watch_ids_use_model_table_not_coco():
    # forward-watch raw 0 = ship: must become "vessel" (synthetic id 81),
    # NOT COCO's "person".
    t = _parse_one("forward-watch", 0)
    assert (t.label, t.cls) == ("vessel", 81)


def test_marine_surveillance_ids_use_model_table():
    t = _parse_one("marine-surveillance", 6)
    assert (t.label, t.cls) == ("warship", 93)


def test_parse_reads_every_box_from_one_array_per_attribute():
    """Multi-box results must decode positionally — the parser reads column i of
    each array, so a mismatch here would silently pair a label with another
    box's geometry."""
    det = _detector("coco")
    frame = Frame(image=np.zeros((4, 4, 3), dtype=np.uint8), seq=1)
    boxes = _FakeBoxes([
        {"cls": 0, "conf": 0.9, "xyxy": [10.0, 10.0, 20.0, 30.0], "tid": 1},
        {"cls": 8, "conf": 0.5, "xyxy": [50.0, 60.0, 90.0, 80.0], "tid": 2},
    ])
    tracks = det._parse([_FakeResult(boxes)], frame, VelocityTracker())
    assert [(t.label, t.cls) for t in tracks] == [("person", 0), ("vessel", 8)]
    assert [(t.x, t.y, t.w, t.h) for t in tracks] == [
        (10.0, 10.0, 10.0, 20.0), (50.0, 60.0, 40.0, 20.0)]
    assert [round(t.confidence, 2) for t in tracks] == [0.9, 0.5]


def test_parse_handles_untracked_boxes():
    # No tracker ids yet (first frames / tracking disabled): no track_id, and
    # certainly no crash reading a missing id array.
    det = _detector("coco")
    frame = Frame(image=np.zeros((4, 4, 3), dtype=np.uint8), seq=1)
    boxes = _FakeBoxes([{"cls": 0, "conf": 0.9, "xyxy": [1.0, 2.0, 3.0, 4.0]}])
    tracks = det._parse([_FakeResult(boxes)], frame, VelocityTracker())
    assert len(tracks) == 1
    assert tracks[0].track_id is None and tracks[0].stable_id is None
