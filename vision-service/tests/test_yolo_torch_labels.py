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


class _Coords:
    def __init__(self, vals):
        self._vals = vals

    def tolist(self):
        return self._vals


class _FakeBox:
    def __init__(self, cls, conf, xyxy, tid=None):
        self.cls = [cls]
        self.conf = [conf]
        self.xyxy = [_Coords(xyxy)]
        self.id = [tid] if tid is not None else None


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
    results = [_FakeResult([_FakeBox(raw_cls, 0.9, [10.0, 10.0, 50.0, 40.0], tid=1)])]
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
