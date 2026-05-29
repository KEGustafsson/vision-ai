"""YOLOv8 detection + ByteTrack tracking via the Ultralytics PyTorch backend.

Runs on CPU (dev/fallback) or CUDA. The Jetson TensorRT backend
(:mod:`yolo_trt`) shares this class with a different weights file.
"""

from __future__ import annotations

from typing import List, Optional

from ..camera.base import Frame
from .base import Detector, RawTrack
from .classmap import label_for
from .tracker import VelocityTracker


class YoloTorchDetector(Detector):
    def __init__(self, weights: str, device: str = "cpu", confidence: float = 0.35,
                 imgsz: int = 640, tracker_cfg: str = "bytetrack.yaml",
                 backend_name: str = "torch-cpu"):
        from ultralytics import YOLO  # imported lazily so mock mode needs no torch

        self.backend = backend_name
        self._model = YOLO(weights)
        self._device = device
        self._conf = confidence
        self._imgsz = imgsz
        self._tracker_cfg = tracker_cfg
        self._vel = VelocityTracker()

    def detect_and_track(self, frame: Frame) -> List[RawTrack]:
        results = self._model.track(
            frame.image, persist=True, tracker=self._tracker_cfg,
            conf=self._conf, imgsz=self._imgsz, device=self._device, verbose=False,
        )
        out: List[RawTrack] = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None:
            return out
        active = set()
        for b in boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1
            tid: Optional[int] = int(b.id[0]) if b.id is not None else None
            label = label_for(cls)
            vx = vy = 0.0
            age = 0
            if tid is not None:
                cx, cy = x1 + w / 2, y1 + h / 2
                vx, vy, age = self._vel.update(tid, frame.seq, cx, cy)
                active.add(tid)
            out.append(RawTrack(track_id=tid, cls=cls, label=label, confidence=conf,
                                x=x1, y=y1, w=w, h=h, vx=vx, vy=vy, age_frames=age))
        self._vel.prune(active, frame.seq)
        return out
