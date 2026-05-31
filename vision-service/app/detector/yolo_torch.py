"""YOLOv8 detection + ByteTrack tracking via the Ultralytics PyTorch backend.

Runs on CPU (dev/fallback) or CUDA. The Jetson TensorRT backend
(:mod:`yolo_trt`) shares this class with a different weights file.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

from ..camera.base import Frame
from .base import Detector, RawTrack
from .classmap import label_for
from .tracker import VelocityTracker


class YoloTorchDetector(Detector):
    """One YOLO model shared across all cameras.

    A single model means one load and one CUDA/TensorRT context — critical on
    the 8 GB Orin Nano, where two concurrent model inits race the CUDA allocator
    and one camera dies (NvMap ENOMEM / CUDACachingAllocator assert). Inference
    is serialised by ``_lock``; Ultralytics keeps its ByteTracker on
    ``model.predictor.trackers``, so we swap in the calling camera's own tracker
    under that lock to keep track IDs and the motion model isolated per camera.
    """

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
        self._lock = threading.Lock()
        # Per-camera state. _trackers holds each camera's Ultralytics ByteTracker
        # list (swapped into the predictor before each call); _vels holds each
        # camera's pixel-velocity history.
        self._trackers: Dict[str, list] = {}
        self._vels: Dict[str, VelocityTracker] = {}

    def detect_and_track(
        self, frame: Frame, stream: str = "default", max_det: int | None = None
    ) -> List[RawTrack]:
        with self._lock:
            # Always call with persist=True. Ultralytics only (re)registers its
            # tracker callbacks when the predictor has no `trackers` attribute,
            # and a persist=False call leaves a callback that recreates the
            # tracker every frame (resetting IDs). So we keep `trackers` set at
            # all times: restore the calling camera's saved tracker, or hand a
            # newly-built one to a camera we haven't seen yet (never delete the
            # attr — that would re-register callbacks and double-update tracks).
            pred = getattr(self._model, "predictor", None)
            if pred is not None:
                pred.trackers = self._trackers.get(stream) or self._new_trackers()
            results = self._model.track(
                frame.image, persist=True, tracker=self._tracker_cfg,
                conf=self._conf, imgsz=self._imgsz, device=self._device,
                max_det=max_det, verbose=False,
            )
            self._trackers[stream] = self._model.predictor.trackers
            vel = self._vels.setdefault(stream, VelocityTracker())
            tracks = self._parse(results, frame, vel)
            if max_det is not None:
                tracks = sorted(tracks, key=lambda tr: tr.confidence, reverse=True)[:max_det]
            return tracks

    def _new_trackers(self) -> list:
        """Build a fresh ByteTracker list the way Ultralytics' on_predict_start
        does, so a new camera starts with its own tracker instead of inheriting
        whichever camera ran last."""
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import IterableSimpleNamespace, YAML
        from ultralytics.utils.checks import check_yaml

        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(self._tracker_cfg)))
        return [TRACKER_MAP[cfg.tracker_type](args=cfg)]

    def _parse(self, results, frame: Frame, vel: VelocityTracker) -> List[RawTrack]:
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
                vx, vy, age = vel.update(tid, frame.seq, cx, cy)
                active.add(tid)
            out.append(RawTrack(track_id=tid, cls=cls, label=label, confidence=conf,
                                x=x1, y=y1, w=w, h=h, vx=vx, vy=vy, age_frames=age))
        vel.prune(active, frame.seq)
        return out
