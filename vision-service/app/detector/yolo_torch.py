"""YOLOv8 detection + ByteTrack tracking via the Ultralytics PyTorch backend.

Runs on CPU (dev/fallback) or CUDA. The Jetson TensorRT backend
(:mod:`yolo_trt`) shares this class with a different weights file.

Two execution paths:

* default — one camera at a time through ``model.track`` (per-camera tracker
  swapped under a lock), as Ultralytics intends for a single stream.
* ``batch_cameras`` — concurrent per-camera calls are coalesced into a single
  ``model.predict`` over both frames (one GPU inference for the pair), then each
  camera's ByteTrack is driven manually. This removes the serialize-on-lock wait
  when a batch-capable engine is loaded; it degrades gracefully to per-frame
  inference if the engine is batch=1 or the cameras don't arrive together.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

from ..camera.base import Frame
from .base import Detector, RawTrack
from .classmap import label_for
from .tracker import VelocityTracker


class _Batcher:
    """Coalesce concurrent detect calls (one per camera thread) into one batched
    run. The first caller of a round becomes the leader: it waits up to
    ``wait_s`` for the other camera, then runs the batch for everyone and hands
    each caller its slice. Robust for two cameras — a caller that misses a round
    just retries, so it can never deadlock (worst case: a solo inference)."""

    def __init__(self, run_fn, max_batch: int, wait_s: float):
        self._run = run_fn
        self._max = max_batch
        self._wait = wait_s
        self._cv = threading.Condition()
        self._pending: dict = {}
        self._done: dict = {}
        self._gen = 0
        self._busy = False

    def submit(self, stream: str, item):
        with self._cv:
            while True:
                self._pending[stream] = item
                self._cv.notify_all()
                if not self._busy:
                    # Leader: gather peers (briefly), then run the whole batch.
                    self._busy = True
                    if len(self._pending) < self._max:
                        self._cv.wait_for(
                            lambda: len(self._pending) >= self._max, timeout=self._wait)
                    batch = dict(self._pending)
                    self._pending.clear()
                    gen = self._gen
                    err = None
                    try:
                        res = self._run(batch)          # holds lock; peers wait
                    except Exception as e:              # pragma: no cover
                        res, err = {}, e
                    self._done = {gen: res}
                    self._gen = gen + 1
                    self._busy = False
                    self._cv.notify_all()
                    if stream in res:
                        return res[stream]
                    if err is not None:
                        raise err
                    continue                            # missed own slice; retry
                # Follower: wait for the in-flight round, then take my slice.
                start = self._gen
                self._cv.wait_for(lambda: self._gen != start)
                r = self._done.get(start)
                if r and stream in r:
                    return r[stream]
                # Not included in that round — loop and try as a fresh leader.


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
                 backend_name: str = "torch-cpu", batch_cameras: bool = False,
                 batch_wait_ms: int = 20, batch_size: int = 2):
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
        # Batched path: one ByteTracker per stream driven manually.
        self._bt: Dict[str, object] = {}
        self._batcher = (_Batcher(self._run_batch, batch_size, batch_wait_ms / 1000.0)
                         if batch_cameras else None)

    def detect_and_track(
        self, frame: Frame, stream: str = "default", max_det: int | None = None
    ) -> List[RawTrack]:
        if self._batcher is not None:
            return self._batcher.submit(stream, (frame, max_det))
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

    def _run_batch(self, batch: dict) -> dict:
        """Run one batched inference over {stream: (frame, max_det)} and return
        {stream: [RawTrack]}. One GPU inference for all frames; each stream's
        ByteTrack is then updated manually (mirrors Ultralytics' tracker wiring).
        """
        import torch

        streams = list(batch.keys())
        imgs = [batch[s][0].image for s in streams]
        max_det = max((batch[s][1] or 0) for s in streams) or None
        with self._lock:  # single CUDA context; only one batched run at a time
            results = self._model.predict(
                imgs, conf=self._conf, imgsz=self._imgsz, device=self._device,
                max_det=max_det, verbose=False)
            out: dict = {}
            for s, result in zip(streams, results):
                tracker = self._bt.get(s) or self._new_trackers()[0]
                self._bt[s] = tracker
                det = result.boxes.cpu().numpy()
                tracks = tracker.update(det, result.orig_img)
                if len(tracks):
                    idx = tracks[:, -1].astype(int)
                    result = result[idx]
                    result.update(boxes=torch.as_tensor(tracks[:, :-1]))
                # else: leave raw boxes (no ids), matching Ultralytics behaviour
                vel = self._vels.setdefault(s, VelocityTracker())
                trk = self._parse([result], batch[s][0], vel)
                md = batch[s][1]
                if md is not None:
                    trk = sorted(trk, key=lambda t: t.confidence, reverse=True)[:md]
                out[s] = trk
            return out

    def _new_trackers(self) -> list:
        """Build a fresh ByteTracker list the way Ultralytics' on_predict_start
        does, so a new camera starts with its own tracker instead of inheriting
        whichever camera ran last."""
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import YAML, IterableSimpleNamespace
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
