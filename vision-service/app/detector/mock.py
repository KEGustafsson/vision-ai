"""Deterministic mock detector for CI and laptop dev.

Prefers the ground-truth boxes carried on synthetic frames (zero dependence on
any model). If a frame has no ground truth (e.g. a video file in mock mode), it
falls back to colour-blob detection so the pipeline still produces output.
Track ids are assigned by greedy nearest-centroid association across frames.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from ..camera.base import Frame
from .base import Detector, RawTrack
from .tracker import VelocityTracker

# HSV colour ranges matching synthetic actors (green=vessel, red=buoy, orange=person).
_COLOURS = [
    (8, "vessel", (40, 80, 80), (80, 255, 255)),
    (0, "person", (10, 120, 120), (25, 255, 255)),
    (80, "buoy", (0, 120, 120), (10, 255, 255)),
]


class MockDetector(Detector):
    backend = "mock"

    def __init__(self, assoc_dist: float = 80.0, reid_opts: dict | None = None):
        self._assoc_dist = assoc_dist
        self._reid_opts = reid_opts or {}
        # Per-camera association/velocity state (one shared detector, many cams).
        self._state: dict = {}  # stream -> {tracker, next_id, prev}

    def _stream_state(self, stream: str) -> dict:
        return self._state.setdefault(
            stream,
            {"tracker": VelocityTracker(**self._reid_opts), "next_id": 1, "prev": {}},
        )

    def detect_and_track(
        self, frame: Frame, stream: str = "default", max_det: int | None = None
    ) -> List[RawTrack]:
        dets = self._from_truth(frame) if frame.truth else self._from_colour(frame.image)
        if max_det is not None:
            dets = sorted(dets, key=lambda d: d[2], reverse=True)[:max_det]
        return self._associate(frame.seq, dets, self._stream_state(stream))

    def _from_truth(self, frame: Frame):
        out = []
        for t in frame.truth:
            out.append((t.cls, t.label, t.confidence, t.x, t.y, t.w, t.h))
        return out

    def _from_colour(self, img: np.ndarray):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        out = []
        for cls, label, lo, hi in _COLOURS:
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) < 100:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                out.append((cls, label, 0.85, float(x), float(y), float(w), float(h)))
        return out

    def _associate(self, seq: int, dets, state: dict) -> List[RawTrack]:
        tracks: List[RawTrack] = []
        used = set()
        active = set()
        new_prev = {}
        prev = state["prev"]
        for cls, label, conf, x, y, w, h in dets:
            cx, cy = x + w / 2, y + h / 2
            # Greedy nearest previous centroid.
            best_id, best_d = None, self._assoc_dist
            for tid, (pcx, pcy) in prev.items():
                if tid in used:
                    continue
                d = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
                if d < best_d:
                    best_id, best_d = tid, d
            if best_id is None:
                best_id = state["next_id"]
                state["next_id"] += 1
            used.add(best_id)
            new_prev[best_id] = (cx, cy)
            # Waterline re-id: a partial re-detection of a known target keeps its
            # canonical id (and display id). Velocity is anchored at the bbox
            # bottom-center — the waterline — which stays put when the detected
            # extent flips partial <-> full (see VelocityTracker.update).
            canon = state["tracker"].resolve(best_id, seq, x, y, w, h, label)
            active.add(canon)
            vx, vy, age = state["tracker"].update(canon, seq, cx, y + h)
            disp = state["tracker"].display_id(canon)
            stable = state["tracker"].stable_id(canon)
            tracks.append(RawTrack(track_id=disp, stable_id=stable, cls=cls,
                                   label=label, confidence=conf, x=x, y=y,
                                   w=w, h=h, vx=vx, vy=vy, age_frames=age))
        state["prev"] = new_prev
        state["tracker"].prune(active, seq)
        return tracks
