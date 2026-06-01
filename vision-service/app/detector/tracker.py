"""Per-track centroid history -> pixel velocity (px/frame), shared by all
backends. Backends provide stable track ids; this fills in velocity and age."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Tuple


class VelocityTracker:
    def __init__(self, history: int = 5):
        self._hist: Dict[int, Deque[Tuple[int, float, float]]] = {}
        self._first_seq: Dict[int, int] = {}
        self._history = history

    def update(self, track_id: int, seq: int, cx: float, cy: float) -> Tuple[float, float, int]:
        """Return (vx, vy, age_frames) for a track centroid at the given seq."""
        if track_id not in self._hist:
            self._hist[track_id] = deque(maxlen=self._history)
            self._first_seq[track_id] = seq
        hist = self._hist[track_id]
        vx = vy = 0.0
        if hist:
            # Average velocity across the whole history window (oldest -> now),
            # not just the last step: a single jittery frame no longer produces a
            # large spurious velocity that would fling a coasted box off-target.
            oseq, ocx, ocy = hist[0]
            dseq = max(seq - oseq, 1)
            vx = (cx - ocx) / dseq
            vy = (cy - ocy) / dseq
        hist.append((seq, cx, cy))
        age = seq - self._first_seq[track_id]
        return vx, vy, age

    def prune(self, active_ids: set, seq: int, max_idle: int = 60) -> None:
        """Drop tracks not seen recently to bound memory."""
        for tid in list(self._hist.keys()):
            if tid in active_ids:
                continue
            last_seq = self._hist[tid][-1][0] if self._hist[tid] else 0
            if seq - last_seq > max_idle:
                self._hist.pop(tid, None)
                self._first_seq.pop(tid, None)
