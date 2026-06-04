"""Per-track centroid history -> pixel velocity (px/frame), shared by all
backends. Backends provide stable track ids; this fills in velocity and age."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Tuple


class VelocityTracker:
    def __init__(self, history: int = 5, id_min: int = 10, id_max: int = 99):
        self._hist: Dict[int, Deque[Tuple[int, float, float]]] = {}
        self._first_seq: Dict[int, int] = {}
        self._history = history
        # Compact, recycled display ids in [id_min, id_max]. The backend trackers
        # (ByteTrack/NvDCF) hand out ever-growing raw ids; we map each raw id to a
        # small 2-digit number and return it to the pool when the track is pruned,
        # so detections always carry a stable, human-readable id in a bounded
        # range. Velocity/age state stays keyed by the raw id internally.
        self._id_min = id_min
        self._id_max = id_max
        self._display: Dict[int, int] = {}
        self._free: list[int] = list(range(id_min, id_max + 1))

    def _alloc_display(self, track_id: int) -> int:
        if self._free:
            return self._free.pop(0)
        # Pool exhausted (more live tracks than the range can hold): fall back to
        # a wrapped value. May collide, but maxTargetsPerStream keeps this rare.
        span = self._id_max - self._id_min + 1
        return self._id_min + (track_id % span)

    def display_id(self, track_id: int) -> int:
        """Bounded 10-99 id for a raw track id (raw id itself if unseen)."""
        return self._display.get(track_id, track_id)

    def update(self, track_id: int, seq: int, cx: float, cy: float) -> Tuple[float, float, int]:
        """Return (vx, vy, age_frames) for a track centroid at the given seq."""
        if track_id not in self._hist:
            self._hist[track_id] = deque(maxlen=self._history)
            self._first_seq[track_id] = seq
            self._display[track_id] = self._alloc_display(track_id)
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
                # Recycle the display id by appending to the *back* of the pool,
                # so freed numbers rotate to the end and are reused last — the
                # allocator keeps cycling through all of 10..99 before handing a
                # just-freed number to a new track.
                disp = self._display.pop(tid, None)
                if disp is not None and self._id_min <= disp <= self._id_max \
                        and disp not in self._free:
                    self._free.append(disp)
