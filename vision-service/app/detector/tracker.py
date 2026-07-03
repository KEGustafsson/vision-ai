"""Per-track anchor-point history -> pixel velocity (px/frame), a compact,
recycled display id, and waterline re-identification, shared by all backends.

Backends provide stable but ever-growing raw track ids (ByteTrack/NvDCF counters
that climb without bound). This class fills in velocity and age, and maps each
raw id to a small, human-readable display id in a bounded range (10..99 by
default) so emitted detections carry a 2-digit number. One instance per camera
stream, so the bounded range is per camera.

Waterline re-identification (:meth:`resolve`) additionally keeps ONE id on a
vessel whose detected box alternates between partial and full extents (hull
only <-> hull+mast). The backend trackers associate by box IoU, so that shape
jump breaks association and mints a fresh raw id — the same target then
flickers between two display ids. But however much of the superstructure the
detector caught, the hull's waterline footprint (bottom edge and horizontal
extent) is the same, so a NEW raw id whose box stands on the footprint of a
recently seen track is aliased to that track and inherits its display id, age,
and velocity history.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Tuple


def reid_options(det) -> dict:
    """VelocityTracker re-identification kwargs from a DetectorConfig."""
    return {
        "reid": det.reid,
        "reid_max_gap": det.reid_max_gap_frames,
        "reid_min_x_overlap": det.reid_min_x_overlap,
        "reid_bottom_tol": det.reid_bottom_tol_frac,
    }


class VelocityTracker:
    def __init__(self, history: int = 5, id_min: int = 10, id_max: int = 99,
                 reid: bool = True, reid_max_gap: int = 16,
                 reid_min_x_overlap: float = 0.5, reid_bottom_tol: float = 0.35):
        if id_min > id_max:
            raise ValueError(f"id_min ({id_min}) must be <= id_max ({id_max})")
        self._hist: Dict[int, Deque[Tuple[int, float, float]]] = {}
        self._first_seq: Dict[int, int] = {}
        self._history = history
        # Compact, recycled display ids in [id_min, id_max]. We map each raw id to
        # a small number and return it to the pool when the track is pruned, so
        # detections always carry a stable id in a bounded range. Velocity/age
        # state stays keyed by the raw id internally.
        #
        # Invariant this relies on: a display id is recycled only after a raw
        # track has been idle for ``prune(max_idle=60)`` frames, which must stay
        # well above the stabilizer's ``max_coast_frames`` (default 8). Otherwise
        # a freed id could be handed to a new track while the stabilizer still
        # holds coasting state under that id, briefly fusing two objects.
        self._id_min = id_min
        self._id_max = id_max
        self._display: Dict[int, int] = {}
        # Deque so allocation (popleft) and recycling (append) are both O(1) and
        # the FIFO order expresses the rotation: freed ids go to the back and are
        # reused last, cycling through the whole range before any reuse.
        self._free: Deque[int] = deque(range(id_min, id_max + 1))
        # Waterline re-identification (see resolve()). _ident holds each
        # canonical track's last box + label + seq; _alias maps re-identified
        # raw ids onto their canonical id. Both are pruned with the track.
        self._reid = reid
        self._reid_max_gap = max(0, reid_max_gap)
        self._reid_min_x_overlap = min(max(reid_min_x_overlap, 0.0), 1.0)
        self._reid_bottom_tol = max(0.0, reid_bottom_tol)
        self._ident: Dict[int, Tuple[float, float, float, float, str, int]] = {}
        self._alias: Dict[int, int] = {}
        # seq each alias was last resolved through, so aliases whose RAW id the
        # backend stopped emitting can expire on their own: a long-lived vessel
        # that flickers mints a new raw id per flip, and without per-alias
        # expiry every one of them would live as long as the canonical track.
        self._alias_seen: Dict[int, int] = {}

    def set_id_range(self, id_min: int, id_max: int) -> None:
        """Resize the recycled display-id pool to follow max-targets-per-frame so
        EVERY emitted id stays within ``[id_min, id_max]``. Live tracks already in
        range keep their id (no needless reshuffle); any live track whose id is
        now out of range is remapped to a fresh in-range id immediately — so a
        shrink can never leave a higher id on the wire (at the cost of a one-time
        id change for those tracks). If more tracks are live than the range holds,
        the allocator wraps within range (ids may then collide, but never exceed).
        """
        if id_min > id_max:
            raise ValueError(f"id_min ({id_min}) must be <= id_max ({id_max})")
        if (id_min, id_max) == (self._id_min, self._id_max):
            return
        self._id_min, self._id_max = id_min, id_max
        in_range_held = {d for d in self._display.values() if id_min <= d <= id_max}
        self._free = deque(i for i in range(id_min, id_max + 1) if i not in in_range_held)
        for tid, disp in self._display.items():
            if disp < id_min or disp > id_max:
                self._display[tid] = self._alloc_display(tid)

    def _alloc_display(self, track_id: int) -> int:
        if self._free:
            return self._free.popleft()
        # Pool exhausted (more live tracks than the range can hold): fall back to
        # a wrapped value. May collide, but maxTargetsPerStream keeps this rare.
        span = self._id_max - self._id_min + 1
        return self._id_min + (track_id % span)

    def display_id(self, track_id: int) -> Optional[int]:
        """Bounded display id for a raw track id, or ``None`` if it was never
        registered via :meth:`update` (callers should map only tracked ids)."""
        return self._display.get(track_id)

    def resolve(self, track_id: int, seq: int, x: float, y: float,
                w: float, h: float, label: str) -> int:
        """Canonical raw id for a detection: re-identify a NEW backend track as
        an already-known target when both boxes stand on the same **waterline
        footprint** — high horizontal overlap and an aligned bottom edge. A
        partial re-detection (hull only) and a full one (hull + mast) differ
        wildly in box height, which breaks the backend's IoU association and
        mints a new raw id, but their waterline footprint is the same target's.

        Call before :meth:`update` / :meth:`display_id` each frame and key all
        per-track state on the returned id. ``person`` is exempt in BOTH
        directions (never re-identified, never a re-id candidate): two people in
        the water near each other must stay two MOB targets — silently fusing
        them could mask one of two live casualties.
        """
        canon = self._alias.get(track_id, track_id)
        if canon != track_id:
            self._alias_seen[track_id] = seq
        if self._reid and label != "person" and canon not in self._hist:
            match = self._match_identity(track_id, seq, x, y, w, h, label)
            if match is not None:
                canon = match
                self._alias[track_id] = canon
                self._alias_seen[track_id] = seq
        self._ident[canon] = (x, y, w, h, label, seq)
        return canon

    def _match_identity(self, track_id: int, seq: int, x: float, y: float,
                        w: float, h: float, label: str) -> Optional[int]:
        """Best recently-seen same-label track whose waterline footprint the
        given box stands on, or ``None``. Best = largest horizontal overlap."""
        best: Optional[int] = None
        best_ov = self._reid_min_x_overlap
        bottom = y + h
        for tid, (ix, iy, iw, ih, ilabel, iseq) in self._ident.items():
            if tid == track_id or ilabel != label:
                continue
            if seq - iseq > self._reid_max_gap:
                continue
            min_w = min(w, iw)
            if min_w <= 0:
                continue
            # Fraction of the narrower box's width shared with the candidate.
            ov = (min(x + w, ix + iw) - max(x, ix)) / min_w
            if ov < best_ov:
                continue
            # Bottom edges must sit on the same waterline, within a tolerance
            # scaled by the SHORTER box (the hull) — a mast-height tolerance
            # would happily bridge two stacked targets.
            if abs(bottom - (iy + ih)) > self._reid_bottom_tol * min(h, ih):
                continue
            best, best_ov = tid, ov
        return best

    def update(self, track_id: int, seq: int, cx: float, cy: float) -> Tuple[float, float, int]:
        """Return (vx, vy, age_frames) for a track anchor point at the given seq.

        Callers pass the box's **bottom-center** (waterline anchor), not its
        centroid: the waterline stays put when a re-identified box flips between
        partial and full extents, so the merged history yields a clean velocity
        where a centroid would see the box's half-height jump every flip.
        """
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
        # Expire aliases whose raw id hasn't been resolved recently, even while
        # their canonical track lives on — otherwise a flickering vessel grows
        # one immortal alias per flip over a long session.
        for raw, last in list(self._alias_seen.items()):
            if raw not in active_ids and seq - last > max_idle:
                self._alias.pop(raw, None)
                self._alias_seen.pop(raw, None)
        for tid in list(self._hist.keys()):
            if tid in active_ids:
                continue
            last_seq = self._hist[tid][-1][0] if self._hist[tid] else 0
            if seq - last_seq > max_idle:
                self._hist.pop(tid, None)
                self._first_seq.pop(tid, None)
                self._ident.pop(tid, None)
                # Aliases die with their canonical track: a raw id the backend
                # resurrects later must start (and re-match) fresh, not point at
                # state that no longer exists.
                for a in [a for a, c in self._alias.items() if c == tid]:
                    self._alias.pop(a, None)
                    self._alias_seen.pop(a, None)
                # Recycle the display id by appending to the *back* of the pool,
                # so freed numbers rotate to the end and are reused last — the
                # allocator keeps cycling through the whole range before handing a
                # just-freed number to a new track.
                disp = self._display.pop(tid, None)
                if disp is not None and self._id_min <= disp <= self._id_max \
                        and disp not in self._free:
                    self._free.append(disp)
