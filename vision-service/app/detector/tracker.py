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

import heapq
import logging
import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)


def reid_options(det) -> dict:
    """VelocityTracker re-identification kwargs from a DetectorConfig."""
    return {
        "reid": det.reid,
        "reid_max_gap": det.reid_max_gap_frames,
        "reid_min_x_overlap": det.reid_min_x_overlap,
        "reid_bottom_tol": det.reid_bottom_tol_frac,
        "reid_max_width_ratio": det.reid_max_width_ratio,
        "reid_buffer_frac": det.reid_buffer_frac_per_frame,
        "reid_buffer_max": det.reid_buffer_max_frac,
        "reid_dir_min_speed": det.reid_dir_min_speed_px,
    }


class VelocityTracker:
    def __init__(self, history: int = 5, id_min: int = 10, id_max: int = 99,
                 reid: bool = True, reid_max_gap: int = 40,
                 reid_min_x_overlap: float = 0.5, reid_bottom_tol: float = 0.35,
                 reid_max_width_ratio: float = 1.6,
                 reid_buffer_frac: float = 0.03, reid_buffer_max: float = 0.25,
                 reid_dir_min_speed: float = 2.0):
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
        # Allocation policy: LOWEST free number first, but an id freed less than
        # _ID_QUARANTINE_FRAMES ago is skipped. Lowest-first keeps the on-screen
        # numbers small and familiar over a long session (a churning scene stays
        # in the 10-30s instead of marching through the whole range); the
        # quarantine ensures a number that just left one vessel cannot reappear
        # on a different one moments later. This also subsumes the older
        # invariant that a freed id must outlive the stabilizer's coast window
        # (max_coast_frames, default 8): the quarantine is far longer.
        self._id_min = id_min
        self._id_max = id_max
        self._display: Dict[int, int] = {}
        # Min-heap of free ids (lowest allocated first) + when each recycled id
        # was freed, for the quarantine check.
        self._free: list = list(range(id_min, id_max + 1))
        self._freed_at: Dict[int, int] = {}
        # Waterline re-identification (see resolve()). _ident holds each
        # canonical track's last box + label + seq; _alias maps re-identified
        # raw ids onto their canonical id. Both are pruned with the track.
        self._reid = reid
        self._reid_max_gap = max(0, reid_max_gap)
        self._reid_min_x_overlap = min(max(reid_min_x_overlap, 0.0), 1.0)
        self._reid_bottom_tol = max(0.0, reid_bottom_tol)
        self._reid_max_width_ratio = max(1.0, reid_max_width_ratio)
        # Buffered matching (C-BIoU, arXiv:2211.14317): the re-id gates widen
        # with the dropout gap, because the velocity prediction gets less
        # certain the longer the target was unseen. Growth per missed frame,
        # capped, both as fractions of the narrower box's dimension.
        self._reid_buffer_frac = max(0.0, reid_buffer_frac)
        self._reid_buffer_max = max(0.0, reid_buffer_max)
        # Direction-consistency gate (OC-SORT's observation-centric momentum,
        # arXiv:2203.14360): a mover at/above this speed (px/frame) can only be
        # re-identified ALONG its direction of travel. 0 disables.
        self._reid_dir_min_speed = max(0.0, reid_dir_min_speed)
        self._ident: Dict[int, Tuple[float, float, float, float, str, int]] = {}
        self._alias: Dict[int, int] = {}
        # seq each alias was last resolved through, so aliases whose RAW id the
        # backend stopped emitting can expire on their own: a long-lived vessel
        # that flickers mints a new raw id per flip, and without per-alias
        # expiry every one of them would live as long as the canonical track.
        self._alias_seen: Dict[int, int] = {}

    # A freed display id is not handed out again for this many frames (~15 s at
    # 10 fps), so a number that just left one vessel can't reappear on another
    # while the operator still associates it with the first.
    _ID_QUARANTINE_FRAMES = 150

    def _alloc_display(self, track_id: int, seq: int) -> int:
        # Lowest free id whose quarantine (if recycled) has expired.
        skipped: list = []
        try:
            while self._free:
                cand = heapq.heappop(self._free)
                if seq - self._freed_at.get(cand, -self._ID_QUARANTINE_FRAMES) \
                        >= self._ID_QUARANTINE_FRAMES:
                    self._freed_at.pop(cand, None)
                    return cand
                skipped.append(cand)
            if skipped:
                # Every free id is quarantined (heavy churn): take the one freed
                # longest ago rather than colliding with a live id.
                oldest = min(skipped, key=lambda c: self._freed_at.get(c, 0))
                skipped.remove(oldest)
                self._freed_at.pop(oldest, None)
                return oldest
        finally:
            for c in skipped:
                heapq.heappush(self._free, c)
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
        given box stands on, or ``None``. Best = largest horizontal overlap.

        Two gates keep an id from being handed to a DIFFERENT vessel that
        happens to occupy a vanished track's spot:

        * **width similarity** — the hull's waterline width is the invariant a
          partial/full flip preserves (a mast adds height, not width), so a
          candidate whose width differs by more than ``reid_max_width_ratio``
          is another vessel, not another extent of the same one.
        * **motion prediction** — the candidate's stored footprint is advanced
          by its last known pixel velocity over the gap before comparing, so a
          moving vessel is re-acquired where it *is now*, and a new arrival
          sitting where a moving vessel *used to be* no longer matches.

        Two association refinements from the MOT literature:

        * **buffered matching** (C-BIoU, arXiv:2211.14317) — the overlap and
          waterline gates WIDEN with the dropout gap (capped), because the
          velocity prediction is less certain the longer the target was
          unseen; a fresh flip is still judged tightly.
        * **direction consistency** (OC-SORT's observation-centric momentum,
          arXiv:2203.14360) — a candidate clearly displaced AGAINST a moving
          track's direction of travel is a different vessel, even when the
          widened gate would geometrically accept it.
        """
        best: Optional[int] = None
        best_ov = self._reid_min_x_overlap
        bottom = y + h
        misses: List[str] = []
        for tid, (ix, iy, iw, ih, ilabel, iseq) in self._ident.items():
            if tid == track_id or ilabel != label:
                continue
            gap = seq - iseq
            if gap > self._reid_max_gap:
                misses.append(f"raw={tid} gap={gap}>{self._reid_max_gap}")
                continue
            min_w = min(w, iw)
            if min_w <= 0:
                continue
            width_ratio = max(w, iw) / min_w
            if width_ratio > self._reid_max_width_ratio:
                misses.append(
                    f"raw={tid} width_ratio={width_ratio:.2f}"
                    f">{self._reid_max_width_ratio} new_w={w:.0f} old_w={iw:.0f}")
                continue
            pvx, pvy = self._last_velocity(tid)
            px, pb = ix + pvx * gap, (iy + ih) + pvy * gap
            buf_frac = min(self._reid_buffer_max, self._reid_buffer_frac * gap)
            ov = (min(x + w, px + iw) - max(x, px)
                  + 2.0 * buf_frac * min_w) / min_w
            if ov < best_ov:
                misses.append(
                    f"raw={tid} overlap={ov:.2f}<{best_ov:.2f}"
                    f" new=({x:.0f},{y:.0f},{w:.0f},{h:.0f})"
                    f" old=({ix:.0f},{iy:.0f},{iw:.0f},{ih:.0f})")
                continue
            bot_diff = abs(bottom - pb)
            bot_tol = (self._reid_bottom_tol + buf_frac) * min(h, ih)
            if bot_diff > bot_tol:
                misses.append(
                    f"raw={tid} bottom_diff={bot_diff:.1f}>{bot_tol:.1f}"
                    f" new_bot={bottom:.0f} old_bot={pb:.0f}"
                    f" min_h={min(h, ih):.0f}")
                continue
            if self._reid_dir_min_speed > 0 and \
                    math.hypot(pvx, pvy) >= self._reid_dir_min_speed:
                dx = (x + w / 2.0) - (ix + iw / 2.0)
                dy = bottom - (iy + ih)
                if math.hypot(dx, dy) > 0.25 * min_w and dx * pvx + dy * pvy < 0:
                    misses.append(f"raw={tid} direction")
                    continue
            best, best_ov = tid, ov
        if misses and best is None and _log.isEnabledFor(logging.DEBUG):
            _log.debug(
                "reid miss: new raw=%d %s box=(%.0f,%.0f,%.0f,%.0f) bot=%.0f"
                " rejected %d candidate(s): %s",
                track_id, label, x, y, w, h, bottom,
                len(misses), "; ".join(misses))
        return best

    def _last_velocity(self, track_id: int) -> Tuple[float, float]:
        """Waterline-anchor pixel velocity (px/frame) from a track's history,
        averaged over the window like :meth:`update`; (0, 0) if unknown."""
        hist = self._hist.get(track_id)
        if not hist or len(hist) < 2:
            return 0.0, 0.0
        (oseq, ocx, ocy), (lseq, lcx, lcy) = hist[0], hist[-1]
        dseq = max(lseq - oseq, 1)
        return (lcx - ocx) / dseq, (lcy - ocy) / dseq

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
            disp = self._alloc_display(track_id, seq)
            self._display[track_id] = disp
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
        # One sample per frame per canonical track: when re-id resolves a partial
        # AND a full detection of the same vessel to one id in a single frame,
        # the second call must not append a duplicate same-seq sample — that
        # would shrink the effective velocity window and skew _last_velocity().
        # First detection wins; velocity/age come out the same either way.
        if not (hist and hist[-1][0] == seq):
            hist.append((seq, cx, cy))
        age = seq - self._first_seq[track_id]
        return vx, vy, age

    def prune(self, active_ids: set, seq: int, max_idle: int = 0,
              max_idle_thin: int = 16) -> None:
        """Drop tracks not seen recently to bound memory. Tracks with fewer
        than 3 sightings ("thin": single-frame glints, wave crests) are dropped
        after the much shorter ``max_idle_thin`` — they were never shown (the
        stabilizer's confirm debounce needs 3 hits), so releasing their display
        id early costs nothing and keeps a churning scene from marching the
        visible numbers through the whole pool.

        ``max_idle`` defaults to ``self._reid_max_gap``: the re-id search window
        and the eviction window must match — otherwise _ident entries for lost
        tracks get pruned before _match_identity can use them, silently breaking
        re-id for any dropout longer than the old hardcoded 60-frame default.
        Pass an explicit value only in tests or when a tighter eviction is wanted.
        """
        if max_idle <= 0:
            max_idle = self._reid_max_gap
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
            idle_limit = max_idle if len(self._hist[tid]) >= 3 \
                else min(max_idle, max_idle_thin)
            last_seq = self._hist[tid][-1][0] if self._hist[tid] else 0
            if seq - last_seq > idle_limit:
                self._hist.pop(tid, None)
                self._first_seq.pop(tid, None)
                self._ident.pop(tid, None)
                # Aliases die with their canonical track: a raw id the backend
                # resurrects later must start (and re-match) fresh, not point at
                # state that no longer exists.
                for a in [a for a, c in self._alias.items() if c == tid]:
                    self._alias.pop(a, None)
                    self._alias_seen.pop(a, None)
                # Recycle the display id: back into the pool, stamped with the
                # frame it was freed so the allocator quarantines it (see
                # _alloc_display) before any reuse.
                disp = self._display.pop(tid, None)
                if disp is not None and self._id_min <= disp <= self._id_max \
                        and disp not in self._free:
                    heapq.heappush(self._free, disp)
                    self._freed_at[disp] = seq
