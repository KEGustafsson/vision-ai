"""Per-track anchor-point history -> pixel velocity (px/frame), a compact,
recycled display id, and waterline re-identification, shared by all backends.

Backends provide stable but ever-growing raw track ids (ByteTrack/NvDCF counters
that climb without bound). This class fills in velocity and age, and maps each
raw id to a small, human-readable display id in a bounded range (10..99 by
default) so emitted detections carry a 2-digit number, plus a per-session
serial (``stable_id``) that is NEVER recycled, for downstream identity (the
SignalK blip name) that must not change physical vessel when a display number
is reused. One instance per camera stream, so both id spaces are per camera.

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
import math
from collections import deque
from typing import Deque, Dict, Optional, Tuple


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
        "max_idle": det.track_memory_frames,
    }


class VelocityTracker:
    def __init__(self, history: int = 5, id_min: int = 10, id_max: int = 99,
                 reid: bool = True, reid_max_gap: int = 120,
                 reid_min_x_overlap: float = 0.5, reid_bottom_tol: float = 0.35,
                 reid_max_width_ratio: float = 1.6,
                 reid_buffer_frac: float = 0.03, reid_buffer_max: float = 0.25,
                 reid_dir_min_speed: float = 2.0,
                 max_idle: int = 260, frame_w: Optional[float] = None):
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
        # Per-session, NEVER-recycled serial per canonical track (stable_id on
        # the wire). The display id above is bounded and recycled, so over a
        # long session the same 2-digit number legitimately names different
        # vessels; downstream identity (SignalK blip name/URN) keys on this
        # serial instead, so a chart contact can never change physical vessel.
        self._serial: Dict[int, int] = {}
        self._next_serial = 1
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
        # How long an idle track's state (velocity history, identity footprint,
        # display id, serial) is retained; prune() uses this unless overridden.
        # MUST cover the deepest backend resurrection window: NvDCF holds a lost
        # raw id in shadow for maxShadowTrackingAge frames (240 in
        # nvdcf_config.yml) and may re-acquire the vessel with the SAME raw id —
        # if we forget sooner, that reborn track gets a fresh display id (the
        # quarantine even guarantees a different number) and a new blip identity
        # downstream, silently defeating the shadow tracking. Also must exceed
        # reid_max_gap, or the waterline re-id loses its candidates first.
        self._max_idle = max(0, max_idle)
        # Frame width, for the edge-clip relaxation in _match_identity(): a box
        # clipped by the left/right frame edge has an unreliable width, so the
        # waterline-width gate must not judge it. None disables (unknown size).
        self._frame_w = frame_w
        self._ident: Dict[int, Tuple[float, float, float, float, str, int]] = {}
        self._alias: Dict[int, int] = {}
        # Alternation-merge evidence (see _merge_alternating): per unordered
        # pair of live canonical ids, how many frames EACH side was detected
        # while the other was briefly dark on the same waterline footprint —
        # and when the pair was last CO-detected side by side (proof of two
        # distinct vessels, which blocks merging and re-aliasing).
        self._alt_evidence: Dict[Tuple[int, int], list] = {}
        self._co_seen: Dict[Tuple[int, int], int] = {}
        # Merge dissolution (see _check_split): the first raw id resolved to
        # each canonical this frame (+ its box), and per-canonical count of
        # contradiction frames — two raw ids resolving to ONE canonical in the
        # same frame at DISJOINT footprints, which proves an earlier merge (or
        # re-id alias) wrong.
        self._frame_res: Dict[int, Tuple[int, int, float, float, float, float]] = {}
        self._split_evidence: Dict[int, int] = {}
        # seq each alias was last resolved through, so aliases whose RAW id the
        # backend stopped emitting can expire on their own: a long-lived vessel
        # that flickers mints a new raw id per flip, and without per-alias
        # expiry every one of them would live as long as the canonical track.
        self._alias_seen: Dict[int, int] = {}

    # A freed display id is not handed out again for this many frames (~25 s at
    # the measured ~6 FPS per camera), so a number that just left one vessel
    # can't reappear on another while the operator still associates it with the
    # first. Must stay above the plugin's blipHoldS (15 s) worth of frames so a
    # still-held chart blip can never watch its number land on a new vessel.
    _ID_QUARANTINE_FRAMES = 150

    # A bbox edge within this many pixels of the frame boundary counts as
    # clipped for the re-id width gate (see _match_identity).
    _EDGE_MARGIN_PX = 2.0

    # Alternation merge (see _merge_alternating). Evidence needed from EACH
    # side of a pair (both tracks must take detected turns — a one-sided count
    # would let a live track swallow a departed neighbour), the total across
    # both sides, how recently the dark side must have been detected for a
    # frame to count as a turn, and how long a co-detection (both tracks seen
    # in ONE frame: two real vessels) blocks the pair from merging.
    _MERGE_CONFIRM_EACH = 3
    _MERGE_CONFIRM_TOTAL = 10
    _MERGE_MAX_GAP = 60
    _CO_BLOCK_FRAMES = 120
    # Merge dissolution: this many contradiction frames (two raw ids on one
    # canonical in the SAME frame at disjoint footprints) un-does the alias.
    # Observed live: a pair merged while genuinely co-located later separated,
    # and the fused id flapped ~90 px between two boats every few frames.
    _SPLIT_CONFIRM = 10
    # Same-frame co-detection is only proof of two DISTINCT vessels when the
    # boxes are side by side. When the narrower box stands at least this
    # fraction inside the wider one (same signature contained_frac dropping
    # uses), it is the classic hull-inside-full DUPLICATE of a single vessel —
    # nested boxes survive NMS, so NvDCF holds both tracks and detects them in
    # the same raw frame routinely (verified live: a pair alternating 84 times
    # in events never merged because raw-level duplicate co-detections kept
    # blocking it). Duplicate-style co-detection counts as merge evidence for
    # both sides (see _merge_alternating).
    _CO_DISTINCT_MAX_OVERLAP = 0.8

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

    def stable_id(self, track_id: int) -> Optional[int]:
        """Per-session serial for a canonical track id: monotonically
        increasing, never recycled, so it names one physical target for the
        whole session (unlike the bounded display id, which is recycled).
        ``None`` if the id was never registered via :meth:`update`."""
        return self._serial.get(track_id)

    def _edge_clipped(self, x: float, w: float) -> bool:
        """True when a box touches the left/right frame edge (within
        _EDGE_MARGIN_PX), i.e. its detected width is clipped and unreliable.
        Always False when the frame width is unknown."""
        if self._frame_w is None:
            return False
        return x <= self._EDGE_MARGIN_PX or \
            x + w >= self._frame_w - self._EDGE_MARGIN_PX

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
        # Merge dissolution: a SECOND raw id resolving to this canonical in
        # the SAME frame at a disjoint footprint contradicts the alias that
        # binds them (a vessel is one detection, or a nested duplicate — not
        # two boxes apart). Enough contradictions un-do the alias, so a merge
        # that later turns out to have bound two real vessels is reversible.
        prev = self._frame_res.get(canon)
        if prev is not None and prev[0] == seq and prev[1] != track_id:
            if self._check_split(canon, prev, track_id, seq, x, y, w, h) == track_id:
                canon = track_id  # this raw id's alias dissolved: own track now
                self._frame_res[canon] = (seq, track_id, x, y, w, h)
        else:
            self._frame_res[canon] = (seq, track_id, x, y, w, h)
        self._ident[canon] = (x, y, w, h, label, seq)
        return canon

    def _check_split(self, canon: int, prev: Tuple, track_id: int, seq: int,
                     x: float, y: float, w: float, h: float) -> Optional[int]:
        """Judge a same-frame double resolution of ``canon`` and, after
        _SPLIT_CONFIRM contradiction frames, dissolve the alias of one of the
        two raw ids (returned; ``None`` when nothing was dissolved). A nested
        pair (the routine hull-inside-full duplicate) is consistent and clears
        the count; partial overlap is ambiguous and neutral; disjoint boxes
        are the contradiction. The dissolved id is the ALIASED one farther
        from the track's last anchor, so the canonical identity stays with
        the vessel that carried it; the pair is co-blocked against an
        immediate re-alias or re-merge."""
        px, py, pw, ph = prev[2:]
        min_w = min(w, pw)
        ov = (min(x + w, px + pw) - max(x, px)) / min_w if min_w > 0 else 1.0
        if ov >= self._CO_DISTINCT_MAX_OVERLAP:
            self._split_evidence.pop(canon, None)
            return None
        if ov >= self._reid_min_x_overlap:
            return None
        n = self._split_evidence.get(canon, 0) + 1
        if n < self._SPLIT_CONFIRM:
            self._split_evidence[canon] = n
            return None
        self._split_evidence.pop(canon, None)
        victims = [r for r in (track_id, prev[1]) if r in self._alias]
        if not victims:
            return None
        if len(victims) == 2 and self._hist.get(canon):
            _, acx, acy = self._hist[canon][-1]

            def _dev(r: int) -> float:
                bx, by, bw, bh = (x, y, w, h) if r == track_id \
                    else (px, py, pw, ph)
                return abs(bx + bw / 2.0 - acx) + abs(by + bh - acy)

            victims.sort(key=_dev, reverse=True)
        victim = victims[0]
        self._alias.pop(victim, None)
        self._alias_seen.pop(victim, None)
        pair = (victim, canon) if victim < canon else (canon, victim)
        self._co_seen[pair] = seq
        return victim

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
        best_ov = -1.0
        for tid in self._ident:
            if tid == track_id:
                continue
            # A pair recently proven distinct (side-by-side co-detection, or a
            # dissolved merge) must not immediately re-alias.
            pair = (track_id, tid) if track_id < tid else (tid, track_id)
            if seq - self._co_seen.get(pair, -self._CO_BLOCK_FRAMES) \
                    < self._CO_BLOCK_FRAMES:
                continue
            ov = self._footprint_overlap(x, y, w, h, label, tid, seq)
            if ov is not None and ov > best_ov:
                best, best_ov = tid, ov
        return best

    def _footprint_overlap(self, x: float, y: float, w: float, h: float,
                           label: str, tid: int, seq: int) -> Optional[float]:
        """Horizontal-overlap score when the given box stands on ``tid``'s
        stored waterline footprint, or ``None`` when any re-id gate rejects
        the pairing. Shared by :meth:`_match_identity` (aliasing a NEW raw id)
        and :meth:`_merge_alternating` (fusing two live tracks), so both
        judge "same footprint" by exactly the same rules."""
        ident = self._ident.get(tid)
        if ident is None:
            return None
        ix, iy, iw, ih, ilabel, iseq = ident
        if ilabel != label:
            return None
        gap = seq - iseq
        if gap > self._reid_max_gap:
            return None
        min_w = min(w, iw)
        if min_w <= 0:
            return None
        # Same hull, or a different vessel? The waterline width must agree —
        # UNLESS either box is clipped by a frame edge, where the detected
        # width is whatever happened to fit on screen (observed live: one
        # vessel exiting frame-right churned through four ids because each
        # re-entry width failed this gate). The overlap, waterline and
        # direction gates below still apply to an edge-clipped candidate.
        if max(w, iw) / min_w > self._reid_max_width_ratio and \
                not (self._edge_clipped(x, w) or self._edge_clipped(ix, iw)):
            return None
        # Advance the stored footprint by the track's waterline velocity so
        # the comparison happens where the vessel should be NOW.
        pvx, pvy = self._last_velocity(tid)
        px, pb = ix + pvx * gap, (iy + ih) + pvy * gap
        # Buffer for this gap: how much the matching space is expanded, as
        # a fraction of the narrower box's dimension (C-BIoU).
        buf_frac = min(self._reid_buffer_max, self._reid_buffer_frac * gap)
        # Fraction of the narrower box's width shared with the candidate,
        # after buffering both boxes horizontally.
        ov = (min(x + w, px + iw) - max(x, px)
              + 2.0 * buf_frac * min_w) / min_w
        if ov < self._reid_min_x_overlap:
            return None
        # Bottom edges must sit on the same waterline, within a tolerance
        # scaled by the SHORTER box (the hull) — a mast-height tolerance
        # would happily bridge two stacked targets. The buffer relaxes it
        # for long gaps (pitch/roll moves the waterline while unseen).
        bottom = y + h
        if abs(bottom - pb) > (self._reid_bottom_tol + buf_frac) * min(h, ih):
            return None
        # Direction consistency: a track that was clearly moving can only
        # be re-acquired by a candidate displaced broadly ALONG its motion.
        # A small displacement (a shape flip in place) is always allowed.
        if self._reid_dir_min_speed > 0 and \
                math.hypot(pvx, pvy) >= self._reid_dir_min_speed:
            dx = (x + w / 2.0) - (ix + iw / 2.0)
            dy = bottom - (iy + ih)
            if math.hypot(dx, dy) > 0.25 * min_w and dx * pvx + dy * pvy < 0:
                return None
        return ov

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
            self._display[track_id] = self._alloc_display(track_id, seq)
            self._serial[track_id] = self._next_serial
            self._next_serial += 1
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
        # Of the frame's candidates, keep the anchor CLOSEST to the previous
        # frame's (continuity): duplicate extents can sit tens of pixels apart,
        # and letting detection order pick would flap the anchor — and with it
        # the velocity — between them from frame to frame.
        if hist and hist[-1][0] == seq:
            if len(hist) >= 2:
                _, pcx, pcy = hist[-2]
                if abs(cx - pcx) + abs(cy - pcy) < \
                        abs(hist[-1][1] - pcx) + abs(hist[-1][2] - pcy):
                    hist[-1] = (seq, cx, cy)
        else:
            hist.append((seq, cx, cy))
        age = seq - self._first_seq[track_id]
        return vx, vy, age

    def _merge_alternating(self, active_ids: set, seq: int) -> None:
        """Fuse two live canonical tracks that are one physical vessel.

        The birth-time waterline re-id (:meth:`resolve`) can only alias a raw
        id at its FIRST sighting; if the gates momentarily failed then (pitch,
        a bad first box), the vessel ends up with two live tracks that take
        turns being detected — measured live: pairs alternating 60-300 times
        while co-detected 0-5 frames — and its id, box extent and published
        range flap between the two forever. This pass repairs that: when one
        track of a same-footprint pair is detected in a frame where the other
        is briefly dark, that is one unit of alternation evidence for the
        pair; enough evidence from BOTH sides merges the younger track into
        the older (which keeps its display id, serial and age).

        A frame where BOTH tracks are detected SIDE BY SIDE is proof of two
        distinct vessels: it resets the pair's evidence and blocks merging
        for _CO_BLOCK_FRAMES. This is what keeps two moored boats sharing a
        footprint apart — vessels genuinely side by side keep co-occurring.
        But a co-detection where one box stands nested inside the other is
        the duplicate signature of a SINGLE vessel — the same geometry
        _drop_contained_targets already treats as one object (hull box inside
        the hull+mast box; nested boxes survive NMS, so the raw tracker sees
        both together routinely even though the event shows only one) — and
        counts FOR the merge, on both sides at once: a hull track co-detected
        under its mast track every frame never takes a detected-while-
        partner-dark turn, so nested co-detections are the only evidence such
        a pair can produce. See _CO_DISTINCT_MAX_OVERLAP. person is exempt as
        everywhere in re-id (two swimmers must never fuse). Requires 3+
        sightings per track so glints don't vote.
        """
        if not self._reid:
            return
        act = [a for a in active_ids
               if a in self._ident and self._ident[a][4] != "person"
               and len(self._hist.get(a, ())) >= 3]
        for a in act:
            if a not in self._ident:
                continue  # merged away as the younger of an earlier pair
            ax, ay, aw, ah, albl, _ = self._ident[a]
            for b, (_, _, _, _, blbl, bseq) in list(self._ident.items()):
                if b == a or blbl == "person" or len(self._hist.get(b, ())) < 3:
                    continue
                pair = (a, b) if a < b else (b, a)
                if b in active_ids:
                    # Judge co-detection once per pair per frame (a < b side).
                    # Side-by-side simultaneous detections are two real
                    # vessels: block and reset. A NESTED pair (overlap of the
                    # narrower >= _CO_DISTINCT_MAX_OVERLAP) is the duplicate
                    # signature of ONE vessel and counts as evidence on BOTH
                    # sides: a hull track detected every frame under a
                    # co-detected hull+mast track never goes dark, so it can
                    # take no alternation turns — the nested co-detections
                    # themselves are the only signal such a pair emits
                    # (verified live: same-footprint pairs with wratio 1.03
                    # sat unmerged through 50+ event-level alternations).
                    if a < b:
                        ov = self._footprint_overlap(
                            ax, ay, aw, ah, albl, b, seq)
                        if ov is not None:
                            if ov < self._CO_DISTINCT_MAX_OVERLAP:
                                self._co_seen[pair] = seq
                                self._alt_evidence.pop(pair, None)
                            else:
                                ev = self._alt_evidence.setdefault(pair, [0, 0])
                                ev[0] += 1
                                ev[1] += 1
                                if min(ev) >= self._MERGE_CONFIRM_EACH and \
                                        sum(ev) >= self._MERGE_CONFIRM_TOTAL:
                                    self._merge(a, b, seq)
                                    if a not in self._ident:
                                        break
                    continue
                gap = seq - bseq
                if gap <= 0 or gap > self._MERGE_MAX_GAP:
                    continue  # dark side not in a brief dropout: no turn taken
                if seq - self._co_seen.get(pair, -self._CO_BLOCK_FRAMES) \
                        < self._CO_BLOCK_FRAMES:
                    continue
                if self._footprint_overlap(
                        ax, ay, aw, ah, albl, b, seq) is None:
                    continue
                ev = self._alt_evidence.setdefault(pair, [0, 0])
                ev[0 if a == pair[0] else 1] += 1
                if min(ev) >= self._MERGE_CONFIRM_EACH and \
                        sum(ev) >= self._MERGE_CONFIRM_TOTAL:
                    self._merge(a, b, seq)
                    if a not in self._ident:
                        break  # a was the younger: stop pairing it further

    def _merge(self, a: int, b: int, seq: int) -> None:
        """Merge canonical track ``b`` and ``a``: the OLDER keeps its display
        id, serial, age and history (operator familiarity + longer track); the
        younger becomes an alias of it and releases its ids. The loser's
        display id goes back through the normal quarantine."""
        keep, lose = ((a, b) if self._first_seq.get(a, seq)
                      <= self._first_seq.get(b, seq) else (b, a))
        for raw, canon in list(self._alias.items()):
            if canon == lose:
                self._alias[raw] = keep
        self._alias[lose] = keep
        self._alias_seen[lose] = seq
        self._hist.pop(lose, None)
        self._first_seq.pop(lose, None)
        self._ident.pop(lose, None)
        self._serial.pop(lose, None)
        self._frame_res.pop(lose, None)
        self._split_evidence.pop(lose, None)
        disp = self._display.pop(lose, None)
        if disp is not None and self._id_min <= disp <= self._id_max \
                and disp not in self._free:
            heapq.heappush(self._free, disp)
            self._freed_at[disp] = seq
        for d in (self._alt_evidence, self._co_seen):
            for p in [p for p in d if lose in p]:
                d.pop(p, None)

    def prune(self, active_ids: set, seq: int, max_idle: Optional[int] = None,
              max_idle_thin: int = 16) -> None:
        """Drop tracks not seen recently to bound memory. ``max_idle`` defaults
        to the constructor's retention window (see ``self._max_idle``: it must
        outlive NvDCF's shadow-tracking age, or a shadow-reacquired raw id
        returns to find its display id already freed and quarantined and the
        same vessel reblips under a new identity). Tracks with fewer than 3
        sightings ("thin": single-frame glints, wave crests) are dropped after
        the much shorter ``max_idle_thin`` — they were never shown (the
        stabilizer's confirm debounce needs 3 hits), so releasing their display
        id early costs nothing and keeps a churning scene from marching the
        visible numbers through the whole pool."""
        if max_idle is None:
            max_idle = self._max_idle
        # Merge alternating same-footprint tracks first, so a pair mid-merge
        # can't have its dark side idle-pruned out from under the evidence.
        self._merge_alternating(active_ids, seq)
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
                # The serial dies with the track and is never re-issued (the
                # counter only climbs), so no successor can inherit it.
                self._serial.pop(tid, None)
                self._frame_res.pop(tid, None)
                self._split_evidence.pop(tid, None)
                for d in (self._alt_evidence, self._co_seen):
                    for p in [p for p in d if tid in p]:
                        d.pop(p, None)
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
