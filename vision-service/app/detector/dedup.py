"""Same-vessel duplicate suppression: collapse multiple detections of ONE
physical vessel into a single target before the event is built.

A sailing vessel routinely double-fires the detector: one box on the hull
alone and another on hull+mast (often under two different class labels, e.g.
``boat`` and ``sailboat``). Neither NMS pass catches this — NMS is per-class
on every backend, and even same-class NMS misses it because the IoU of a
hull-only box against a hull+mast box is roughly small_area/big_area, well
below the NMS threshold. Both boxes then become separate tracks, and the
plugin renders two chart blips (and doubled alerts) for one vessel.

Two passes, one :class:`TargetDeduper` instance per camera:

* **vessel-family merge** — two targets whose labels BOTH belong to
  :data:`~.classmap.VESSEL_FAMILY` are treated as the same vessel when the
  intersection covers at least ``vessel_ios`` of the *smaller* box
  (intersection-over-smaller, which is high for hull-vs-hull+mast even when
  IoU is low). The merge is **sticky**: the loser→winner pairing is remembered
  for ``hold_frames`` so the same track keeps winning while the pair overlaps
  (with a relaxed hysteresis threshold), instead of the surviving track id
  flapping frame to frame — that flapping is what kept two plugin blips alive.
  If the winner track disappears, the pairing expires and the other track
  surfaces again, so the vessel itself can never be suppressed away.
  Small-craft labels (kayak, buoy, person) are deliberately NOT in the family:
  a kayak occluding a larger vessel is two real objects and must never merge.

* **containment drop** — the pre-existing conservative rule for any label
  pair: a box lying almost entirely (``contained_frac``) inside a larger box
  is dropped (a buoy/person on a vessel's deck, a duplicate nested box). A
  person-in-water is never dropped (MOB safety).
"""

from __future__ import annotations

from ..schemas import BBox, Target
from .classmap import VESSEL_FAMILY


def _contained_fraction(inner: BBox, outer: BBox) -> float:
    """Fraction of *inner*'s area that overlaps *outer* (0..1)."""
    ix = max(inner.x, outer.x)
    iy = max(inner.y, outer.y)
    ax = min(inner.x + inner.w, outer.x + outer.w)
    ay = min(inner.y + inner.h, outer.y + outer.h)
    overlap = max(0.0, ax - ix) * max(0.0, ay - iy)
    inner_area = inner.w * inner.h
    return overlap / inner_area if inner_area > 0 else 0.0


def _intersection_over_smaller(a: BBox, b: BBox) -> float:
    """Intersection area as a fraction of the SMALLER box's area (0..1).

    The right overlap measure for the hull vs hull+mast case: the hull box is
    nearly fully covered by the taller box, so IoS is ~1.0 while IoU is only
    ~small/big (which is why NMS never merges the pair).
    """
    small = a if a.w * a.h <= b.w * b.h else b
    other = b if small is a else a
    return _contained_fraction(small, other)


def _drop_contained_targets(targets: list, frac: float) -> list:
    """Drop detections whose bbox lies largely inside a larger detection's bbox
    (a buoy/person on a vessel's deck, a duplicate nested box). The larger
    containing object is kept; a person-in-water is never dropped (MOB safety)."""
    if frac >= 1.0 or len(targets) < 2:
        return targets
    keep = [True] * len(targets)
    for i, outer in enumerate(targets):
        for j, inner in enumerate(targets):
            if i == j or not keep[i] or not keep[j] or inner.is_person_in_water:
                continue
            # `inner` must be the strictly smaller box of the pair.
            if outer.bbox.w * outer.bbox.h <= inner.bbox.w * inner.bbox.h:
                continue
            if _contained_fraction(inner.bbox, outer.bbox) > frac:
                keep[j] = False
    return [t for k, t in enumerate(targets) if keep[k]]


class TargetDeduper:
    """Per-camera duplicate suppressor; state is the sticky loser→winner map."""

    def __init__(self, vessel_ios: float = 0.55, contained_frac: float = 0.8,
                 hold_frames: int = 16, hysteresis: float = 0.7):
        # >= 1.0 disables the vessel-family merge (containment pass still runs).
        self.vessel_ios = vessel_ios
        self.contained_frac = contained_frac
        self.hold_frames = max(0, hold_frames)
        # While a pairing is live the merge threshold relaxes by this factor, so
        # a pair jittering around vessel_ios doesn't split/re-merge every frame.
        self.hysteresis = min(max(hysteresis, 0.0), 1.0)
        # loser track id -> (winner track id, seq of the last merge).
        self._pairs: dict[int, tuple[int, int]] = {}

    def update(self, targets: list[Target], seq: int) -> list[Target]:
        targets = self._merge_vessel_duplicates(targets, seq)
        return _drop_contained_targets(targets, self.contained_frac)

    # ── vessel-family merge ────────────────────────────────────────────────

    def _pair_live(self, a: Target, b: Target) -> bool:
        """True if a live pairing exists between the two tracks (either role)."""
        for loser, winner in ((a, b), (b, a)):
            if loser.track_id is None or winner.track_id is None:
                continue
            rec = self._pairs.get(loser.track_id)
            if rec is not None and rec[0] == winner.track_id:
                return True
        return False

    def _merge_vessel_duplicates(self, targets: list[Target], seq: int) -> list[Target]:
        # Expire pairings not refreshed within hold_frames, so a track that lost
        # once isn't suppressed forever and two vessels that separate re-split.
        for loser, (_, last) in list(self._pairs.items()):
            if seq - last > self.hold_frames:
                del self._pairs[loser]
        if self.vessel_ios >= 1.0:
            return targets

        vessels = [t for t in targets if t.label in VESSEL_FAMILY]
        if len(vessels) < 2:
            return targets

        # Sticky-winner bonus: a track that already won against another track
        # present in THIS frame outranks everything, so the surviving id stays
        # stable even when the loser's fresh box momentarily scores better.
        present = {t.track_id for t in vessels if t.track_id is not None}
        sticky_winners = {
            winner for loser, (winner, _) in self._pairs.items()
            if loser in present and winner in present
        }

        def priority(t: Target):
            return (
                t.track_id in sticky_winners,
                not t.coasting,        # a fresh detection beats a coasted ghost
                t.age_frames,          # the longest-lived track keeps its blip
                t.bbox.w * t.bbox.h,   # the fuller (hull+mast) box of the pair
                t.confidence,
            )

        dropped: set[int] = set()
        kept: list[Target] = []
        for cand in sorted(vessels, key=priority, reverse=True):
            winner = None
            for w in kept:
                thr = self.vessel_ios * (self.hysteresis if self._pair_live(cand, w) else 1.0)
                if _intersection_over_smaller(cand.bbox, w.bbox) >= thr:
                    winner = w
                    break
            if winner is None:
                kept.append(cand)
                continue
            dropped.add(id(cand))
            if cand.track_id is not None and winner.track_id is not None:
                self._pairs[cand.track_id] = (winner.track_id, seq)
                # A pairing is one-directional; drop any stale reverse record so
                # the two tracks can't hold a mutual claim on each other.
                rec = self._pairs.get(winner.track_id)
                if rec is not None and rec[0] == cand.track_id:
                    del self._pairs[winner.track_id]
        if not dropped:
            return targets
        return [t for t in targets if id(t) not in dropped]
