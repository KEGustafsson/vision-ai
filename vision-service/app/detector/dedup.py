"""Same-vessel duplicate suppression: collapse multiple detections of ONE
physical vessel into a single target — with a single, stable target number —
before the event is built.

A sailing vessel routinely double-fires the detector: one box on the hull
alone and another on hull+mast (often under two different class labels, e.g.
``boat`` and ``sailboat``). Neither NMS pass catches this — NMS is per-class
on every backend, and even same-class NMS misses it because the IoU of a
hull-only box against a hull+mast box is roughly small_area/big_area, well
below the NMS threshold. Both boxes then become separate tracks, and the
plugin renders two chart blips (and doubled alerts) for one vessel.

The same vessel also produces duplicate *numbers over time*: when the box
flips between hull-size and full-size the tracker's IoU association breaks
and issues a NEW track id, while the old number's blip is held downstream for
blipHoldS (15 s) — so the chart shows two numbers even when each frame only
ever contained one box. Suppressing duplicates per frame is therefore not
enough; the published identity has to survive the flip.

One :class:`TargetDeduper` instance per camera, three mechanisms:

* **vessel-family merge** — two targets whose labels BOTH belong to
  :data:`~.classmap.VESSEL_FAMILY` are treated as the same vessel when the
  intersection covers at least ``vessel_ios`` of the *smaller* box
  (intersection-over-smaller, which is high for hull-vs-hull+mast even when
  IoU is low). The merge is **sticky**: the loser→winner pairing is remembered
  for ``hold_frames`` (with a relaxed hysteresis threshold) so the same track
  keeps winning while the pair overlaps. If the winner track disappears, the
  pairing expires and the other track surfaces again, so the vessel itself
  can never be suppressed away. Small-craft labels (kayak, buoy, person) are
  deliberately NOT in the family: a kayak occluding a larger vessel is two
  real objects and must never merge.

* **published-id continuity (aliasing)** — the emitted target NUMBER outlives
  tracker churn. When a brand-new track appears where a recently-vanished
  vessel was (succession), or a fresh track beats a veteran track in a merge
  (inheritance), the new track is re-published under the old track's id for
  as long as it lives. Downstream keeps updating ONE blip instead of holding
  the dead number for blipHoldS next to the new one. Aliases break the moment
  the original id shows up as a live track again (two real vessels that
  separated, or a recycled display id), so two distinct vessels can never be
  fused by a stale alias. ``succession_frames`` must stay below the
  VelocityTracker display-id recycle window (``prune(max_idle=60)``) so an
  alias can't resurrect a number that was already handed to a new track.

* **containment drop** — the pre-existing conservative rule for any label
  pair: a box lying almost entirely (``contained_frac``) inside a larger box
  is dropped (a buoy/person on a vessel's deck, a duplicate nested box). A
  person-in-water is never dropped (MOB safety).
"""

from __future__ import annotations

from typing import Optional

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
    """Per-camera duplicate suppressor; state: sticky pairings + id aliases."""

    def __init__(self, vessel_ios: float = 0.55, contained_frac: float = 0.8,
                 hold_frames: int = 16, hysteresis: float = 0.7,
                 succession_frames: int = 50):
        # >= 1.0 disables the vessel-family merge AND id continuity (the
        # containment pass still runs).
        self.vessel_ios = vessel_ios
        self.contained_frac = contained_frac
        self.hold_frames = max(0, hold_frames)
        # While a pairing is live the merge threshold relaxes by this factor, so
        # a pair jittering around vessel_ios doesn't split/re-merge every frame.
        self.hysteresis = min(max(hysteresis, 0.0), 1.0)
        # How long a vanished vessel's number stays claimable by a successor
        # track appearing in the same spot. MUST stay below the display-id
        # recycle window (VelocityTracker prune max_idle=60) so we never stitch
        # onto a number that was already reissued to an unrelated new track.
        self.succession_frames = max(0, succession_frames)
        # loser PUBLISHED id -> (winner PUBLISHED id, seq of the last merge).
        self._pairs: dict[int, tuple[int, int]] = {}
        # Identity continuity: raw tracker id -> published id it inherits.
        self._alias: dict[int, int] = {}
        # raw tracker id -> last seq it was present (GC for _alias).
        self._alias_seen: dict[int, int] = {}
        # published vessel id -> (last emitted bbox, seq); succession candidates.
        self._last_seen: dict[int, tuple[BBox, int]] = {}

    def update(self, targets: list[Target], seq: int) -> list[Target]:
        if self.vessel_ios < 1.0:
            targets = self._dedup_vessels(targets, seq)
        return _drop_contained_targets(targets, self.contained_frac)

    # ── vessel-family merge + published-id continuity ─────────────────────

    def _pair_live(self, a: Target, b: Target) -> bool:
        """True if a live pairing exists between the two tracks (either role)."""
        for loser, winner in ((a, b), (b, a)):
            if loser.track_id is None or winner.track_id is None:
                continue
            rec = self._pairs.get(loser.track_id)
            if rec is not None and rec[0] == winner.track_id:
                return True
        return False

    def _expire(self, seq: int) -> None:
        # Pairings not refreshed within hold_frames lapse, so a track that lost
        # once isn't suppressed forever and two vessels that separate re-split.
        for loser, (_, last) in list(self._pairs.items()):
            if seq - last > self.hold_frames:
                del self._pairs[loser]
        # A vanished number stops being claimable after succession_frames, and
        # an alias whose raw track is gone that long is dead too.
        for pid, (_, last) in list(self._last_seen.items()):
            if seq - last > self.succession_frames:
                del self._last_seen[pid]
        for raw, last in list(self._alias_seen.items()):
            if seq - last > self.succession_frames:
                del self._alias_seen[raw]
                self._alias.pop(raw, None)

    def _dedup_vessels(self, targets: list[Target], seq: int) -> list[Target]:
        self._expire(seq)
        vessels = [t for t in targets if t.label in VESSEL_FAMILY]
        if not vessels:
            return targets
        # Raw (tracker) id per target, recorded before any rewrite: aliases are
        # keyed by the id the tracker will keep using on later frames.
        raw_of: dict[int, Optional[int]] = {id(t): t.track_id for t in vessels}
        raw_present = {t.track_id for t in vessels if t.track_id is not None}

        # Alias resolution. Revival guard first: if the number's original owner
        # is back as a live track (the pair really was two vessels and they
        # separated, or the display id got recycled), the alias must break —
        # impersonating a live track would fuse two real objects on the wire.
        for raw, pub in list(self._alias.items()):
            if pub in raw_present:
                del self._alias[raw]
        for t in vessels:
            raw = raw_of[id(t)]
            if raw is None:
                continue
            self._alias_seen[raw] = seq
            pub = self._alias.get(raw)
            if pub is not None:
                t.track_id = pub
        published = {t.track_id for t in vessels if t.track_id is not None}

        # Succession: a brand-new track that appears where a recently-vanished
        # vessel was inherits that vessel's number, so a tracker id churn (the
        # box flipping hull <-> hull+mast breaks IoU association) doesn't mint
        # a new target number while the old one is still held downstream.
        for t in vessels:
            raw = raw_of[id(t)]
            if raw is None or t.track_id != raw or raw in self._last_seen:
                continue  # untracked, already aliased, or a known identity
            best_id, best_ios = None, self.vessel_ios
            for old_id, (box, _) in self._last_seen.items():
                if old_id in published:
                    continue  # number still in use by a live track
                ios = _intersection_over_smaller(t.bbox, box)
                if ios >= best_ios:
                    best_id, best_ios = old_id, ios
            if best_id is not None:
                self._alias[raw] = best_id
                t.track_id = best_id
                published.add(best_id)

        dropped = self._merge(vessels, raw_of, seq)
        out = [t for t in targets if id(t) not in dropped]

        # Two live tracks can end up published under one number only when their
        # aliases both point at the same dead id and the boxes have since
        # separated (so the merge no longer collapses them). Revert the younger
        # claim to its raw tracker id — one number must never label two boxes.
        by_pub: dict[int, Target] = {}
        for t in out:
            if t.label not in VESSEL_FAMILY or t.track_id is None:
                continue
            prev = by_pub.get(t.track_id)
            if prev is None:
                by_pub[t.track_id] = t
                continue
            keep, revert = (prev, t) if prev.age_frames >= t.age_frames else (t, prev)
            by_pub[keep.track_id] = keep
            raw = raw_of.get(id(revert))
            if raw is not None:
                self._alias.pop(raw, None)
                revert.track_id = raw

        for t in out:
            if t.label in VESSEL_FAMILY and t.track_id is not None:
                self._last_seen[t.track_id] = (t.bbox, seq)
        return out

    def _merge(self, vessels: list[Target], raw_of: dict, seq: int) -> set:
        """Greedy per-frame merge of overlapping vessel-family boxes. Returns
        the ``id()`` set of suppressed targets; may rewrite a winner's track_id
        (inheritance) so the published number stays the veteran one."""
        if len(vessels) < 2:
            return set()

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
            if cand.track_id is None or winner.track_id is None \
                    or cand.track_id == winner.track_id:
                continue
            # Inheritance: when a brand-new track beats the track whose number
            # was being published until now (tracker churn: the fresh full box
            # outranks the veteran coasted one), publish the winner under that
            # established number so downstream sees one continuous target
            # instead of a new number beside a held old one. Keyed on the
            # PUBLISHED history (_last_seen), never raw track age — a long-lived
            # but always-suppressed loser must not hijack the number.
            if cand.track_id in self._last_seen and winner.track_id not in self._last_seen:
                raw_w = raw_of.get(id(winner))
                if raw_w is not None:
                    self._alias[raw_w] = cand.track_id
                    winner.track_id = cand.track_id
            if cand.track_id != winner.track_id:
                self._pairs[cand.track_id] = (winner.track_id, seq)
                # A pairing is one-directional; drop any stale reverse record so
                # the two tracks can't hold a mutual claim on each other.
                rec = self._pairs.get(winner.track_id)
                if rec is not None and rec[0] == cand.track_id:
                    del self._pairs[winner.track_id]
        return dropped
