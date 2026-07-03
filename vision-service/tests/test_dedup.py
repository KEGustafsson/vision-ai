"""Same-vessel duplicate suppression: a sailing vessel double-firing the
detector (hull-only box + hull+mast box, often under two class labels) must
collapse to ONE target with a stable surviving track id, while genuinely
distinct objects (a kayak or buoy overlapping a vessel, a person in the water)
are never merged away."""

from app.detector.dedup import TargetDeduper, _drop_contained_targets
from app.schemas import BBox, Geometry, Target


def _target(label: str, tid: int, x: float, y: float, w: float, h: float,
            conf: float = 0.8, age: int = 0, coasting: bool = False,
            piw: bool = False) -> Target:
    return Target(
        track_id=tid, label=label, coco_class=8, confidence=conf,
        bbox=BBox(x=x, y=y, w=w, h=h), is_person_in_water=piw,
        geometry=Geometry(relative_bearing_deg=0.0),
        age_frames=age, coasting=coasting,
    )


# The canonical case: hull-only box nested in the lower part of a hull+mast
# box. IoU is only ~0.35 (below every NMS threshold) but IoS is ~1.0.
def _sailing_pair(hull_conf: float = 0.8, mast_conf: float = 0.7):
    hull = _target("boat", 11, x=100, y=160, w=80, h=40, conf=hull_conf, age=10)
    full = _target("sailboat", 12, x=100, y=80, w=80, h=120, conf=mast_conf, age=10)
    return hull, full


def test_hull_and_mast_boxes_collapse_to_one():
    d = TargetDeduper()
    hull, full = _sailing_pair()
    out = d.update([hull, full], seq=1)
    assert len(out) == 1
    # The fuller (hull+mast) box wins on area when age is equal.
    assert out[0].track_id == full.track_id


def test_partial_overlap_below_containment_still_merges():
    # Hull box wider than the sail box, so it is NOT >=80% contained (the old
    # filter missed this), but the intersection still covers most of it.
    d = TargetDeduper(vessel_ios=0.55, contained_frac=0.8)
    hull = _target("vessel", 11, x=90, y=160, w=100, h=40, age=5)
    full = _target("sailboat", 12, x=100, y=80, w=80, h=120, age=5)
    assert _drop_contained_targets([hull, full], 0.8) == [hull, full]  # old gap
    out = d.update([hull, full], seq=1)
    assert [t.track_id for t in out] == [full.track_id]


def test_cross_class_pair_merges():
    # Per-class NMS never compares boat vs sailboat; the deduper must.
    d = TargetDeduper()
    boat = _target("boat", 11, x=0, y=50, w=60, h=30, age=3)
    sail = _target("sailboat", 12, x=0, y=0, w=60, h=80, age=3)
    assert len(d.update([boat, sail], seq=1)) == 1


def test_sticky_winner_keeps_the_same_track_id():
    # Once track 12 wins, it keeps winning on later frames even when the loser
    # momentarily scores better (higher age/conf) — a flapping winner would
    # keep two chart blips alive downstream.
    d = TargetDeduper()
    hull, full = _sailing_pair()
    assert d.update([hull, full], seq=1)[0].track_id == 12
    older_hull = _target("boat", 11, x=100, y=160, w=80, h=40, conf=0.95, age=50)
    coasted_full = _target("sailboat", 12, x=100, y=80, w=80, h=120,
                           conf=0.4, age=10, coasting=True)
    out = d.update([older_hull, coasted_full], seq=2)
    assert [t.track_id for t in out] == [12]


def test_loser_surfaces_when_winner_track_dies():
    # The vessel itself can never be suppressed away: once the winning track is
    # gone, the remaining track is emitted.
    d = TargetDeduper()
    hull, full = _sailing_pair()
    d.update([hull, full], seq=1)
    out = d.update([hull], seq=2)
    assert [t.track_id for t in out] == [hull.track_id]


def test_pairing_expires_after_hold_frames():
    d = TargetDeduper(hold_frames=4)
    hull, full = _sailing_pair()
    d.update([hull, full], seq=1)
    assert d._pairs  # pairing recorded
    d.update([hull], seq=10)  # past hold window
    assert not d._pairs


def test_separating_vessels_split_again():
    # Two real vessels that once overlapped (crossing) must re-split once the
    # boxes separate, sticky pairing or not.
    d = TargetDeduper()
    a = _target("vessel", 11, x=0, y=0, w=60, h=40, age=10)
    b = _target("vessel", 12, x=10, y=0, w=60, h=40, age=20)
    assert len(d.update([a, b], seq=1)) == 1
    a2 = _target("vessel", 11, x=200, y=0, w=60, h=40, age=11)
    b2 = _target("vessel", 12, x=0, y=0, w=60, h=40, age=21)
    assert len(d.update([a2, b2], seq=2)) == 2


def test_kayak_overlapping_vessel_is_not_merged():
    # kayak is outside the vessel family: an occluding small craft is a real,
    # collision-relevant object. Only the conservative containment rule (>=0.8
    # nested) may drop it — a 60%-overlap kayak survives.
    d = TargetDeduper()
    kayak = _target("kayak", 11, x=0, y=20, w=30, h=20)
    vessel = _target("vessel", 12, x=18, y=0, w=100, h=60)
    out = d.update([kayak, vessel], seq=1)
    assert {t.track_id for t in out} == {11, 12}


def test_person_in_water_is_never_dropped():
    d = TargetDeduper()
    person = _target("person", 11, x=10, y=10, w=10, h=10, piw=True)
    vessel = _target("vessel", 12, x=0, y=0, w=100, h=60)
    out = d.update([person, vessel], seq=1)
    assert {t.track_id for t in out} == {11, 12}


def test_nested_buoy_on_deck_still_dropped_by_containment():
    # The pre-existing conservative rule is preserved: a box nested >=80%
    # inside a larger one is deck clutter and drops, regardless of family.
    d = TargetDeduper()
    buoy = _target("buoy", 11, x=20, y=20, w=10, h=10)
    vessel = _target("vessel", 12, x=0, y=0, w=100, h=60)
    out = d.update([buoy, vessel], seq=1)
    assert [t.track_id for t in out] == [12]


def test_vessel_ios_one_disables_family_merge():
    d = TargetDeduper(vessel_ios=1.0, contained_frac=0.8)
    hull = _target("vessel", 11, x=90, y=160, w=100, h=40)
    full = _target("sailboat", 12, x=100, y=80, w=80, h=120)
    out = d.update([hull, full], seq=1)
    assert len(out) == 2


def test_untracked_boxes_merge_without_sticky_state():
    d = TargetDeduper()
    hull, full = _sailing_pair()
    hull.track_id = None
    out = d.update([hull, full], seq=1)
    assert len(out) == 1
    assert not d._pairs


def test_three_way_pileup_collapses_to_one():
    # hull, hull+mast and a mid box all on one vessel -> single target.
    d = TargetDeduper()
    hull = _target("boat", 11, x=100, y=160, w=80, h=40, age=5)
    mid = _target("vessel", 13, x=100, y=120, w=80, h=80, age=5)
    full = _target("sailboat", 12, x=100, y=80, w=80, h=120, age=5)
    out = d.update([hull, mid, full], seq=1)
    assert [t.track_id for t in out] == [12]
