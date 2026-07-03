import pytest

from app.detector.tracker import VelocityTracker


def test_display_ids_land_in_range():
    vt = VelocityTracker()
    for raw in (1000, 5, 999999):
        vt.update(raw, seq=0, cx=0.0, cy=0.0)
        assert 10 <= vt.display_id(raw) <= 99


def test_display_id_stable_across_frames():
    vt = VelocityTracker()
    vt.update(42, seq=0, cx=0.0, cy=0.0)
    first = vt.display_id(42)
    for seq in range(1, 5):
        vt.update(42, seq=seq, cx=float(seq), cy=0.0)
        assert vt.display_id(42) == first


def test_display_id_none_for_unseen_raw_id():
    vt = VelocityTracker()
    assert vt.display_id(7) is None


def test_distinct_raw_ids_get_distinct_display_ids():
    vt = VelocityTracker()
    for raw in range(1, 6):
        vt.update(raw, seq=0, cx=0.0, cy=0.0)
    disps = {vt.display_id(raw) for raw in range(1, 6)}
    assert len(disps) == 5


def test_recycled_id_is_reused_last_rotation():
    # Tiny range so the pool is easy to reason about.
    vt = VelocityTracker(id_min=10, id_max=12)
    for raw in (1, 2, 3):
        vt.update(raw, seq=0, cx=0.0, cy=0.0)
    assert vt.display_id(1) == 10  # allocated front-to-back

    # raw 1 goes idle and is pruned; its id (10) returns to the *back* of pool.
    for seq in range(1, 5):
        vt.update(2, seq=seq, cx=0.0, cy=0.0)
        vt.update(3, seq=seq, cx=0.0, cy=0.0)
        vt.prune({2, 3}, seq=seq, max_idle=2)
    assert vt.display_id(1) is None  # pruned

    # New track does NOT immediately reuse 10 while other ids are free... but
    # here the pool only held 10, so it does come back — rotation guarantees it
    # is handed out only after everything ahead of it in the queue.
    vt.update(4, seq=5, cx=0.0, cy=0.0)
    assert vt.display_id(4) == 10


def test_rotation_prefers_unused_ids_before_recycled():
    vt = VelocityTracker(id_min=10, id_max=13)  # pool: 10,11,12,13
    vt.update(1, seq=0, cx=0.0, cy=0.0)  # -> 10
    vt.update(2, seq=0, cx=0.0, cy=0.0)  # -> 11
    # Drop raw 1 (id 10 recycled to back: queue is now 12,13,10).
    for seq in range(1, 5):
        vt.update(2, seq=seq, cx=0.0, cy=0.0)
        vt.prune({2}, seq=seq, max_idle=2)
    # Next two new tracks should get the still-unused 12 and 13 before 10.
    vt.update(3, seq=5, cx=0.0, cy=0.0)
    vt.update(4, seq=5, cx=0.0, cy=0.0)
    assert vt.display_id(3) == 12
    assert vt.display_id(4) == 13
    # Only now is the recycled 10 handed out.
    vt.update(5, seq=5, cx=0.0, cy=0.0)
    assert vt.display_id(5) == 10


def test_pool_exhaustion_falls_back_within_range():
    vt = VelocityTracker(id_min=10, id_max=12)  # only 3 slots
    for raw in range(1, 10):  # request more than the pool holds
        vt.update(raw, seq=0, cx=0.0, cy=0.0)
        assert 10 <= vt.display_id(raw) <= 12


def test_separate_instances_are_independent_per_stream():
    a = VelocityTracker()
    b = VelocityTracker()
    a.update(1, seq=0, cx=0.0, cy=0.0)
    b.update(1, seq=0, cx=0.0, cy=0.0)
    # Both start fresh, so both hand out the first id of the range.
    assert a.display_id(1) == b.display_id(1) == 10


def test_invalid_range_rejected():
    with pytest.raises(ValueError):
        VelocityTracker(id_min=99, id_max=10)


def test_set_id_range_bounds_new_ids():
    vt = VelocityTracker(id_min=10, id_max=99)
    vt.set_id_range(10, 17)  # follow max_targets=8
    for raw in range(1, 40):
        vt.update(raw, seq=0, cx=0.0, cy=0.0)
        assert 10 <= vt.display_id(raw) <= 17  # always within the new range


def test_set_id_range_remaps_out_of_range_live_ids():
    vt = VelocityTracker(id_min=10, id_max=99)
    vt.update(1, seq=0, cx=0.0, cy=0.0)  # gets 10
    vt.set_id_range(20, 23)  # 10 is now out of range
    # The live track is remapped into range immediately — no higher id lingers.
    assert 20 <= vt.display_id(1) <= 23


def test_set_id_range_keeps_in_range_live_ids():
    vt = VelocityTracker(id_min=10, id_max=99)
    vt.update(1, seq=0, cx=0.0, cy=0.0)  # gets 10
    vt.set_id_range(10, 17)  # 10 still in range
    assert vt.display_id(1) == 10  # unchanged — no needless reshuffle


def test_set_id_range_all_live_ids_bounded_after_shrink():
    vt = VelocityTracker(id_min=10, id_max=99)
    for raw in range(1, 30):  # spread across the wide range
        vt.update(raw, seq=0, cx=0.0, cy=0.0)
    vt.set_id_range(10, 17)  # shrink to max_targets=8
    for raw in range(1, 30):
        assert 10 <= vt.display_id(raw) <= 17  # EVERY live id now in range


def test_set_id_range_invalid_rejected():
    vt = VelocityTracker()
    with pytest.raises(ValueError):
        vt.set_id_range(30, 10)


# --- Waterline re-identification -------------------------------------------
# The backend tracker mints a NEW raw id when a vessel's detected box flips
# between partial (hull only) and full (hull + mast) extents — the IoU jump
# breaks its association. resolve() must alias the new raw id back onto the
# known target (same waterline footprint) so the display id doesn't flicker.

# A vessel at x=100..300 with its waterline at y=400.
HULL = dict(x=100.0, y=360.0, w=200.0, h=40.0)     # hull only: short box
FULL = dict(x=100.0, y=250.0, w=200.0, h=150.0)    # hull + mast: tall box

def _touch(vt, raw_id, seq, box, label="vessel"):
    """Drive one detection through the resolve->update->display_id sequence
    exactly like the backends do; returns (canonical_id, display_id)."""
    canon = vt.resolve(raw_id, seq, box["x"], box["y"], box["w"], box["h"], label)
    vt.update(canon, seq, box["x"] + box["w"] / 2, box["y"] + box["h"])
    return canon, vt.display_id(canon)


def test_reid_keeps_one_id_across_partial_full_alternation():
    vt = VelocityTracker()
    _, disp = _touch(vt, 1, seq=0, box=FULL)
    # The detector alternates hull-only / full boxes under two raw ids.
    for seq in range(1, 8):
        raw = 1 if seq % 2 else 2
        box = FULL if seq % 2 else HULL
        _, d = _touch(vt, raw, seq, box)
        assert d == disp  # same vessel, same detection number


def test_reid_alias_is_sticky_and_ages_continuously():
    vt = VelocityTracker()
    canon0, _ = _touch(vt, 1, seq=0, box=FULL)
    canon1, _ = _touch(vt, 2, seq=1, box=HULL)
    assert canon1 == canon0
    # Once aliased, the raw id resolves to the canonical without re-matching.
    canon2, _ = _touch(vt, 2, seq=2, box=HULL)
    assert canon2 == canon0
    # Age keeps counting from the FIRST sighting of the target, either raw id.
    _, vy, age = vt.update(canon0, 3, HULL["x"] + HULL["w"] / 2, HULL["y"] + HULL["h"])
    assert age == 3


def test_reid_velocity_stays_clean_across_shape_flips():
    # The waterline (bottom-center) anchor is identical for HULL and FULL boxes,
    # so alternating shapes must NOT produce a spurious vertical velocity.
    vt = VelocityTracker()
    _touch(vt, 1, seq=0, box=FULL)
    for seq in range(1, 6):
        raw = 1 if seq % 2 else 2
        box = FULL if seq % 2 else HULL
        canon = vt.resolve(raw, seq, box["x"], box["y"], box["w"], box["h"], "vessel")
        vx, vy, _ = vt.update(canon, seq, box["x"] + box["w"] / 2, box["y"] + box["h"])
        assert vx == 0.0 and vy == 0.0  # stationary vessel reads as stationary


def test_reid_requires_same_label():
    vt = VelocityTracker()
    canon0, _ = _touch(vt, 1, seq=0, box=FULL, label="vessel")
    canon1, _ = _touch(vt, 2, seq=1, box=HULL, label="buoy")
    assert canon1 != canon0


def test_reid_never_fuses_person_tracks():
    # Two people in the water near each other must stay two MOB targets.
    box_a = dict(x=100.0, y=380.0, w=30.0, h=20.0)
    box_b = dict(x=110.0, y=380.0, w=30.0, h=20.0)  # overlapping footprint
    vt = VelocityTracker()
    canon0, _ = _touch(vt, 1, seq=0, box=box_a, label="person")
    canon1, _ = _touch(vt, 2, seq=1, box=box_b, label="person")
    assert canon1 != canon0


def test_reid_rejects_disjoint_horizontal_footprint():
    vt = VelocityTracker()
    far = dict(HULL, x=HULL["x"] + 500.0)  # same waterline, elsewhere in frame
    canon0, _ = _touch(vt, 1, seq=0, box=HULL)
    canon1, _ = _touch(vt, 2, seq=1, box=far)
    assert canon1 != canon0


def test_reid_rejects_misaligned_waterline():
    vt = VelocityTracker()
    # Same horizontal extent but floating well above the hull's bottom edge
    # (e.g. superstructure detected separately): NOT the same waterline.
    floating = dict(x=100.0, y=200.0, w=200.0, h=40.0)
    canon0, _ = _touch(vt, 1, seq=0, box=HULL)
    canon1, _ = _touch(vt, 2, seq=1, box=floating)
    assert canon1 != canon0


def test_reid_window_expires():
    vt = VelocityTracker(reid_max_gap=5)
    canon0, _ = _touch(vt, 1, seq=0, box=FULL)
    # A matching box appearing long after the gap window is a NEW target.
    canon1, _ = _touch(vt, 2, seq=10, box=HULL)
    assert canon1 != canon0


def test_reid_can_be_disabled():
    vt = VelocityTracker(reid=False)
    _, disp0 = _touch(vt, 1, seq=0, box=FULL)
    canon1, disp1 = _touch(vt, 2, seq=1, box=HULL)
    assert canon1 == 2 and disp1 != disp0


def test_reid_co_occurring_partial_and_full_share_the_display_id():
    # Both boxes in the SAME frame (nested duplicate detection).
    vt = VelocityTracker()
    _, disp0 = _touch(vt, 1, seq=0, box=FULL)
    _, disp1 = _touch(vt, 2, seq=0, box=HULL)
    assert disp1 == disp0


def test_reid_stale_aliases_expire_while_canonical_lives_on():
    # A vessel that flickers for a long session mints a new raw id per flip;
    # each alias must expire once its raw id stops being emitted, or the alias
    # map grows one immortal entry per flip for the canonical track's lifetime.
    vt = VelocityTracker()
    canon, _ = _touch(vt, 1, seq=0, box=FULL)
    for seq in range(1, 200):
        raw = 1 if seq % 2 else 100 + seq  # a fresh raw id on every "hull" flip
        c, _ = _touch(vt, raw, seq, box=FULL if seq % 2 else HULL)
        assert c == canon
        vt.prune({c}, seq)
    # Only recently-resolved aliases survive; the map stays bounded by the
    # prune window instead of growing with every flip.
    assert len(vt._alias) <= 60


def test_reid_alias_dies_with_pruned_track():
    vt = VelocityTracker()
    canon0, _ = _touch(vt, 1, seq=0, box=FULL)
    canon1, _ = _touch(vt, 2, seq=1, box=HULL)
    assert canon1 == canon0
    # Target gone: prune everything well past max_idle.
    vt.prune(set(), seq=100, max_idle=10)
    assert vt.display_id(canon0) is None
    # The backend resurrects raw id 2 much later for something new at the same
    # spot: it must start fresh (the old identity/alias state is gone).
    canon2, disp2 = _touch(vt, 2, seq=101, box=HULL)
    assert canon2 == 2 and disp2 is not None
