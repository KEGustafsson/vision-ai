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


def test_lowest_free_id_is_allocated_first():
    # Numbers stay small and familiar: allocation always takes the lowest free
    # id, regardless of how large the raw tracker ids grow.
    vt = VelocityTracker()
    for raw in (1001, 52, 999999):
        vt.update(raw, seq=0, cx=0.0, cy=0.0)
    assert [vt.display_id(r) for r in (1001, 52, 999999)] == [10, 11, 12]


def test_recycled_id_is_quarantined_before_reuse():
    vt = VelocityTracker(id_min=10, id_max=12)
    vt.update(1, seq=0, cx=0.0, cy=0.0)  # -> 10
    vt.prune(set(), seq=20)              # single-sighting track prunes fast
    assert vt.display_id(1) is None
    # A new track soon after must NOT be handed 10 (an operator may still
    # associate that number with the old vessel): next-lowest instead.
    vt.update(2, seq=21, cx=0.0, cy=0.0)
    assert vt.display_id(2) == 11
    # Long after the quarantine, 10 is simply the lowest free id again.
    vt.update(3, seq=400, cx=0.0, cy=0.0)
    assert vt.display_id(3) == 10


def test_all_free_ids_quarantined_reuses_longest_freed():
    vt = VelocityTracker(id_min=10, id_max=11)
    vt.update(1, seq=0, cx=0.0, cy=0.0)   # -> 10
    vt.prune(set(), seq=20)               # 10 freed (quarantined)
    vt.update(2, seq=21, cx=0.0, cy=0.0)  # -> 11
    # Only 10 is free and it is quarantined; heavy churn must still get an id
    # (the longest-freed one) rather than colliding with the live 11.
    vt.update(3, seq=22, cx=0.0, cy=0.0)
    assert vt.display_id(3) == 10


def test_thin_tracks_release_their_ids_quickly():
    # A single-sighting glint holds a display id only briefly; an established
    # track keeps its id through the same idle gap.
    vt = VelocityTracker()
    vt.update(1, seq=0, cx=0.0, cy=0.0)               # glint: one sighting
    for seq in range(3):
        vt.update(2, seq=seq, cx=0.0, cy=0.0)         # established: 3 hits
    vt.prune(set(), seq=20)  # idle 20: past thin limit (16), below max (60)
    assert vt.display_id(1) is None
    assert vt.display_id(2) == 11


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


def test_reid_rejects_width_mismatch():
    # A same-label box far narrower than the stored footprint on the same
    # waterline (a small boat passing where a vanished big one was) is a
    # DIFFERENT vessel: the hull's waterline width is the invariant a
    # partial/full flip preserves. reid_max_width_ratio defaults to 3.0 (it
    # must pass hull vs hull+sails, ~2.8x); this box is 200/60 = 3.3x narrower,
    # past the gate.
    vt = VelocityTracker()
    small = dict(x=150.0, y=370.0, w=60.0, h=30.0)  # inside HULL's footprint
    canon0, _ = _touch(vt, 1, seq=0, box=HULL)
    canon1, _ = _touch(vt, 2, seq=1, box=small)
    assert canon1 != canon0


def test_reid_reacquires_a_moving_vessel_at_its_predicted_position():
    # A vessel crossing at 30 px/frame drops out for 5 frames. Its stored
    # footprint is advanced by its known velocity, so the re-detection where the
    # vessel actually IS matches even though it no longer overlaps the old spot.
    vt = VelocityTracker()
    moving = dict(x=100.0, y=360.0, w=100.0, h=40.0)
    canon0 = None
    for seq in range(5):
        canon0, _ = _touch(vt, 1, seq, dict(moving, x=100.0 + 30.0 * seq))
    ahead = dict(moving, x=100.0 + 30.0 * 9)  # where it should be at seq 9
    canon1, _ = _touch(vt, 2, seq=9, box=ahead)
    assert canon1 == canon0


def test_reid_does_not_give_a_movers_id_to_a_newcomer_at_its_old_spot():
    # Same moving vessel as above — but the box appearing after the gap sits at
    # the mover's OLD position. Without motion prediction it would inherit the
    # id; with it, the footprint has moved on and the newcomer stays distinct.
    vt = VelocityTracker()
    moving = dict(x=100.0, y=360.0, w=100.0, h=40.0)
    canon0 = None
    for seq in range(5):
        canon0, _ = _touch(vt, 1, seq, dict(moving, x=100.0 + 30.0 * seq))
    old_spot = dict(moving, x=100.0 + 30.0 * 4)  # where it was LAST seen
    canon1, _ = _touch(vt, 2, seq=9, box=old_spot)
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


def test_reid_buffered_gate_reacquires_after_long_dropout():
    # C-BIoU-style buffered matching: a stationary vessel drops out for 20
    # frames and re-appears shifted by 60% of its width (bobbing / prediction
    # error accumulated while unseen). Raw footprint overlap is 0.4 < 0.5, so
    # the strict gate alone would mint a new id; the buffer widens the
    # matching space with the gap and the vessel keeps its id.
    vt = VelocityTracker()
    box = dict(x=100.0, y=360.0, w=200.0, h=40.0)
    canon0 = None
    for seq in range(3):
        canon0, _ = _touch(vt, 1, seq, box)
    shifted = dict(box, x=220.0)
    canon1, _ = _touch(vt, 2, seq=22, box=shifted)
    assert canon1 == canon0


def test_reid_buffer_stays_tight_for_short_gaps():
    # The same 60%-of-width shift ONE frame later is not a plausible flip of
    # the same hull: the buffer is proportional to the gap, so a fresh
    # candidate is still judged (almost) as strictly as before.
    vt = VelocityTracker()
    box = dict(x=100.0, y=360.0, w=200.0, h=40.0)
    canon0 = None
    for seq in range(3):
        canon0, _ = _touch(vt, 1, seq, box)
    shifted = dict(box, x=220.0)
    canon1, _ = _touch(vt, 2, seq=3, box=shifted)
    assert canon1 != canon0


def test_reid_direction_gate_rejects_a_candidate_behind_a_mover():
    # OC-SORT-style momentum: a vessel crossing at 8 px/frame disappears; a
    # same-width box then appears clearly BEHIND it (against its motion).
    # The buffered overlap against the predicted footprint would accept it —
    # the direction-consistency gate must not.
    def run(min_speed):
        vt = VelocityTracker(reid_dir_min_speed=min_speed)
        moving = dict(x=100.0, y=360.0, w=200.0, h=40.0)
        canon0 = None
        for seq in range(5):
            canon0, _ = _touch(vt, 1, seq, dict(moving, x=100.0 + 8.0 * seq))
        behind = dict(moving, x=100.0 + 8.0 * 4 - 80.0)
        canon1, _ = _touch(vt, 2, seq=8, box=behind)
        return canon0, canon1

    canon0, canon1 = run(min_speed=2.0)
    assert canon1 != canon0
    # Sanity: with the gate disabled the geometry alone WOULD hand the id
    # over — proving the gate (not the overlap check) is what rejected it.
    canon0, canon1 = run(min_speed=0.0)
    assert canon1 == canon0


def test_reid_direction_gate_still_allows_in_place_shape_flips():
    # A partial/full flip has near-zero displacement; the direction gate must
    # never break the bread-and-butter re-id case, even on a moving vessel.
    vt = VelocityTracker()
    canon0 = None
    for seq in range(5):
        canon0, _ = _touch(vt, 1, seq, dict(FULL, x=FULL["x"] + 8.0 * seq))
    hull_next = dict(HULL, x=HULL["x"] + 8.0 * 5)
    canon1, _ = _touch(vt, 2, seq=5, box=hull_next)
    assert canon1 == canon0


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
