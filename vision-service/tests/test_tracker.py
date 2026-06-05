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
