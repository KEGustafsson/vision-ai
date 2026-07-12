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
    vt.prune(set(), seq=20)  # idle 20: past thin limit (16), below retention (260)
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
    # A much narrower same-label box on the same waterline (a small boat passing
    # in front of a vanished big one) is a DIFFERENT vessel: the hull's waterline
    # width is the invariant a partial/full flip preserves.
    vt = VelocityTracker()
    small = dict(x=150.0, y=370.0, w=90.0, h=30.0)  # inside HULL's footprint
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
        vt.prune({c}, seq, max_idle=60)  # explicit window: the subject here is
        # alias expiry, not the (much longer) default identity retention
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


# --- Identity retention across shadow-tracking dropouts ----------------------
# NvDCF holds a lost raw id in shadow for maxShadowTrackingAge (240) frames and
# re-acquires the vessel with the SAME raw id. The default retention window
# must cover that, or the reborn track finds its display id freed+quarantined
# and the same vessel reblips under a new identity.

def test_default_retention_survives_a_shadow_length_dropout():
    vt = VelocityTracker()
    for seq in range(4):
        vt.update(1, seq=seq, cx=0.0, cy=0.0)  # established track
    first = vt.display_id(1)
    for seq in range(4, 244):                  # unseen for 240 frames
        vt.prune(set(), seq)
    vt.update(1, seq=244, cx=0.0, cy=0.0)      # shadow re-acquisition
    assert vt.display_id(1) == first
    assert vt.stable_id(1) is not None


def test_retention_still_prunes_past_the_window():
    vt = VelocityTracker(max_idle=260)
    for seq in range(4):
        vt.update(1, seq=seq, cx=0.0, cy=0.0)
    vt.prune(set(), seq=300)  # idle 297 > 260: identity is released
    assert vt.display_id(1) is None


# --- Per-session stable serial (wire stable_id) ------------------------------

def test_stable_ids_are_monotonic_and_never_recycled():
    vt = VelocityTracker()
    vt.update(1, seq=0, cx=0.0, cy=0.0)
    vt.update(2, seq=0, cx=50.0, cy=0.0)
    s1, s2 = vt.stable_id(1), vt.stable_id(2)
    assert (s1, s2) == (1, 2)
    # Track 1 dies and its display id eventually returns to the pool, but its
    # serial must never be re-issued: the successor gets a fresh one.
    vt.prune({2}, seq=600, max_idle=10)
    assert vt.stable_id(1) is None
    vt.update(3, seq=601, cx=0.0, cy=0.0)
    assert vt.stable_id(3) == 3


def test_stable_id_survives_partial_full_reid_alias():
    # The alias resolves to the canonical id, so the serial rides through the
    # same waterline re-id that preserves the display id.
    vt = VelocityTracker()
    canon0, _ = _touch(vt, 1, seq=0, box=FULL)
    serial = vt.stable_id(canon0)
    canon1, _ = _touch(vt, 2, seq=1, box=HULL)
    assert canon1 == canon0
    assert vt.stable_id(canon1) == serial


def test_stable_id_none_for_unseen_raw_id():
    vt = VelocityTracker()
    assert vt.stable_id(7) is None


# --- Edge-clip relaxation of the re-id width gate ----------------------------
# A box clipped by the left/right frame edge has an unreliable width (observed
# live: a vessel exiting frame-right churned ids because each re-entry width
# failed the gate). With frame_w known, the width gate stands down for clipped
# boxes; the overlap/waterline gates still apply.

def test_reid_width_gate_relaxed_for_edge_clipped_box():
    vt = VelocityTracker(frame_w=1280.0)
    # Established vessel flush against the right edge, width clipped to 80.
    clipped = dict(x=1200.0, y=360.0, w=80.0, h=40.0)
    canon0, _ = _touch(vt, 1, seq=0, box=clipped)
    # It re-enters two frames later, more of the hull visible: width 160
    # (ratio 2.0 > reid_max_width_ratio 1.6 — the plain gate would reject).
    wider = dict(x=1120.0, y=360.0, w=160.0, h=40.0)
    canon1, _ = _touch(vt, 2, seq=2, box=wider)
    assert canon1 == canon0


def test_reid_width_gate_still_applies_away_from_edges():
    # Same width jump mid-frame is a DIFFERENT vessel and must not inherit.
    vt = VelocityTracker(frame_w=1280.0)
    narrow = dict(x=500.0, y=360.0, w=80.0, h=40.0)
    canon0, _ = _touch(vt, 1, seq=0, box=narrow)
    wide = dict(x=460.0, y=360.0, w=160.0, h=40.0)
    canon1, _ = _touch(vt, 2, seq=2, box=wide)
    assert canon1 != canon0


def test_reid_width_gate_not_relaxed_when_frame_width_unknown():
    vt = VelocityTracker()  # frame_w=None: relaxation disabled
    clipped = dict(x=1200.0, y=360.0, w=80.0, h=40.0)
    canon0, _ = _touch(vt, 1, seq=0, box=clipped)
    wider = dict(x=1120.0, y=360.0, w=160.0, h=40.0)
    canon1, _ = _touch(vt, 2, seq=2, box=wider)
    assert canon1 != canon0


# --- Alternation merge --------------------------------------------------------
# Live capture showed one vessel holding TWO live tracks that take turns being
# detected (pairs alternating 60-300 times while co-detected 0-5 frames): the
# birth-time re-id missed its one chance to alias them, and id/extent/range
# flapped between the two forever. _merge_alternating repairs the pair; a
# frame where BOTH are detected is proof of two real vessels and blocks it.

def _frame(vt, seq, detections):
    """One backend frame: resolve+update each (raw_id, box[, label]), then
    prune with the frame's active canonical set — exactly the per-frame call
    sequence of the pipelines."""
    active = set()
    for det in detections:
        raw, box = det[0], det[1]
        label = det[2] if len(det) > 2 else "vessel"
        canon = vt.resolve(raw, seq, box["x"], box["y"], box["w"], box["h"], label)
        vt.update(canon, seq, box["x"] + box["w"] / 2, box["y"] + box["h"])
        active.add(canon)
    vt.prune(active, seq)
    return active

# Born with its bottom half a hull-height off the waterline: the birth re-id's
# bottom gate rightly rejects it, so the vessel ends up with a second track.
BAD_BIRTH = dict(HULL, y=HULL["y"] - 0.5 * HULL["h"])


def test_alternating_tracks_on_one_footprint_merge():
    vt = VelocityTracker()
    for seq in range(4):                     # track 1 establishes (FULL extent)
        _frame(vt, seq, [(1, FULL)])
    disp1 = vt.display_id(1)
    serial1 = vt.stable_id(1)
    _frame(vt, 4, [(2, BAD_BIRTH)])          # second track born past the gates
    assert vt.resolve(2, 4, **{k: BAD_BIRTH[k] for k in "xywh"}, label="vessel") == 2
    for seq in range(5, 40):                 # then they take turns on the footprint
        raw, box = (1, FULL) if seq % 2 else (2, HULL)
        _frame(vt, seq, [(raw, box)])
    # Merged: raw 2 now resolves to track 1, which kept its display id/serial.
    canon = vt.resolve(2, 40, HULL["x"], HULL["y"], HULL["w"], HULL["h"], "vessel")
    assert canon == 1
    assert vt.display_id(1) == disp1
    assert vt.stable_id(1) == serial1
    assert vt.display_id(2) is None          # the younger track released its ids
    assert vt.stable_id(2) is None


def test_co_detected_tracks_never_merge():
    # Two real boats: one moored, one arriving alongside. Once both are
    # detected in the SAME frames SIDE BY SIDE (offset boxes: two distinct
    # vessels, not a nested duplicate), the pair is blocked from merging —
    # even through later frames where waves make them alternate detection.
    vt = VelocityTracker()
    far = dict(HULL, x=HULL["x"] + 400.0)
    for seq in range(4):                      # both establish, apart
        _frame(vt, seq, [(1, HULL), (2, far)])
    for seq in range(4, 14):                  # boat 2 slides alongside boat 1
        x = far["x"] - 30.0 * (seq - 3)
        _frame(vt, seq, [(1, HULL), (2, dict(HULL, x=x))])
    # Offset by ~35% of the width: clearly side by side (overlap 0.65 is past
    # the re-id gate but below the nested-duplicate threshold of 0.8).
    beside = dict(HULL, x=HULL["x"] + 70.0)
    for seq in range(14, 60):                 # waves: they alternate detection
        dets = [(1, HULL), (2, beside)] if seq % 5 == 0 else \
            [(1, HULL)] if seq % 2 else [(2, beside)]
        _frame(vt, seq, dets)
    assert vt.resolve(2, 60, beside["x"], beside["y"], beside["w"], beside["h"],
                      "vessel") == 2
    assert vt.display_id(1) != vt.display_id(2)


def test_nested_duplicate_co_detection_does_not_block_merge():
    # The live failure mode: ONE vessel double-boxed — the raw tracker holds a
    # hull track and a hull+mast track and detects BOTH in the same frame
    # every few frames (nested boxes survive NMS). Those duplicate-style
    # co-detections must not block the alternation merge.
    vt = VelocityTracker()
    for seq in range(4):
        _frame(vt, seq, [(1, FULL)])
    _frame(vt, 4, [(2, BAD_BIRTH)])           # hull track born past the gates
    for seq in range(5, 40):
        if seq % 4 == 0:                      # nested duplicate co-detection
            _frame(vt, seq, [(1, FULL), (2, HULL)])
        else:                                 # otherwise they take turns
            raw, box = (1, FULL) if seq % 2 else (2, HULL)
            _frame(vt, seq, [(raw, box)])
    canon = vt.resolve(2, 40, HULL["x"], HULL["y"], HULL["w"], HULL["h"], "vessel")
    assert canon == 1
    assert vt.display_id(2) is None


def test_one_sided_activity_never_merges():
    # A track squatting on a DEPARTED neighbour's footprint accumulates
    # evidence on its side only; without the partner taking detected turns,
    # the pair must not merge (the newcomer must not swallow the old identity).
    vt = VelocityTracker()
    for seq in range(4):                     # old track establishes, then goes dark
        _frame(vt, seq, [(1, HULL)])
    _frame(vt, 4, [(2, BAD_BIRTH)])          # newcomer born past the birth re-id
    for seq in range(5, 40):                 # only the newcomer is ever detected
        _frame(vt, seq, [(2, HULL)])
    assert vt.resolve(2, 40, HULL["x"], HULL["y"], HULL["w"], HULL["h"],
                      "vessel") == 2
    assert vt.display_id(1) is not None      # old identity intact (retention)


def test_person_tracks_never_merge_by_alternation():
    # Two swimmers seen alternately (waves) must stay two MOB targets.
    box_a = dict(x=100.0, y=380.0, w=30.0, h=20.0)
    vt = VelocityTracker()
    for seq in range(4):
        _frame(vt, seq, [(1, box_a, "person")])
    _frame(vt, 4, [(2, dict(box_a, y=box_a["y"] - 12.0), "person")])
    for seq in range(5, 40):
        raw = 1 if seq % 2 else 2
        _frame(vt, seq, [(raw, box_a, "person")])
    assert vt.resolve(2, 40, box_a["x"], box_a["y"], box_a["w"], box_a["h"],
                      "person") == 2
    assert vt.display_id(1) != vt.display_id(2)


def test_persistent_nested_duplicate_merges_without_alternation():
    # The second live failure mode: the hull track is detected EVERY frame —
    # co-detected nested under the hull+mast track whenever that one appears —
    # so it never takes a detected-while-partner-dark turn. The nested
    # co-detections themselves must carry the pair to a merge.
    vt = VelocityTracker()
    for seq in range(4):
        _frame(vt, seq, [(1, HULL)])
    _frame(vt, 4, [(2, dict(FULL, y=FULL["y"] - 0.5 * HULL["h"]))])  # born past gates
    for seq in range(5, 40):                  # hull always seen; mast box joins it
        dets = [(1, HULL), (2, FULL)] if seq % 2 else [(1, HULL)]
        _frame(vt, seq, dets)
    canon = vt.resolve(2, 40, FULL["x"], FULL["y"], FULL["w"], FULL["h"], "vessel")
    assert canon == 1
    assert vt.display_id(2) is None


# --- Merge dissolution (split) ------------------------------------------------
# A merge binds two raw ids to one canonical. If the pair later proves to be
# two real vessels — both detected in the SAME frame at DISJOINT footprints,
# repeatedly — the alias must dissolve so each vessel gets its own id again
# (observed live: a fused id flapped ~90 px between two separated boats).

def test_wrong_merge_dissolves_when_tracks_separate():
    vt = VelocityTracker()
    for seq in range(4):
        _frame(vt, seq, [(1, FULL)])
    _frame(vt, 4, [(2, BAD_BIRTH)])
    for seq in range(5, 40):                 # alternation earns the merge
        raw, box = (1, FULL) if seq % 2 else (2, HULL)
        _frame(vt, seq, [(raw, box)])
    assert vt.resolve(2, 40, HULL["x"], HULL["y"], HULL["w"], HULL["h"],
                      "vessel") == 1         # merged
    # ... but they were two boats: both now detected each frame, apart.
    apart = dict(HULL, x=HULL["x"] + 300.0)
    for seq in range(41, 60):
        _frame(vt, seq, [(1, HULL), (2, apart)])
    canon2 = vt.resolve(2, 60, apart["x"], apart["y"], apart["w"], apart["h"],
                        "vessel")
    assert canon2 == 2                       # alias dissolved: own track again
    assert vt.display_id(2) is not None
    assert vt.display_id(2) != vt.display_id(1)


def test_split_pair_is_blocked_from_immediate_remerge():
    vt = VelocityTracker()
    for seq in range(4):
        _frame(vt, seq, [(1, FULL)])
    _frame(vt, 4, [(2, BAD_BIRTH)])
    for seq in range(5, 40):
        raw, box = (1, FULL) if seq % 2 else (2, HULL)
        _frame(vt, seq, [(raw, box)])
    apart = dict(HULL, x=HULL["x"] + 300.0)
    for seq in range(41, 60):
        _frame(vt, seq, [(1, HULL), (2, apart)])
    assert vt.resolve(2, 60, apart["x"], apart["y"], apart["w"], apart["h"],
                      "vessel") == 2
    # Track 2 immediately slides back onto track 1's footprint: the fresh
    # co-block must keep the birth re-id AND the alternation merge from
    # re-binding the pair right away.
    for seq in range(61, 90):
        raw, box = (1, FULL) if seq % 2 else (2, HULL)
        _frame(vt, seq, [(raw, box)])
    assert vt.resolve(2, 90, HULL["x"], HULL["y"], HULL["w"], HULL["h"],
                      "vessel") == 2


def test_nested_same_frame_duplicates_never_split():
    # The routine hull-inside-full double box must keep its alias forever.
    vt = VelocityTracker()
    for seq in range(4):
        _frame(vt, seq, [(1, FULL)])
    canon, _ = _touch(vt, 2, 4, HULL)        # birth re-id aliases 2 -> 1
    assert canon == 1
    for seq in range(5, 80):                 # both boxes detected every frame
        _frame(vt, seq, [(1, FULL), (2, HULL)])
    assert vt.resolve(2, 80, HULL["x"], HULL["y"], HULL["w"], HULL["h"],
                      "vessel") == 1


def test_velocity_anchor_prefers_continuity_over_call_order():
    # Same-frame duplicate anchors ~100 px apart: whichever order the calls
    # arrive, the sample kept must be the one continuing the track's motion,
    # so a stationary vessel's velocity stays ~zero.
    vt = VelocityTracker()
    for seq in range(3):
        vt.update(1, seq, cx=200.0, cy=400.0)
    vt.update(1, 3, cx=300.0, cy=400.0)      # far duplicate arrives FIRST
    vx, vy, _ = vt.update(1, 3, cx=201.0, cy=400.0)  # true box second
    vx, vy, _ = vt.update(1, 4, cx=201.0, cy=400.0)
    assert abs(vx) < 1.0 and abs(vy) < 1.0


def test_wrong_merge_dissolves_when_tracks_stack_vertically():
    # Same-frame double resolution with heavy X-overlap but SEPARATED bottom
    # edges is two targets at different ranges, not a nested duplicate — the
    # split must fire on it just like on horizontally disjoint boxes.
    vt = VelocityTracker()
    for seq in range(4):
        _frame(vt, seq, [(1, FULL)])
    _frame(vt, 4, [(2, BAD_BIRTH)])
    for seq in range(5, 40):                 # alternation earns the merge
        raw, box = (1, FULL) if seq % 2 else (2, HULL)
        _frame(vt, seq, [(raw, box)])
    assert vt.resolve(2, 40, HULL["x"], HULL["y"], HULL["w"], HULL["h"],
                      "vessel") == 1
    stacked = dict(HULL, y=HULL["y"] - 150.0)  # same x-extent, waterline apart
    for seq in range(41, 60):
        _frame(vt, seq, [(1, HULL), (2, stacked)])
    assert vt.resolve(2, 60, stacked["x"], stacked["y"], stacked["w"],
                      stacked["h"], "vessel") == 2


def test_serial_counter_carries_across_tracker_replacement():
    # A supervised pipeline rebuild replaces the tracker; the replacement is
    # seeded with the predecessor's counter so stable ids never repeat within
    # one service session.
    old = VelocityTracker()
    for raw in (1, 2, 3):
        old.update(raw, seq=0, cx=0.0, cy=0.0)
    new = VelocityTracker(serial_start=old.next_serial)
    new.update(1, seq=0, cx=0.0, cy=0.0)
    assert new.stable_id(1) == 4
