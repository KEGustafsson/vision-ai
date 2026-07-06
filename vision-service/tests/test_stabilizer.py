"""Track stabilizer: MOB-critical person tracks confirm faster than the generic
appearance debounce, so a person in the water isn't held back for extra frames."""

from dataclasses import replace

from app.detector.base import RawTrack
from app.detector.stabilizer import TrackStabilizer, cap_targets_sticky


def _track(label: str, tid: int = 1, conf: float = 0.9) -> RawTrack:
    return RawTrack(track_id=tid, cls=0, label=label, confidence=conf,
                    x=0.0, y=0.0, w=10.0, h=10.0)


def test_person_confirmed_on_first_frame():
    s = TrackStabilizer(confirm_frames=3, person_confirm_frames=1)
    out = s.update([_track("person")], seq=1, conf_on=0.5)
    # MOB-critical: emitted immediately, not after confirm_frames.
    assert [t.label for t in out] == ["person"]


def test_non_person_debounced_until_confirm_frames():
    s = TrackStabilizer(confirm_frames=3, person_confirm_frames=1)
    assert s.update([_track("vessel")], 1, 0.5) == []
    assert s.update([_track("vessel")], 2, 0.5) == []
    out = s.update([_track("vessel")], 3, 0.5)
    assert [t.label for t in out] == ["vessel"]


def test_person_confirm_respects_a_higher_threshold():
    # If an operator raises person_confirm_frames, the person path honours it.
    s = TrackStabilizer(confirm_frames=5, person_confirm_frames=2)
    assert s.update([_track("person")], 1, 0.5) == []
    out = s.update([_track("person")], 2, 0.5)
    assert [t.label for t in out] == ["person"]


def _box(x: float, y: float = 50.0, w: float = 40.0, h: float = 20.0,
         tid: int = 1, conf: float = 0.9) -> RawTrack:
    return RawTrack(track_id=tid, cls=8, label="vessel", confidence=conf,
                    x=x, y=y, w=w, h=h)


def test_smoothing_calms_a_stationary_jittery_box():
    # A station-keeping vessel whose detected box trembles ±3 px per frame must
    # emit a near-still box (this is the "calm view" behaviour).
    s = TrackStabilizer(confirm_frames=1)
    xs = []
    for seq in range(1, 40):
        jitter = 3.0 if seq % 2 else -3.0
        out = s.update([_box(100.0 + jitter)], seq, 0.5)
        xs.append(out[0].x)
    raw_step = 6.0  # the input moves 6 px every frame
    steps = [abs(a - b) for a, b in zip(xs[21:], xs[20:])]
    assert max(steps) < raw_step / 4  # at least 4x calmer than the input


def test_smoothing_follows_a_moving_box_with_window_lag_only():
    # Rolling average = follow the detections. A mover's box trails by about
    # half the window (no prediction, no overshoot) and keeps its true size.
    s = TrackStabilizer(confirm_frames=1)
    lead_lag, width = 0.0, 0.0
    for seq in range(1, 40):
        raw_x = 100.0 + 10.0 * seq  # 10 px/frame crossing target
        out = s.update([_box(raw_x)], seq, 0.5)
        lead_lag = (raw_x + 40.0) - (out[0].x + out[0].w)
        width = out[0].w
    assert 0.0 <= lead_lag <= 25.0  # ~(window-1)/2 frames behind, never ahead
    assert abs(width - 40.0) < 1e-6  # no smear, no inflation


def test_size_flips_are_smoothed_not_snapped():
    # A vessel detected alternately as hull-only and hull+mast flips its raw box
    # height by 230 px frame-to-frame under ONE id (re-id keeps the id; the shape
    # still flaps). Box SIZE is NOT gated (only the WATERLINE is), so instead of
    # freezing the hull box and then SNAPPING to the mast extent, the rolling
    # average absorbs the flip: the shown height settles to a stable intermediate
    # value and never swings the full raw range — a smooth box, not a jumpy one.
    s = TrackStabilizer(confirm_frames=1, smooth_window=10)
    hs = []
    for seq in range(1, 40):
        h = 40.0 if seq % 2 else 270.0  # hull only <-> hull + mast
        out = s.update([_box(100.0, y=300.0 - h, h=h)], seq, 0.5)
        hs.append(out[0].h)
    settled = hs[12:]  # after the smoothing window has filled
    assert max(settled) - min(settled) < 1.0     # stable, no frame-to-frame snap
    assert 40.0 < min(settled) and max(settled) < 270.0  # never at a raw extreme


def test_jump_gate_rejects_a_single_frame_teleport():
    # A detector glitch throws the box across the frame for one frame. Real
    # targets don't teleport: the spike must not appear in the output at all.
    s = TrackStabilizer(confirm_frames=1)
    xs = []
    for seq in range(1, 30):
        raw_x = 500.0 if seq == 15 else 100.0  # one-frame leap
        out = s.update([_box(raw_x)], seq, 0.5)
        xs.append(out[0].x)
    assert max(xs) < 101.0  # the teleport frame emitted the held box


def test_jump_gate_accepts_a_persistent_relocation():
    # If the "implausible" position persists, it is real (the detector was
    # wrong BEFORE, or re-seated on the true object): after jump_confirm
    # consecutive frames the box follows to the new place and stays.
    s = TrackStabilizer(confirm_frames=1, jump_confirm=3)
    xs = []
    for seq in range(1, 30):
        raw_x = 100.0 if seq < 15 else 500.0  # permanent move at seq 15
        out = s.update([_box(raw_x)], seq, 0.5)
        xs.append(out[0].x)
    assert xs[13] == 100.0   # still held on the first out-of-gate frame
    assert xs[-1] == 500.0   # settled at the real new position
    assert max(x for x in xs) <= 500.0  # never overshoots (no prediction)


def test_smoothing_keeps_the_waterline_steady_across_height_flips():
    # Partial <-> full extent flips (hull only <-> hull + mast) share one id via
    # re-id; whether a flip is averaged in (small) or gate-rejected (large),
    # a steady detected bottom edge (waterline) stays exactly steady.
    s = TrackStabilizer(confirm_frames=1)
    for seq in range(1, 30):
        h = 40.0 if seq % 2 else 150.0
        out = s.update([_box(100.0, y=400.0 - h, h=h)], seq, 0.5)
        bottom = out[0].y + out[0].h
        assert abs(bottom - 400.0) < 1e-6


def test_smoothing_can_be_disabled():
    s = TrackStabilizer(confirm_frames=1, smooth=False)
    for seq in range(1, 10):
        jitter = 3.0 if seq % 2 else -3.0
        out = s.update([_box(100.0 + jitter)], seq, 0.5)
        assert out[0].x == 100.0 + jitter  # raw box passes through untouched


def test_low_confidence_outlier_moves_the_smoothed_box_less():
    # Confidence-weighted smoothing (the NSA-Kalman idea): the same displaced
    # in-gate box must perturb the shown average less when the detector was
    # unsure about it than when it was confident.
    def shift_after_outlier(conf):
        s = TrackStabilizer(confirm_frames=1)
        for seq in range(1, 6):
            s.update([_box(100.0)], seq, 0.5)
        out = s.update([_box(112.0, conf=conf)], 6, 0.5)
        return out[0].x - 100.0

    weak, strong = shift_after_outlier(0.1), shift_after_outlier(0.9)
    assert 0.0 < weak < strong


def test_conf_weighting_can_be_disabled():
    # With conf_weight off the window is a plain unweighted average again:
    # the outlier's confidence no longer matters.
    def shift_after_outlier(conf):
        s = TrackStabilizer(confirm_frames=1, conf_weight=False)
        for seq in range(1, 6):
            s.update([_box(100.0)], seq, 0.5)
        out = s.update([_box(112.0, conf=conf)], 6, 0.5)
        return out[0].x - 100.0

    assert shift_after_outlier(0.1) == shift_after_outlier(0.9)


def _coast_alive(s, tid, last_seen, until):
    """Frames past last_seen the track kept being emitted (coasted)."""
    alive = 0
    for seq in range(last_seen + 1, until):
        out = s.update([], seq, 0.5)
        if any(t.track_id == tid for t in out):
            alive = seq - last_seen
    return alive


def test_locked_track_coasts_longer_than_a_young_one():
    # Track lock: an established track (>= lock_hits fresh detections) rides
    # out a dropout twice as long as a young one before being dropped.
    young = TrackStabilizer(confirm_frames=1, max_coast_frames=4,
                            lock_hits=10, lock_coast_factor=2.0)
    for seq in range(1, 4):                      # 3 hits: not locked
        young.update([_box(100.0)], seq, 0.5)
    assert _coast_alive(young, 1, last_seen=3, until=20) == 4

    locked = TrackStabilizer(confirm_frames=1, max_coast_frames=4,
                             lock_hits=10, lock_coast_factor=2.0)
    for seq in range(1, 13):                     # 12 hits: locked
        locked.update([_box(100.0)], seq, 0.5)
    assert _coast_alive(locked, 1, last_seen=12, until=30) == 8


def test_track_lock_can_be_disabled():
    s = TrackStabilizer(confirm_frames=1, max_coast_frames=4,
                        lock_hits=0, lock_coast_factor=2.0)
    for seq in range(1, 13):
        s.update([_box(100.0)], seq, 0.5)
    assert _coast_alive(s, 1, last_seen=12, until=30) == 4


def test_sticky_cap_keeps_incumbents_across_confidence_noise():
    a = _box(0.0, tid=1, conf=0.50)
    b = _box(200.0, tid=2, conf=0.49)
    out = cap_targets_sticky([a, b], 1, set(), 0.05)
    prev = {t.track_id for t in out}
    assert prev == {1}
    # Next frame the challenger noses ahead WITHIN the margin: incumbent stays,
    # so the pair doesn't swap the slot (and blink) on every confidence wobble.
    out = cap_targets_sticky(
        [replace(a, confidence=0.48), replace(b, confidence=0.50)], 1, prev, 0.05)
    assert [t.track_id for t in out] == [1]
    # A challenger clearly stronger than incumbent + margin takes the slot.
    out = cap_targets_sticky(
        [replace(a, confidence=0.48), replace(b, confidence=0.60)], 1, prev, 0.05)
    assert [t.track_id for t in out] == [2]


def test_sticky_cap_never_squeezes_out_a_person():
    person = RawTrack(track_id=9, cls=0, label="person", confidence=0.2,
                      x=0.0, y=0.0, w=5.0, h=5.0)
    vessels = [_box(50.0 * i, tid=i, conf=0.9) for i in range(1, 4)]
    out = cap_targets_sticky(vessels + [person], 2, set(), 0.05)
    assert any(t.label == "person" for t in out)


def test_same_id_duplicates_in_one_frame_keep_the_stronger():
    # Waterline re-id can put the same display id on a partial AND a full
    # detection of one vessel in a single frame; only the stronger must emit.
    s = TrackStabilizer(confirm_frames=1)
    weak = _track("vessel", tid=7, conf=0.6)
    strong = _track("vessel", tid=7, conf=0.9)
    out = s.update([weak, strong], seq=1, conf_on=0.5)
    assert len(out) == 1
    assert out[0].confidence == 0.9
    # Order must not matter.
    s2 = TrackStabilizer(confirm_frames=1)
    out2 = s2.update([strong, weak], seq=1, conf_on=0.5)
    assert len(out2) == 1
    assert out2[0].confidence == 0.9
