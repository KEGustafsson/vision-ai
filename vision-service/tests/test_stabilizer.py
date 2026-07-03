"""Track stabilizer: MOB-critical person tracks confirm faster than the generic
appearance debounce, so a person in the water isn't held back for extra frames."""

from app.detector.base import RawTrack
from app.detector.stabilizer import TrackStabilizer


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


def test_bbox_smoothing_damps_hull_mast_flicker():
    # A sailing vessel's box alternates hull-only (short) / hull+mast (tall).
    # The emitted box must move gradually between the two, and its bottom edge
    # (the waterline, shared by both) must stay put so ranging is unaffected.
    s = TrackStabilizer(confirm_frames=1, bbox_ema_alpha=0.5)
    hull = RawTrack(track_id=1, cls=8, label="vessel", confidence=0.9,
                    x=100.0, y=160.0, w=80.0, h=40.0)
    tall = RawTrack(track_id=1, cls=8, label="vessel", confidence=0.9,
                    x=100.0, y=80.0, w=80.0, h=120.0)
    s.update([hull], 1, 0.5)
    out = s.update([tall], 2, 0.5)[0]
    assert out.y == 120.0 and out.h == 80.0          # halfway, not a jump
    assert out.y + out.h == 200.0                     # waterline preserved
    out = s.update([hull], 3, 0.5)[0]
    assert out.h == 60.0                              # eased back, no flicker
    assert out.y + out.h == 200.0


def test_bbox_smoothing_disabled_at_alpha_one():
    s = TrackStabilizer(confirm_frames=1, bbox_ema_alpha=1.0)
    hull = RawTrack(track_id=1, cls=8, label="vessel", confidence=0.9,
                    x=100.0, y=160.0, w=80.0, h=40.0)
    tall = RawTrack(track_id=1, cls=8, label="vessel", confidence=0.9,
                    x=100.0, y=80.0, w=80.0, h=120.0)
    s.update([hull], 1, 0.5)
    out = s.update([tall], 2, 0.5)[0]
    assert (out.y, out.h) == (80.0, 120.0)            # raw box passes through


def test_person_confirm_respects_a_higher_threshold():
    # If an operator raises person_confirm_frames, the person path honours it.
    s = TrackStabilizer(confirm_frames=5, person_confirm_frames=2)
    assert s.update([_track("person")], 1, 0.5) == []
    out = s.update([_track("person")], 2, 0.5)
    assert [t.label for t in out] == ["person"]
