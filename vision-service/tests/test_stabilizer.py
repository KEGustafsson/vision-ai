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


def test_person_confirm_respects_a_higher_threshold():
    # If an operator raises person_confirm_frames, the person path honours it.
    s = TrackStabilizer(confirm_frames=5, person_confirm_frames=2)
    assert s.update([_track("person")], 1, 0.5) == []
    out = s.update([_track("person")], 2, 0.5)
    assert [t.label for t in out] == ["person"]


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
