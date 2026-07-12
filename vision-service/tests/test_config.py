import pytest

from app.config import DetectorConfig


def test_identity_retention_must_cover_reid_gap():
    # Retention shorter than the re-id gap would prune the waterline re-id's
    # candidate footprints before the gap closes — reject at config load.
    with pytest.raises(ValueError, match="track_memory_frames"):
        DetectorConfig(track_memory_frames=50, reid_max_gap_frames=120)


def test_default_identity_windows_are_consistent():
    d = DetectorConfig()
    assert d.track_memory_frames >= d.reid_max_gap_frames
