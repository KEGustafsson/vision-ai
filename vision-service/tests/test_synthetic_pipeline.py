"""End-to-end mock pipeline: synthetic frames -> mock detector -> events,
plus an API smoke test via FastAPI's TestClient."""

import time

from fastapi.testclient import TestClient

from app.camera.synthetic import SyntheticSource
from app.config import load_settings
from app.detector.mock import MockDetector
from app.main import create_app


def test_synthetic_source_and_mock_detector_produce_tracks():
    src = SyntheticSource("forward", with_mob=True)
    det = MockDetector()
    labels = set()
    all_tracks = []
    for _ in range(5):
        frame = src.read()
        tracks = det.detect_and_track(frame)
        all_tracks.extend(tracks)
        for t in tracks:
            labels.add(t.label)
    # We expect vessels, a buoy and a person from the synthetic scene.
    assert "vessel" in labels
    assert "buoy" in labels
    assert "person" in labels
    # Track ids should be stable/assigned across all detections.
    assert all_tracks and all(t.track_id is not None for t in all_tracks)


def test_app_health_and_events():
    settings = load_settings("mock")
    app = create_app(settings)
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["mode"] == "mock"

        # Give the worker threads a moment to produce events.
        deadline = time.time() + 5
        events = []
        while time.time() < deadline and not events:
            events = client.get("/events/recent").json()
            time.sleep(0.2)
        assert events, "no detection events produced"
        assert events[-1]["camera"] in ("forward", "aft")
        assert "targets" in events[-1]
