"""A control request from a superseded client must not be applied.

The SignalK plugin is stopped and restarted whenever its settings are saved. A
request composed before that restart can still be delivered afterwards, landing
behind the new instance's own push and quietly restoring the settings the
operator just changed. The plugin aborts what it can, but a request the
container has already accepted is beyond the client's reach — so the container
refuses one carrying an older ordering token.
"""

import time

from fastapi.testclient import TestClient

from app.api import rest
from app.config import load_settings
from app.main import create_app


def _app():
    return create_app(load_settings("mock"))


def _client():
    return TestClient(_app())


def _worker_confidences(app):
    return {w.confidence for w in app.state.pipeline.workers.values()}


def test_a_newer_client_supersedes_an_older_one():
    app = _app()
    with TestClient(app) as client:
        fresh = client.post("/control", json={"confidence": 0.5, "client_generation": 2000})
        assert fresh.status_code == 200
        assert fresh.json()["applied"]["confidence"] == 0.5

        # The previous plugin instance's request, delivered late.
        stale = client.post("/control", json={"confidence": 0.9, "client_generation": 1000})
        assert stale.status_code == 409
        assert "stale control request" in stale.json()["detail"]

        # ...and it changed nothing: the live threshold is still the new
        # instance's value, not the superseded one's.
        assert _worker_confidences(app) == {0.5}


def test_the_same_client_is_never_fenced_against_itself():
    # Every sync from one instance carries the same token; they must all apply.
    with _client() as client:
        for value in (0.4, 0.5, 0.6):
            r = client.post("/control", json={"confidence": value, "client_generation": 7})
            assert r.status_code == 200
            assert r.json()["applied"]["confidence"] == value


def test_a_request_without_a_token_still_applies():
    # An older plugin sends no token: it must keep working, and it must not
    # move the fence for the client that does send one.
    with _client() as client:
        assert client.post("/control", json={"client_generation": 500}).status_code == 200
        assert client.post("/control", json={"confidence": 0.42}).status_code == 200
        assert client.post(
            "/control", json={"confidence": 0.43, "client_generation": 500}).status_code == 200


def test_the_fence_expires_so_a_reset_clock_cannot_lock_a_client_out(monkeypatch):
    """A boat computer without an RTC boots at some earlier time, so a restarted
    plugin can stamp a LOWER token than the container remembers from before the
    reboot. The fence only has to cover the seconds around a restart, so it
    self-disarms once the client has been quiet — otherwise that plugin would be
    refused for the life of the container."""
    # Shorten the window rather than moving the clock: the camera workers read
    # the same monotonic clock from their own threads.
    monkeypatch.setattr(rest, "_CONTROL_FENCE_TTL_S", 0.05)
    with _client() as client:
        assert client.post("/control", json={"client_generation": 2_000_000}).status_code == 200
        # Immediately after, a lower token is a stale request.
        assert client.post("/control", json={"client_generation": 1_000}).status_code == 409

        # After the idle window, the same lower token is accepted: this is a
        # rebooted client, not an overtaken one.
        time.sleep(0.1)
        r = client.post("/control", json={"confidence": 0.33, "client_generation": 1_000})
        assert r.status_code == 200
        assert r.json()["applied"]["confidence"] == 0.33


def test_a_stale_request_is_refused_before_anything_is_applied():
    # The refusal has to come first: applying some fields and then rejecting
    # would leave the container in a state neither client asked for.
    with _client() as client:
        client.post("/control", json={"client_generation": 100, "enabled": True})
        stale = client.post(
            "/control",
            json={"client_generation": 50, "enabled": False, "max_targets": 3,
                  "labels": ["person"]},
        )
        assert stale.status_code == 409
        health = client.get("/health").json()
        assert health["detection_enabled"] is True
        assert health["max_targets"] != 3
