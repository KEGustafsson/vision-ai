"""A departed WebSocket client must free its subscriber slot.

The stream only ever sends, so a disconnect reaches the handler as a receive
message (or a failed send) — never as a queue event. A client that leaves while
its filter matches nothing (or while detection is off, so nothing is published)
would otherwise stay parked forever, holding one of max_ws_clients.
"""

import time

from fastapi.testclient import TestClient

from app.config import load_settings
from app.main import create_app


def _wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def test_client_slot_is_released_when_a_filtered_client_disconnects():
    settings = load_settings("mock")
    app = create_app(settings)
    with TestClient(app) as client:
        # A filter that never matches: events keep arriving on the subscription
        # but nothing is ever sent to this client, so only the receive side can
        # notice it left.
        with client.websocket_connect("/ws/events?camera=nosuch") as ws:
            assert _wait_for(lambda: getattr(app.state, "ws_clients", 0) == 1)
            assert ws is not None
        assert _wait_for(lambda: getattr(app.state, "ws_clients", 0) == 0)


def test_events_still_stream_to_a_matching_client():
    settings = load_settings("mock")
    app = create_app(settings)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/events?camera=forward") as ws:
            event = ws.receive_json()
            assert event["camera"] == "forward"
            assert "targets" in event
        assert _wait_for(lambda: getattr(app.state, "ws_clients", 0) == 0)
