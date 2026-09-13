"""Frame delivery: wake-on-frame instead of polling, and the display path only
runs while somebody is actually watching.

Polling cost up to half a frame period of display latency per client and a
wakeup per poll tick even with nothing to send; annotating + JPEG-encoding a
frame nobody looks at costs the vessel's CPU the same either way.
"""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from app.config import load_settings
from app.main import create_app
from app.util.ringbuffer import LatestFrame


def run(coro):
    """Run a coroutine on a fresh loop (the test deps carry no asyncio plugin)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_subscriber_is_woken_by_a_frame_from_a_producer_thread():
    async def scenario():
        frames = LatestFrame()
        frames.bind_loop(asyncio.get_running_loop())
        waiter = frames.subscribe("forward")
        # The producer is another thread (camera worker / GStreamer thread).
        await asyncio.get_running_loop().run_in_executor(
            None, frames.set, "forward", b"jpeg-bytes")
        # Already signalled: no polling, no sleep.
        await asyncio.wait_for(waiter.wait(), timeout=1.0)
        assert frames.get_if_new("forward", 0) == (1, b"jpeg-bytes")

    run(scenario())


def test_waiter_is_not_woken_by_another_camera():
    async def scenario():
        frames = LatestFrame()
        frames.bind_loop(asyncio.get_running_loop())
        waiter = frames.subscribe("forward")
        frames.set("aft", b"aft-frame")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(waiter.wait(), timeout=0.05)

    run(scenario())


def test_unsubscribe_drops_the_waiter_so_a_gone_client_is_never_woken():
    async def scenario():
        frames = LatestFrame()
        frames.bind_loop(asyncio.get_running_loop())
        waiter = frames.subscribe("forward")
        frames.unsubscribe("forward", waiter)
        assert frames.wanted("forward") is False
        frames.set("forward", b"x")
        assert not waiter.is_set()

    run(scenario())


def test_paused_store_wakes_nobody():
    # Detection off: no frame is stored, so no client may be woken to re-send.
    async def scenario():
        frames = LatestFrame()
        frames.bind_loop(asyncio.get_running_loop())
        waiter = frames.subscribe("forward")
        frames.pause()
        frames.set("forward", b"should-be-dropped")
        assert not waiter.is_set()
        assert frames.get("forward") is None

    run(scenario())


def test_frames_are_wanted_only_while_someone_is_watching():
    frames = LatestFrame()
    assert frames.wanted("forward") is False        # nobody watching
    waiter = frames.subscribe("forward")
    assert frames.wanted("forward") is True         # a live stream client
    frames.unsubscribe("forward", waiter)
    assert frames.wanted("forward") is False
    frames.note_demand("forward", ttl_s=5.0)        # a one-off /snapshot
    assert frames.wanted("forward") is True
    frames.note_demand("forward", ttl_s=0.0)        # ...which expires
    assert frames.wanted("forward") is False
    assert frames.wanted("aft") is False            # demand is per camera


def test_wait_for_frame_returns_none_when_no_frame_arrives():
    async def scenario():
        frames = LatestFrame()
        frames.bind_loop(asyncio.get_running_loop())
        assert await frames.wait_for_frame("forward", timeout=0.05) is None
        # The waiter it used must not be left behind.
        assert frames.wanted("forward") is False

    run(scenario())


def test_snapshot_waits_for_a_frame_instead_of_404ing():
    """End-to-end: the pipeline encodes nothing until asked, and a snapshot
    request is what asks — so the first request must still return an image."""
    settings = load_settings("mock")
    app = create_app(settings)
    with TestClient(app) as client:
        r = client.get("/snapshot/forward")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "image/jpeg"
        assert r.content[:2] == b"\xff\xd8"  # JPEG SOI


def test_display_path_goes_idle_and_drops_its_frame_when_nobody_watches():
    settings = load_settings("mock")
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/snapshot/forward").status_code == 200
        pipeline = app.state.pipeline
        # Demand expires; the worker then stops encoding and clears the frame it
        # would otherwise keep serving as if it were current.
        pipeline.frames.note_demand("forward", ttl_s=0.0)
        deadline = time.time() + 5
        while time.time() < deadline and pipeline.frames.get("forward") is not None:
            time.sleep(0.05)
        assert pipeline.frames.get("forward") is None
        # Detection itself never stopped — only the display path idled.
        assert client.get("/events/recent?n=1").json()


def test_wait_for_frame_returns_a_frame_stored_before_the_wait_began():
    """snapshot() looks once, then waits. A frame stored between those two
    steps sets no event the waiter can see, so without re-reading the store the
    2 s wait would end in a spurious 404 for a camera that is producing fine."""
    async def scenario():
        frames = LatestFrame()
        frames.bind_loop(asyncio.get_running_loop())
        # Landing before any waiter exists is exactly the race: no wake-up.
        frames.set("forward", b"already-here")
        assert await frames.wait_for_frame("forward", timeout=0.05) == b"already-here"
        assert frames.wanted("forward") is False  # waiter cleaned up

    run(scenario())


def test_snapshot_404s_immediately_for_an_unknown_camera():
    settings = load_settings("mock")
    app = create_app(settings)
    with TestClient(app) as client:
        started = time.monotonic()
        r = client.get("/snapshot/nosuch")
        elapsed = time.monotonic() - started
        assert r.status_code == 404
        # Waiting out the snapshot timeout for a camera that can never produce
        # would be a free way to tie up the event loop.
        assert elapsed < 1.0
        # ...and the unknown name left no demand entry behind.
        assert app.state.pipeline.frames.wanted("nosuch") is False
