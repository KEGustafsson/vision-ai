"""Annotated MJPEG stream at /stream/{camera}.mjpg (multipart/x-mixed-replace).

Serves the latest annotated JPEG produced by the pipeline at a steady cadence.
Multiple browser clients can connect; each pulls independently from the shared
latest-frame buffer so slow clients never stall inference.
"""

from __future__ import annotations

import asyncio
import contextlib

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

router = APIRouter()

_BOUNDARY = "frame"


@router.get("/stream/{camera}.mjpg")
async def stream(request: Request, camera: str):
    pipeline = request.app.state.pipeline
    if camera not in pipeline.workers:
        raise HTTPException(status_code=404, detail=f"unknown camera {camera}")
    # Liveness floor for the wait below: at least one wakeup per frame period
    # (and at least one a second) so a stalled camera or a vanished client is
    # noticed promptly without polling a healthy stream.
    fps = pipeline.settings.server.target_fps
    idle_timeout = min(1.0, 1.0 / max(fps, 1.0))

    # Cap concurrent stream clients so a peer can't exhaust the encode/poll budget
    # by opening unbounded MJPEG connections. The counter lives on app.state and
    # is only touched from the (single-threaded) event loop, so no lock is needed.
    state = request.app.state
    limit = pipeline.settings.server.max_stream_clients
    if getattr(state, "mjpeg_clients", 0) >= limit:
        raise HTTPException(status_code=503, detail="too many stream clients")
    state.mjpeg_clients = getattr(state, "mjpeg_clients", 0) + 1

    async def gen():
        # Wait to be woken by the producer instead of polling: each frame is
        # forwarded the moment it is stored (no half-a-poll-interval of added
        # display latency, and an idle camera costs no wakeups), and only
        # frames we haven't sent yet go out. Output rate then equals the real
        # production rate (~4-9 fps) — no duplicate frames to saturate the
        # client link and drift the video behind real time. The bounded wait is
        # a liveness floor: it re-checks a disconnected client, and covers a
        # producer that never binds a loop (e.g. a plain TestClient run).
        frames = pipeline.frames
        new_frame = frames.subscribe(camera)
        last_seq = 0
        try:
            while True:
                if await request.is_disconnected():
                    break
                # Clear BEFORE reading so a frame stored between the read and
                # the wait leaves the event set and is picked up immediately.
                new_frame.clear()
                last_seq, jpeg = frames.get_if_new(camera, last_seq)
                if jpeg:
                    # A transport error on the write surfaces in Starlette's
                    # response task, not here (this generator only sees
                    # GeneratorExit), and the disconnect check above already
                    # ends the stream — so no handler around the yield.
                    yield (b"--" + _BOUNDARY.encode() + b"\r\n"
                           b"Content-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                           + jpeg + b"\r\n")
                    continue
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(new_frame.wait(), timeout=idle_timeout)
        finally:
            frames.unsubscribe(camera, new_frame)
            state.mjpeg_clients = max(0, getattr(state, "mjpeg_clients", 1) - 1)

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
