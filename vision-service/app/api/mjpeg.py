"""Annotated MJPEG stream at /stream/{camera}.mjpg (multipart/x-mixed-replace).

Serves the latest annotated JPEG produced by the pipeline at a steady cadence.
Multiple browser clients can connect; each pulls independently from the shared
latest-frame buffer so slow clients never stall inference.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

router = APIRouter()

_BOUNDARY = "frame"


@router.get("/stream/{camera}.mjpg")
async def stream(request: Request, camera: str):
    pipeline = request.app.state.pipeline
    if camera not in pipeline.workers:
        raise HTTPException(status_code=404, detail=f"unknown camera {camera}")
    fps = pipeline.settings.server.target_fps

    async def gen():
        # Poll a little faster than the configured cap so a freshly produced
        # frame is forwarded promptly, but only emit frames we haven't sent yet.
        # Output rate then equals the real production rate (~4-9 fps), not the
        # poll rate — no duplicate frames to saturate the client link and drift
        # the video behind real time.
        period = 1.0 / max(fps * 2.0, 1.0)
        last_seq = 0
        while True:
            if await request.is_disconnected():
                break
            last_seq, jpeg = pipeline.frames.get_if_new(camera, last_seq)
            if jpeg:
                yield (b"--" + _BOUNDARY.encode() + b"\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                       + jpeg + b"\r\n")
            await asyncio.sleep(period)

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
