"""WebSocket detection-event stream at /ws/events.

On connect the client receives the recent buffered events (late-join catch-up),
then a live push for every new event. An optional ?camera= filter restricts the
stream to one camera.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    await websocket.accept()
    pipeline = websocket.app.state.pipeline
    camera = websocket.query_params.get("camera")

    def match(ev: dict) -> bool:
        return camera is None or ev.get("camera") == camera

    # Late-join catch-up.
    for ev in pipeline.events.recent(20):
        if match(ev):
            await websocket.send_json(ev)

    queue = pipeline.events.subscribe()
    try:
        while True:
            ev = await queue.get()
            if match(ev):
                await websocket.send_json(ev)
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:  # pragma: no cover
        pass
    finally:
        pipeline.events.unsubscribe(queue)
