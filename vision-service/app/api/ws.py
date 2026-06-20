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

    # Cap concurrent subscribers (see ServerConfig.max_ws_clients). Touched only
    # from the event loop, so a plain counter on app.state needs no lock.
    state = websocket.app.state
    limit = pipeline.settings.server.max_ws_clients
    if getattr(state, "ws_clients", 0) >= limit:
        await websocket.close(code=1013)  # 1013 = "try again later"
        return
    state.ws_clients = getattr(state, "ws_clients", 0) + 1

    def match(ev: dict) -> bool:
        return camera is None or ev.get("camera") == camera

    queue = None
    try:
        # Late-join catch-up.
        for ev in pipeline.events.recent(20):
            if match(ev):
                await websocket.send_json(ev)

        queue = pipeline.events.subscribe()
        while True:
            ev = await queue.get()
            if match(ev):
                await websocket.send_json(ev)
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:  # pragma: no cover
        pass
    finally:
        if queue is not None:
            pipeline.events.unsubscribe(queue)
        state.ws_clients = max(0, getattr(state, "ws_clients", 1) - 1)
