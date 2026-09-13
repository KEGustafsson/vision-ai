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
        # A loop that only ever SENDS never learns that the client went away:
        # the disconnect is delivered as a receive message (or as a failed
        # send), so with no events flowing — detection disabled at a dock, or a
        # ?camera= filter matching nothing — a departed client would sit here
        # indefinitely, holding one of the max_ws_clients slots. Race the event
        # queue against the socket instead.
        receiver = asyncio.ensure_future(websocket.receive())
        getter = None
        try:
            while True:
                if getter is None:
                    getter = asyncio.ensure_future(queue.get())
                done, _ = await asyncio.wait(
                    {receiver, getter}, return_when=asyncio.FIRST_COMPLETED)
                if receiver in done:
                    message = receiver.result()
                    if message.get("type") == "websocket.disconnect":
                        break
                    # Clients aren't expected to send anything; ignore it and
                    # keep listening for the disconnect.
                    receiver = asyncio.ensure_future(websocket.receive())
                if getter in done:
                    ev = getter.result()
                    getter = None
                    if match(ev):
                        await websocket.send_json(ev)
        finally:
            receiver.cancel()
            if getter is not None:
                getter.cancel()
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:  # pragma: no cover
        pass
    finally:
        if queue is not None:
            pipeline.events.unsubscribe(queue)
        state.ws_clients = max(0, getattr(state, "ws_clients", 1) - 1)
