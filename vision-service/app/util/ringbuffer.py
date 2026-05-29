"""Thread-safe buffers that decouple the inference loop from API consumers.

``LatestFrame`` holds the most recent annotated JPEG per camera (for MJPEG).
``EventBuffer`` keeps a bounded history of recent detection events and notifies
async subscribers (WebSocket clients) when new events arrive.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from typing import Deque, Dict, List, Optional


class LatestFrame:
    def __init__(self):
        self._lock = threading.Lock()
        self._frames: Dict[str, bytes] = {}

    def set(self, camera: str, jpeg: bytes) -> None:
        with self._lock:
            self._frames[camera] = jpeg

    def get(self, camera: str) -> Optional[bytes]:
        with self._lock:
            return self._frames.get(camera)


class EventBuffer:
    def __init__(self, maxlen: int = 200):
        self._lock = threading.Lock()
        self._events: Deque[dict] = deque(maxlen=maxlen)
        self._subscribers: List[asyncio.Queue] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, event: dict) -> None:
        """Called from the inference thread; fans out to async subscribers."""
        with self._lock:
            self._events.append(event)
            subs = list(self._subscribers)
        if self._loop is None:
            return
        for q in subs:
            self._loop.call_soon_threadsafe(self._safe_put, q, event)

    @staticmethod
    def _safe_put(q: asyncio.Queue, event: dict) -> None:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass

    def recent(self, n: int = 20) -> List[dict]:
        with self._lock:
            return list(self._events)[-n:]

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)
