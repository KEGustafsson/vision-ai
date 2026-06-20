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
        # Monotonic per-camera counter bumped on every set(). MJPEG clients use it
        # to send each frame at most once (the inference loop produces only
        # ~4-9 fps; re-sending the latest frame at the full poll rate would
        # triple the stream bandwidth and pile duplicate frames into the
        # client's socket buffer, drifting the video seconds behind real time).
        self._seq: Dict[str, int] = {}
        # When paused, set() is rejected. The pause flag and the frame store
        # share one lock so a worker's "am I still enabled?" check and its frame
        # write are atomic w.r.t. pause(): a frame encoded just before a disable
        # can never land after pause() has cleared the store.
        self._paused = False

    def set(self, camera: str, jpeg: bytes) -> None:
        with self._lock:
            if self._paused:
                return  # detection disabled: never publish a new frame
            self._frames[camera] = jpeg
            self._seq[camera] = self._seq.get(camera, 0) + 1

    def get(self, camera: str) -> Optional[bytes]:
        with self._lock:
            return self._frames.get(camera)

    def get_if_new(self, camera: str, last_seq: int) -> tuple[int, Optional[bytes]]:
        """Return ``(seq, jpeg)`` only when a frame newer than *last_seq* exists;
        otherwise ``(last_seq, None)``. Lets a stream skip unchanged frames so its
        output rate tracks the real production rate instead of the poll rate."""
        with self._lock:
            seq = self._seq.get(camera, 0)
            if seq == last_seq:
                return last_seq, None
            return seq, self._frames.get(camera)

    def clear(self, camera: str) -> None:
        """Drop the retained frame for a camera so MJPEG/snapshot stop serving a
        stale image once that camera is paused."""
        with self._lock:
            self._frames.pop(camera, None)

    def pause(self) -> None:
        """Atomically stop accepting frames and drop every retained one. Used by
        the master detection-off toggle so no in-flight frame can resurface."""
        with self._lock:
            self._paused = True
            self._frames.clear()

    def resume(self) -> None:
        """Re-allow frame writes (master detection-on)."""
        with self._lock:
            self._paused = False


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
        # Guard n=0 (``[-0:]`` is ``[0:]`` and would dump the whole buffer) and
        # negative n (would slice from the wrong end).
        if n <= 0:
            return []
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
