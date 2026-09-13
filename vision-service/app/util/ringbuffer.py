"""Thread-safe buffers that decouple the inference loop from API consumers.

``LatestFrame`` holds the most recent annotated JPEG per camera (for MJPEG).
``EventBuffer`` keeps a bounded history of recent detection events and notifies
async subscribers (WebSocket clients) when new events arrive.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional

# How long a one-off frame request (a /snapshot) keeps a camera's display path
# alive after it. Long enough that a client polling snapshots once a second
# always finds a fresh frame waiting, short enough that the pipeline goes back
# to spending the CPU on detection soon after the client stops.
DEMAND_TTL_S = 5.0


class LatestFrame:
    def __init__(self):
        self._lock = threading.Lock()
        self._frames: Dict[str, bytes] = {}
        # Async waiters (one per MJPEG client) woken the moment a frame for
        # their camera lands, so a stream forwards it immediately instead of
        # discovering it on the next poll tick.
        self._waiters: Dict[str, List[asyncio.Event]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Monotonic deadline per camera set by a one-off frame request, so a
        # /snapshot keeps the display path running briefly without a stream.
        self._demand_until: Dict[str, float] = {}
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

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the event loop that async waiters live on (see :meth:`subscribe`).
        Called once at startup, like :meth:`EventBuffer.bind_loop`."""
        self._loop = loop

    def set(self, camera: str, jpeg: bytes) -> None:
        with self._lock:
            if self._paused:
                return  # detection disabled: never publish a new frame
            self._frames[camera] = jpeg
            self._seq[camera] = self._seq.get(camera, 0) + 1
            waiters = list(self._waiters.get(camera, ()))
        # Wake this camera's streams from the producer thread. Frames are
        # produced on a worker/GStreamer thread, so hop to the loop thread.
        if waiters and self._loop is not None:
            self._loop.call_soon_threadsafe(self._wake, waiters)

    @staticmethod
    def _wake(waiters: List["asyncio.Event"]) -> None:
        for ev in waiters:
            ev.set()

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

    def note_demand(self, camera: str, ttl_s: float = DEMAND_TTL_S) -> None:
        """Record that someone wants frames for *camera* right now."""
        with self._lock:
            self._demand_until[camera] = time.monotonic() + ttl_s

    def wanted(self, camera: str) -> bool:
        """Whether anyone is waiting for this camera's annotated frames — a live
        stream client, or a recent one-off request. False lets the producer skip
        annotating and JPEG-encoding a frame nobody will look at."""
        with self._lock:
            if self._waiters.get(camera):
                return True
            return time.monotonic() < self._demand_until.get(camera, 0.0)

    async def wait_for_frame(self, camera: str, timeout: float) -> Optional[bytes]:
        """Wait for a frame for *camera*, or None if none arrives in time.
        Subscribing for the wait also marks demand, so a producer that had gone
        idle starts encoding again."""
        # Subscribe BEFORE looking, so a frame stored between the look and the
        # wait sets the event rather than being missed; and read the store again
        # after the wait, so a frame that landed as the timeout expired is
        # returned instead of a spurious "no frame".
        ev = self.subscribe(camera)
        try:
            jpeg = self.get(camera)
            if jpeg is not None:
                return jpeg
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(ev.wait(), timeout=timeout)
            return self.get(camera)
        finally:
            self.unsubscribe(camera, ev)

    def subscribe(self, camera: str) -> "asyncio.Event":
        """Register an :class:`asyncio.Event` set whenever a frame for *camera*
        is stored. Lets a stream wait for the next frame instead of polling —
        the frame reaches the client as soon as it exists, and an idle camera
        costs no wakeups. Always pair with :meth:`unsubscribe`."""
        ev = asyncio.Event()
        with self._lock:
            self._waiters.setdefault(camera, []).append(ev)
        return ev

    def unsubscribe(self, camera: str, ev: "asyncio.Event") -> None:
        with self._lock:
            waiters = self._waiters.get(camera)
            if not waiters:
                return
            if ev in waiters:
                waiters.remove(ev)
            if not waiters:
                self._waiters.pop(camera, None)


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
        if self._loop is None or not subs:
            return
        # One hop for the whole fan-out: call_soon_threadsafe writes to the
        # loop's self-pipe, so one call per subscriber per frame is one syscall
        # per subscriber per frame from the inference thread.
        self._loop.call_soon_threadsafe(self._fanout, subs, event)

    @staticmethod
    def _fanout(subs: List[asyncio.Queue], event: dict) -> None:
        for q in subs:
            EventBuffer._safe_put(q, event)

    @staticmethod
    def _safe_put(q: asyncio.Queue, event: dict) -> None:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            # A slow subscriber must not be fed a backlog of history while the
            # live scene moves on: shed its OLDEST queued event and keep this
            # one. Dropping the newest instead would hand a safety consumer
            # progressively staler detections the longer it lags.
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - raced empty
                pass
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - raced full
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
