"""Plain RTSP capture via OpenCV (software decode). Fallback for non-Jetson
hosts or cameras where HW decode is unavailable.

Reconnects automatically: RTSP feeds drop (network glitch, camera reboot, a run
of undecodable H.264 frames) and OpenCV's VideoCapture does not recover on its
own — once read() fails it fails forever. So on read failure we re-open the
capture (throttled) instead of leaving the camera dead until a restart.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import cv2

from .base import Frame, FrameSource, redact_url

# Don't hammer a down camera: wait this long between reconnect attempts.
_REOPEN_INTERVAL_S = 3.0
_READ_WAIT_S = 1.0
_FFMPEG_LOW_LATENCY_OPTIONS = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "max_delay;0|"
    "reorder_queue_size;0|"
    "timeout;5000000"
)


class RtspCpuSource(FrameSource):
    def __init__(self, name: str, url: str):
        super().__init__(name)
        self._url = url
        # Fail fast on unreachable cameras and bias FFmpeg toward live frames
        # instead of preserving a deep RTSP jitter buffer.
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", _FFMPEG_LOW_LATENCY_OPTIONS)
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_reopen = 0.0
        self._closed = False
        self._lock = threading.Lock()
        self._frame_ready = threading.Condition(self._lock)
        self._latest_img = None
        self._latest_seq = 0
        self._last_delivered_seq = 0
        cap = self._open_capture()
        if cap is None:
            raise RuntimeError(f"cannot open RTSP stream: {redact_url(url)}")
        self._cap = cap
        self._w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        self._h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        self._reader = threading.Thread(
            target=self._reader_loop, name=f"rtsp-cpu-{name}", daemon=True)
        self._reader.start()

    def _open_capture(self) -> Optional["cv2.VideoCapture"]:
        cap = cv2.VideoCapture(self._url)
        # Keep latency low: small internal buffer.
        for prop, val in ((cv2.CAP_PROP_BUFFERSIZE, 1),
                          (getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", -1), 5000),
                          (getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", -1), 5000)):
            if prop != -1:
                try:
                    cap.set(prop, val)
                except Exception:
                    pass
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _reconnect(self) -> None:
        """Throttled re-open after a read failure. Returns immediately if a
        previous attempt was too recent so we don't spin on a down camera."""
        now = time.monotonic()
        if now - self._last_reopen < _REOPEN_INTERVAL_S:
            return
        self._last_reopen = now
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._cap = self._open_capture()  # may be None; retried next interval

    def _reader_loop(self) -> None:
        """Continuously drain FFmpeg so slow inference never sees stale frames."""
        while not self._closed:
            if self._cap is None:
                self._reconnect()
                time.sleep(0.05)
                continue

            ok, img = self._cap.read()
            if not ok:
                self._reconnect()
                time.sleep(0.05)
                continue

            seq = self._next_seq()
            with self._frame_ready:
                self._latest_img = img
                self._latest_seq = seq
                self._frame_ready.notify_all()

    @property
    def width(self) -> int:
        return self._w

    @property
    def height(self) -> int:
        return self._h

    def read(self) -> Optional[Frame]:
        deadline = time.monotonic() + _READ_WAIT_S
        with self._frame_ready:
            while (
                not self._closed
                and self._latest_seq == self._last_delivered_seq
                and time.monotonic() < deadline
            ):
                self._frame_ready.wait(timeout=max(0.0, deadline - time.monotonic()))
            if self._closed or self._latest_img is None:
                return None
            self._last_delivered_seq = self._latest_seq
            # Copy under the lock: the reader thread rebinds (and OpenCV may reuse
            # the underlying buffer for) ``_latest_img`` on the next frame, so the
            # consumer must not share the live buffer it could overwrite mid-encode.
            return Frame(image=self._latest_img.copy(), seq=self._latest_seq)

    def close(self) -> None:
        self._closed = True
        with self._frame_ready:
            self._frame_ready.notify_all()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if threading.current_thread() is not self._reader:
            self._reader.join(timeout=1.0)
