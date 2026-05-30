"""Plain RTSP capture via OpenCV (software decode). Fallback for non-Jetson
hosts or cameras where HW decode is unavailable.

Reconnects automatically: RTSP feeds drop (network glitch, camera reboot, a run
of undecodable H.264 frames) and OpenCV's VideoCapture does not recover on its
own — once read() fails it fails forever. So on read failure we re-open the
capture (throttled) instead of leaving the camera dead until a restart.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import cv2

from .base import Frame, FrameSource

# Don't hammer a down camera: wait this long between reconnect attempts.
_REOPEN_INTERVAL_S = 3.0


class RtspCpuSource(FrameSource):
    def __init__(self, name: str, url: str):
        super().__init__(name)
        self._url = url
        # Fail fast on unreachable cameras instead of stalling the worker.
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|timeout;5000000"
        )
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_reopen = 0.0
        cap = self._open_capture()
        if cap is None:
            raise RuntimeError(f"cannot open RTSP stream: {url}")
        self._cap = cap
        self._w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        self._h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

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

    @property
    def width(self) -> int:
        return self._w

    @property
    def height(self) -> int:
        return self._h

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            self._reconnect()
            return None
        ok, img = self._cap.read()
        if not ok:
            # Stream dropped or a frame failed to decode — try to re-establish
            # it; the worker keeps polling and recovers once frames flow again.
            self._reconnect()
            return None
        return Frame(image=img, seq=self._next_seq())

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
