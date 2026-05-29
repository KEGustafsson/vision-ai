"""Plain RTSP capture via OpenCV (software decode). Fallback for non-Jetson
hosts or cameras where HW decode is unavailable."""

from __future__ import annotations

import os
from typing import Optional

import cv2

from .base import Frame, FrameSource


class RtspCpuSource(FrameSource):
    def __init__(self, name: str, url: str):
        super().__init__(name)
        self._url = url
        # Fail fast on unreachable cameras instead of stalling the worker.
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|timeout;5000000"
        )
        self._cap = cv2.VideoCapture(url)
        # Keep latency low: small internal buffer.
        for prop, val in ((cv2.CAP_PROP_BUFFERSIZE, 1),
                          (getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", -1), 5000),
                          (getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", -1), 5000)):
            if prop != -1:
                try:
                    self._cap.set(prop, val)
                except Exception:
                    pass
        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(f"cannot open RTSP stream: {url}")
        self._w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        self._h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    @property
    def width(self) -> int:
        return self._w

    @property
    def height(self) -> int:
        return self._h

    def read(self) -> Optional[Frame]:
        ok, img = self._cap.read()
        if not ok:
            return None
        return Frame(image=img, seq=self._next_seq())

    def close(self) -> None:
        self._cap.release()
