"""Hardware-accelerated RTSP capture on NVIDIA Jetson via GStreamer.

Uses ``nvv4l2decoder`` (HW H.264/H.265 decode) + ``nvvidconv`` and feeds frames
to OpenCV through an ``appsink``. Requires a GStreamer-enabled OpenCV build,
which ships in the Jetson base image.
"""

from __future__ import annotations

from typing import Optional

import cv2

from .base import Frame, FrameSource


def build_pipeline(url: str, codec: str = "h264", width: int = 1280, height: int = 720) -> str:
    depay = "rtph264depay ! h264parse" if codec == "h264" else "rtph265depay ! h265parse"
    return (
        f"rtspsrc location={url} latency=100 ! {depay} ! "
        f"nvv4l2decoder ! nvvidconv ! "
        f"video/x-raw,format=BGRx,width={width},height={height} ! "
        f"videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
    )


class RtspGstreamerSource(FrameSource):
    def __init__(self, name: str, url: str, codec: str = "h264",
                 width: int = 1280, height: int = 720):
        super().__init__(name)
        self._w = width
        self._h = height
        pipeline = build_pipeline(url, codec, width, height)
        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open GStreamer pipeline for {url}")

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
