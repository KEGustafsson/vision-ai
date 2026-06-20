"""Hardware-accelerated RTSP capture on NVIDIA Jetson via GStreamer.

Uses ``nvv4l2decoder`` (HW H.264/H.265 decode) + ``nvvidconv`` and feeds frames
to OpenCV through an ``appsink``. Requires a GStreamer-enabled OpenCV build,
which ships in the Jetson base image.
"""

from __future__ import annotations

from typing import Optional

import cv2

from .base import Frame, FrameSource, redact_url


def _validate_url(url: str) -> None:
    if not url.startswith(("rtsp://", "rtspt://")):
        raise ValueError(f"RTSP url must start with rtsp://: {redact_url(url)!r}")
    # Reject characters that would break out of the GStreamer pipeline string.
    if any(c in url for c in " \t\n!'\"\\"):
        raise ValueError(f"RTSP url contains unsafe characters: {redact_url(url)!r}")


def build_pipeline(url: str, codec: str = "h264",
                   width: Optional[int] = None, height: Optional[int] = None) -> str:
    _validate_url(url)
    if codec not in ("h264", "h265"):
        raise ValueError(f"unsupported codec: {codec!r}")
    depay = "rtph264depay ! h264parse" if codec == "h264" else "rtph265depay ! h265parse"
    # protocols=tcp forces RTP-over-RTSP (interleaved) instead of separate UDP
    # streams: UDP RTP can't traverse the Docker bridge NAT back to the
    # container, so the default (UDP-first) pipeline never prerolls. TCP uses the
    # single outbound RTSP connection and works through NAT.
    #
    # width/height are left unset by default so nvvidconv passes the camera's
    # native resolution through unscaled — forcing a fixed size (e.g. 1280x720)
    # would distort a 4:3 (1280x960) sensor into 16:9 and corrupt the vertical
    # geometry (horizon depression / range) which assumes square pixels.
    scale = f",width={width},height={height}" if width and height else ""
    return (
        f"rtspsrc location={url} protocols=tcp latency=50 drop-on-latency=true ! {depay} ! "
        f"nvv4l2decoder ! nvvidconv ! "
        f"video/x-raw,format=BGRx{scale} ! "
        f"videoconvert ! video/x-raw,format=BGR ! appsink sync=false drop=true max-buffers=1"
    )


class RtspGstreamerSource(FrameSource):
    def __init__(self, name: str, url: str, codec: str = "h264",
                 width: Optional[int] = None, height: Optional[int] = None):
        super().__init__(name)
        # Native resolution by default; populated from the first decoded frame.
        self._w = width or 0
        self._h = height or 0
        pipeline = build_pipeline(url, codec, width, height)
        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open GStreamer pipeline for {redact_url(url)}")

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
        if not self._w:
            self._h, self._w = img.shape[:2]
        return Frame(image=img, seq=self._next_seq())

    def close(self) -> None:
        self._cap.release()
