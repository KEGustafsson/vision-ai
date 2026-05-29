"""Loop a local video file as a pseudo-camera (mock/dev mode)."""

from __future__ import annotations

from typing import Optional

import cv2

from .base import Frame, FrameSource


class VideoFileSource(FrameSource):
    def __init__(self, name: str, path: str):
        super().__init__(name)
        self._path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video file: {path}")
        self._w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def width(self) -> int:
        return self._w

    @property
    def height(self) -> int:
        return self._h

    def read(self) -> Optional[Frame]:
        ok, img = self._cap.read()
        if not ok:
            # Loop back to the start.
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, img = self._cap.read()
            if not ok:
                return None
        return Frame(image=img, seq=self._next_seq())

    def close(self) -> None:
        self._cap.release()
