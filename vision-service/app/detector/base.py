"""Detector abstraction. A detector consumes a frame and returns tracked
detections (boxes with stable track ids and pixel velocity)."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List, Optional

from ..camera.base import Frame


@dataclass
class RawTrack:
    track_id: Optional[int]
    cls: int
    label: str
    confidence: float
    x: float          # bbox top-left
    y: float
    w: float
    h: float
    vx: float = 0.0   # centroid px/frame
    vy: float = 0.0
    age_frames: int = 0

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


class Detector(abc.ABC):
    backend: str = "base"

    @abc.abstractmethod
    def detect_and_track(self, frame: Frame, stream: str = "default") -> List[RawTrack]:
        """Detect + track on a frame.

        ``stream`` names the camera the frame came from. A single detector is
        shared across all cameras (one model / CUDA context on the Jetson), so
        it must keep tracker state isolated per ``stream`` — otherwise frames
        from different cameras interleave through one tracker and corrupt the
        track IDs / motion model.
        """
        ...
