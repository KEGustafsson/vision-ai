"""Frame source abstraction. All camera backends expose the same interface so
the pipeline is agnostic to mode (Jetson RTSP, CPU RTSP, video file, synthetic)."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class TruthBox:
    """Ground-truth detection emitted by synthetic sources (mock mode only).

    Lets the mock detector return deterministic detections without a model.
    Real camera sources never populate this.
    """

    cls: int
    label: str
    x: float
    y: float
    w: float
    h: float
    confidence: float = 0.9


@dataclass
class Frame:
    image: np.ndarray                       # BGR HxWx3
    seq: int
    truth: List[TruthBox] = field(default_factory=list)


class FrameSource(abc.ABC):
    """A source of video frames for one camera."""

    def __init__(self, name: str):
        self.name = name
        self._seq = 0

    @property
    @abc.abstractmethod
    def width(self) -> int: ...

    @property
    @abc.abstractmethod
    def height(self) -> int: ...

    @abc.abstractmethod
    def read(self) -> Optional[Frame]:
        """Return the next frame, or None if the source is exhausted/unavailable."""

    def close(self) -> None:  # pragma: no cover - default no-op
        pass

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq
