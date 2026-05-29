"""Synthetic frame source — a procedurally animated sea scene with moving
"vessels", a "buoy", and an optional person-in-water, plus a drawn horizon.

Zero external assets, fully deterministic given a seed, so it doubles as the
ground-truth generator for the mock detector and for CI. Targets are drawn in
distinctive solid colours and also reported as :class:`TruthBox` so the mock
detector can return them without running a model.
"""

from __future__ import annotations

import math
from typing import List

import cv2
import numpy as np

from .base import Frame, FrameSource, TruthBox

# Canonical (class id, label, BGR colour) for synthetic actors.
VESSEL = (8, "vessel", (0, 200, 0))
BUOY = (80, "buoy", (0, 0, 255))
PERSON = (0, "person", (0, 140, 255))


class SyntheticSource(FrameSource):
    def __init__(self, name: str, width: int = 1280, height: int = 720,
                 horizon_frac: float = 0.45, with_mob: bool = False, fps: float = 10.0):
        super().__init__(name)
        self._w = width
        self._h = height
        self.horizon_y = int(height * horizon_frac)
        self.with_mob = with_mob
        self._fps = fps
        # Phase offset so forward/aft scenes differ.
        self._phase = 0.0 if name == "forward" else math.pi

    @property
    def width(self) -> int:
        return self._w

    @property
    def height(self) -> int:
        return self._h

    def _background(self) -> np.ndarray:
        img = np.empty((self._h, self._w, 3), dtype=np.uint8)
        img[: self.horizon_y] = (235, 206, 135)   # sky (light blue)
        img[self.horizon_y:] = (120, 70, 30)       # sea (dark blue)
        cv2.line(img, (0, self.horizon_y), (self._w, self.horizon_y), (180, 150, 90), 2)
        return img

    def read(self) -> Frame:
        seq = self._next_seq()
        t = seq / max(self._fps, 1e-6)
        img = self._background()
        truth: List[TruthBox] = []

        # A vessel tracking across the scene, just below the horizon.
        vx = (0.5 + 0.4 * math.sin(0.15 * t + self._phase)) * self._w
        vy = self.horizon_y + 18
        vw, vh = 120, 46
        self._draw(img, truth, VESSEL, vx, vy, vw, vh)

        # A near vessel, lower in frame (closer), moving the other way.
        nvx = (0.5 - 0.3 * math.sin(0.1 * t + self._phase)) * self._w
        nvy = self.horizon_y + int(0.35 * (self._h - self.horizon_y))
        self._draw(img, truth, VESSEL, nvx, nvy, 200, 80)

        # A buoy bobbing in place.
        bx = 0.2 * self._w
        by = self.horizon_y + 40 + 6 * math.sin(2.0 * t)
        self._draw(img, truth, BUOY, bx, by, 28, 36)

        # Optional person in the water (man-overboard scenario).
        if self.with_mob:
            px = 0.7 * self._w
            py = self.horizon_y + int(0.5 * (self._h - self.horizon_y))
            self._draw(img, truth, PERSON, px, py, 26, 30)

        return Frame(image=img, seq=seq, truth=truth)

    def _draw(self, img, truth, actor, cx, cy, w, h):
        cls, label, colour = actor
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, -1)
        truth.append(TruthBox(cls=cls, label=label, x=float(x1), y=float(y1),
                              w=float(w), h=float(h)))
