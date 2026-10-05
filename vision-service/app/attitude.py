"""Latest boat attitude, as forwarded by the SignalK plugin (POST /attitude).

The container knows nothing about SignalK; the plugin reads
``navigation.attitude`` (smoothed per its own settings) and pushes it here. The
camera workers read it per frame to compensate the horizon for pitch/roll (see
``app/geometry/attitude.py``). Ages are measured on the container's monotonic
clock from receipt, so the two hosts' wall clocks never need to agree.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple


class AttitudeStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sample: Optional[Tuple[float, float]] = None  # (pitch, roll) rad
        self._at = 0.0

    def set(self, pitch: float, roll: float, now: Optional[float] = None) -> None:
        with self._lock:
            self._sample = (pitch, roll)
            self._at = time.monotonic() if now is None else now

    def clear(self) -> None:
        with self._lock:
            self._sample = None

    def get(self, max_age_s: float, now: Optional[float] = None) -> Optional[Tuple[float, float]]:
        """(pitch, roll) in radians, or None when absent, older than
        ``max_age_s``, or ``max_age_s`` <= 0 (compensation disabled)."""
        if max_age_s <= 0:
            return None
        with self._lock:
            sample, at = self._sample, self._at
        if sample is None:
            return None
        t = time.monotonic() if now is None else now
        if t - at > max_age_s:
            return None
        return sample

    def status(self, now: Optional[float] = None) -> Optional[Tuple[float, float, float]]:
        """(pitch, roll, age_s) of the last sample regardless of age, for /health."""
        with self._lock:
            sample, at = self._sample, self._at
        if sample is None:
            return None
        t = time.monotonic() if now is None else now
        return sample[0], sample[1], t - at
