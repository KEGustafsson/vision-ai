"""Horizon line estimation.

In calibrated installs the horizon row is fixed in config. When ``auto_horizon``
is enabled we estimate it from the strongest near-horizontal intensity gradient
(sky is brighter than sea), which is cheap and robust enough as a fallback.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def detect_horizon_y(image: np.ndarray) -> Optional[float]:
    """Return the estimated horizon row, or None if indeterminate."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Row-mean profile; the horizon is where it drops fastest top->bottom.
    profile = gray.mean(axis=1).astype(np.float32)
    grad = np.abs(np.diff(profile))
    if grad.size == 0:
        return None
    y = int(np.argmax(grad))
    # Reject degenerate detections at the very top/bottom edge.
    if y < 5 or y > image.shape[0] - 5:
        return None
    return float(y)
