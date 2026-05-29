"""Monocular range estimation for objects on the water surface.

Two methods:

1. **Horizon depression** (preferred for floating objects). Given the camera
   height ``h`` above the waterline and the vertical pixel offset of the
   object's waterline below the detected horizon, the depression angle is
   ``theta = (object_y - horizon_y) * IFOV`` where ``IFOV = vfov / H``. The
   range to the waterline contact is ``h / tan(theta)``.

2. **Known-size fallback**. If the real-world width of the class is known,
   ``range = focal_px * real_width / pixel_width`` with
   ``focal_px = (W/2) / tan(hfov/2)``.

Both return ``(range_m, confidence)``; confidence is a coarse 0..1 heuristic
that the plugin uses to gate which targets get georeferenced.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple


def vfov_from_hfov(hfov_deg: float, width: int, height: int) -> float:
    """Vertical FOV assuming square pixels."""
    if width <= 0:
        return hfov_deg
    hfov = math.radians(hfov_deg)
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * (height / width))
    return math.degrees(vfov)


def range_by_horizon(object_y: float, horizon_y: float, height_m: float,
                     hfov_deg: float, width: int, image_height: int) -> Optional[Tuple[float, float]]:
    """Return (range_m, confidence) or None if the object is at/above the horizon."""
    if object_y <= horizon_y:
        return None
    vfov = vfov_from_hfov(hfov_deg, width, image_height)
    ifov_deg = vfov / image_height
    theta = math.radians((object_y - horizon_y) * ifov_deg)
    if theta <= 1e-4:
        return None
    rng = height_m / math.tan(theta)
    # Confidence falls off as the object nears the horizon (tiny angles -> noisy).
    depression_frac = (object_y - horizon_y) / max(image_height - horizon_y, 1)
    conf = max(0.2, min(0.9, depression_frac * 1.5))
    return rng, conf


def focal_px(hfov_deg: float, width: int) -> float:
    return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def range_by_size(pixel_width: float, real_width_m: float, hfov_deg: float,
                  width: int) -> Optional[Tuple[float, float]]:
    if pixel_width <= 1.0 or real_width_m <= 0:
        return None
    rng = focal_px(hfov_deg, width) * real_width_m / pixel_width
    # Known-size ranging is coarse without precise calibration.
    return rng, 0.4
