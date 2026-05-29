"""Relative bearing from a pixel x-coordinate.

For a rectilinear lens with horizontal field of view ``hfov`` over image width
``W``, the bearing of a column ``px`` relative to the optical axis is::

    bearing_deg = (hfov / 2) * (1 - 2 * px / W)

Positive values are to starboard (right of frame centre), negative to port.
A per-camera mounting offset (forward bow = 0 deg, aft = 180 deg) is added by
the caller via the camera config.
"""

from __future__ import annotations


def relative_bearing_deg(px: float, image_width: int, hfov_deg: float) -> float:
    if image_width <= 0:
        return 0.0
    return (hfov_deg / 2.0) * (1.0 - 2.0 * px / image_width)
