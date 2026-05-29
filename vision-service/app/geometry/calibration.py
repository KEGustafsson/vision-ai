"""Combine per-camera config + detector output into the geometry fields of the
detection event (relative bearing, range, range method/confidence)."""

from __future__ import annotations

from typing import Optional, Tuple

from ..config import CameraConfig, GeometryConfig
from ..detector.base import RawTrack
from .bearing import relative_bearing_deg
from .range import range_by_horizon, range_by_size


def estimate_bearing(track: RawTrack, cam: CameraConfig, width: int) -> float:
    rel = relative_bearing_deg(track.cx, width, cam.hfov_deg)
    return rel + cam.bearing_offset_deg


def estimate_range(track: RawTrack, cam: CameraConfig, geo: GeometryConfig,
                   width: int, height: int, horizon_y: Optional[float]
                   ) -> Tuple[Optional[float], Optional[str], float]:
    """Return (range_m, method, confidence)."""
    waterline_y = track.y + track.h  # bottom of bbox = waterline contact
    if horizon_y is not None:
        res = range_by_horizon(waterline_y, horizon_y, cam.height_m,
                               cam.hfov_deg, width, height)
        if res is not None:
            return res[0], "horizon", res[1]
    # Fall back to known-size ranging if we have a width prior for the label.
    real_w = geo.known_widths_m.get(track.label)
    if real_w:
        res = range_by_size(track.w, real_w, cam.hfov_deg, width)
        if res is not None:
            return res[0], "known_size", res[1]
    return None, None, 0.0
