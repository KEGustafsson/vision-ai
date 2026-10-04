"""Combine per-camera config + detector output into the geometry fields of the
detection event (relative bearing, range, range method/confidence)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from ..config import CameraConfig, GeometryConfig
from ..detector.base import RawTrack
from ..detector.classmap import is_person_in_water
from .attitude import HorizonLine, horizon_line
from .bearing import relative_bearing_deg
from .range import range_by_horizon, range_by_size


def estimate_bearing(track: RawTrack, cam: CameraConfig, width: int) -> float:
    rel = relative_bearing_deg(track.cx, width, cam.hfov_deg)
    return rel + cam.bearing_offset_deg


def resolve_horizon(cam: CameraConfig, base_y: Optional[float], width: int, height: int,
                    attitude: Optional[Tuple[float, float]]
                    ) -> Tuple[Optional[HorizonLine], bool]:
    """Horizon line for this frame, and whether IMU attitude was applied.

    ``base_y`` is the calibrated (``cam.horizon_y``) or auto-detected row. With a
    fresh ``attitude`` (pitch, roll in radians):

    * calibrated row — shifted and tilted by the attitude change from the
      camera's calibration reference (``horizon_ref_*_deg``);
    * auto-detected row — already measured at the current pitch, so only the
      heel slope is added (the reference is the current attitude).

    Without attitude (feature off, stale, or degenerate geometry) the line is
    the plain level row, exactly as before compensation existed.
    """
    if base_y is None:
        return None, False
    level = HorizonLine(y_center=base_y, slope=0.0, cx=width / 2.0)
    if attitude is None:
        return level, False
    pitch, roll = attitude
    if cam.horizon_y is not None:
        ref_pitch = math.radians(cam.horizon_ref_pitch_deg)
        ref_roll = math.radians(cam.horizon_ref_roll_deg)
    else:
        ref_pitch, ref_roll = pitch, roll
    line = horizon_line(base_y, width, height, cam.hfov_deg, cam.bearing_offset_deg,
                        pitch, roll, ref_pitch, ref_roll)
    if line is None:
        return level, False
    return line, True


def person_in_water(track: RawTrack, horizon: Optional[HorizonLine],
                    base_y: Optional[float]) -> bool:
    """Man-overboard candidate test against the horizon at the person's column.

    Fails toward alerting: a person counts as in the water if their waterline
    is below EITHER the compensated horizon or the plain level row it was
    derived from. Compensation then can only add MOB candidates (on the side
    where heel lowered the horizon), never remove one the uncompensated
    system would have raised — a wrong-signed or misaligned IMU can't hide a
    person in the water.
    """
    waterline_y = track.y + track.h
    hy = horizon.y_at(track.cx) if horizon is not None else None
    return (is_person_in_water(track.label, waterline_y, hy)
            or is_person_in_water(track.label, waterline_y, base_y))


def estimate_range(track: RawTrack, cam: CameraConfig, geo: GeometryConfig,
                   width: int, height: int, horizon_y: Optional[float],
                   horizon_slope: float = 0.0
                   ) -> Tuple[Optional[float], Optional[str], float]:
    """Return (range_m, method, confidence).

    ``horizon_y`` is the horizon row at the frame centre column; a non-zero
    ``horizon_slope`` (heel, rows per column) evaluates it at the target's own
    column and measures the depression perpendicular to the tilted horizon.
    """
    waterline_y = track.y + track.h  # bottom of bbox = waterline contact
    if horizon_y is not None:
        hy, object_y = horizon_y, waterline_y  # level: exactly the uncompensated path
        if horizon_slope:
            hy = horizon_y + horizon_slope * (track.cx - width / 2.0)
            depression_px = (waterline_y - hy) / math.sqrt(1.0 + horizon_slope * horizon_slope)
            object_y = hy + depression_px
        res = range_by_horizon(object_y, hy, cam.height_m,
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
