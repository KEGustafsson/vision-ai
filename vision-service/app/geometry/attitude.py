"""Horizon line compensated for the boat's attitude (IMU pitch/roll).

The calibrated ``horizon_y`` is one pixel row, measured with the boat at some
attitude (ideally level, at rest). Underway the hull squats or planes, a sailing
boat heels, and loading changes the trim — and the cameras tilt with the hull.
A pitch of 3 deg moves the horizon ~34 px on a 1280x960 / 90 deg camera, while a
vessel 1 km off sits ~2 px below the horizon from 2.5 m: an uncompensated trim
change makes horizon-depression range meaningless. Heel also TILTS the horizon,
so the error grows towards the frame edges.

Given the boat's attitude (SignalK ``navigation.attitude``, forwarded by the
plugin) this module works out where the horizon now lies in a camera's image:
its row at the frame centre column plus its slope. It is exact for a pinhole
camera (lens distortion and the ~0.05 deg dip of the sea horizon are already in
the calibrated row and are ignored for the *change*):

* the world "up" vector is rotated into the boat frame (x forward, y starboard,
  z down) by the SignalK attitude — roll +ve = list to starboard, pitch +ve =
  bow up;
* then into the camera frame (x right, y down, z along the optical axis) for a
  camera looking at ``bearing_offset_deg`` off the bow, tilted down by a mount
  angle that is solved from the calibrated row at the reference attitude;
* the horizon is where a pixel ray is perpendicular to "up": a straight line.

Only the *difference* from the reference attitude matters, so the IMU's own
mounting misalignment is absorbed by recording its reading when ``horizon_y``
is calibrated (``horizon_ref_pitch_deg``/``horizon_ref_roll_deg``).

Angles are radians at runtime (SI, as SignalK); degrees only in config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .range import focal_px

# Beyond this the horizon is (nearly) out of the camera's view along its axis —
# looking straight up or down — and the line is undefined or meaningless.
_MAX_AXIS_ELEVATION = math.radians(80.0)


@dataclass(frozen=True)
class HorizonLine:
    """Horizon in image pixels: ``y_center`` at column ``cx`` with ``slope``
    (rows per column, +ve = lower on the right / starboard side of the image)."""

    y_center: float
    slope: float
    cx: float

    def y_at(self, x: float) -> float:
        return self.y_center + self.slope * (x - self.cx)


def _up_in_level_camera(pitch: float, roll: float, bearing: float) -> Tuple[float, float, float]:
    """World-up expressed in a LEVEL camera frame looking at ``bearing`` off the
    bow: returns (along the optical axis, along camera-right, along camera-down)."""
    sp, cp = math.sin(pitch), math.cos(pitch)
    sr, cr = math.sin(roll), math.cos(roll)
    # Up in the boat frame (x fwd, y stbd, z down) for a ZYX (yaw-pitch-roll) attitude.
    ux_b, uy_b, uz_b = sp, -sr * cp, -cr * cp
    sb, cb = math.sin(bearing), math.cos(bearing)
    along = cb * ux_b + sb * uy_b
    right = -sb * ux_b + cb * uy_b
    down = uz_b
    return along, right, down


def horizon_line(
    base_y: float,
    width: int,
    height: int,
    hfov_deg: float,
    bearing_offset_deg: float,
    pitch: float,
    roll: float,
    ref_pitch: float = 0.0,
    ref_roll: float = 0.0,
) -> Optional[HorizonLine]:
    """Horizon line at attitude (``pitch``, ``roll``) for a camera whose horizon
    sat at row ``base_y`` (frame centre column) at (``ref_pitch``, ``ref_roll``).

    Passing the current attitude as the reference keeps the centre row at
    ``base_y`` and only adds the heel slope — used for an auto-detected horizon,
    whose row is measured live. Returns None when the geometry is degenerate
    (camera axis near vertical), so the caller falls back to the uncompensated
    horizon.
    """
    if width <= 0 or height <= 0:
        return None
    f = focal_px(hfov_deg, width)
    cx, cy = width / 2.0, height / 2.0
    bearing = math.radians(bearing_offset_deg)

    def psi(p: float, r: float) -> Tuple[Optional[float], float, float]:
        along, right, down = _up_in_level_camera(p, r, bearing)
        k = math.hypot(along, down)
        if k < 1e-6:
            return None, right, k
        # Elevation-like angle of the level camera's axis: up = -k(sin psi, cos psi)
        # in the (along, down) plane, so psi = -(axis elevation).
        return math.atan2(-along, -down), right, k

    psi_ref, _, _ = psi(ref_pitch, ref_roll)
    psi_now, right, k = psi(pitch, roll)
    if psi_ref is None or psi_now is None:
        return None
    # Mount tilt solved from the calibrated row: base_y = cy - f*tan(psi_ref + t).
    # The current axis angle is then psi_now + t.
    phi = math.atan((cy - base_y) / f) + psi_now - psi_ref
    if abs(phi) >= _MAX_AXIS_ELEVATION:
        return None
    c = math.cos(phi)
    y_center = cy - f * math.tan(phi)
    slope = right / (k * c)
    return HorizonLine(y_center=y_center, slope=slope, cx=cx)
