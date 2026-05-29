"""Authoritative data contract for the Marine Vision-AI system.

The :class:`DetectionEvent` model is the single source of truth for the JSON
that flows from the vision container to the SignalK plugin over the WebSocket
(``/ws/events``).  The JSON Schema consumed by the plugin is generated from
this module via ``scripts/export_schema.py`` so the two sides cannot drift.

Units on the wire are the container's *native* units: angles in **degrees**,
ranges in **metres**, pixel velocities in **pixels/frame**.  The plugin converts
to SI (radians/metres/m·s⁻¹) once, at the boundary, before publishing to SignalK.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


class Camera(str, Enum):
    forward = "forward"
    aft = "aft"


class Backend(str, Enum):
    tensorrt = "tensorrt"
    torch_cuda = "torch-cuda"
    torch_cpu = "torch-cpu"
    mock = "mock"


class RangeMethod(str, Enum):
    horizon = "horizon"
    known_size = "known_size"


class CalibrationStatus(str, Enum):
    ok = "ok"
    uncalibrated = "uncalibrated"
    auto = "auto"


class BBox(BaseModel):
    """Bounding box in pixels, top-left origin."""

    x: float
    y: float
    w: float
    h: float


class Geometry(BaseModel):
    """Monocular geometry estimated by the container.

    ``relative_bearing_deg`` is measured from the camera's optical axis,
    positive to starboard (right of frame), negative to port.
    """

    relative_bearing_deg: float
    range_m: Optional[float] = None
    range_method: Optional[RangeMethod] = None
    range_confidence: float = Field(0.0, ge=0.0, le=1.0)


class PixelVelocity(BaseModel):
    """Centroid velocity in pixels per processed frame, from track history."""

    vx: float = 0.0
    vy: float = 0.0


class Target(BaseModel):
    track_id: Optional[int] = None
    label: str
    coco_class: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BBox
    is_person_in_water: bool = False
    geometry: Geometry
    pixel_velocity: PixelVelocity = Field(default_factory=PixelVelocity)
    first_seen: Optional[str] = None
    age_frames: int = 0


class FrameSize(BaseModel):
    w: int
    h: int


class Inference(BaseModel):
    backend: Backend
    latency_ms: float = 0.0


class DetectionEvent(BaseModel):
    """One event per processed frame, per camera."""

    schema_version: str = SCHEMA_VERSION
    camera: Camera
    timestamp: str  # ISO-8601 UTC
    frame_seq: int
    frame_size: FrameSize
    horizon_y: Optional[float] = None
    inference: Inference
    calibration_status: CalibrationStatus = CalibrationStatus.uncalibrated
    targets: List[Target] = Field(default_factory=list)


# --- Control API models -----------------------------------------------------


class ControlRequest(BaseModel):
    """POST /control — change runtime behaviour without a restart."""

    active_camera: Optional[Camera] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    mode_hint: Optional[str] = None  # e.g. "underway" | "docking" | "anchored"


class HealthResponse(BaseModel):
    status: str = "ok"
    mode: str
    backend: Backend
    cameras: List[Camera]
    uptime_s: float
