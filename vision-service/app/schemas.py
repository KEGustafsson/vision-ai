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
    deepstream = "deepstream"  # NVIDIA DeepStream: nvinfer + nvtracker GPU pipeline


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

    ``relative_bearing_deg`` is the **bow-relative** bearing: the angle off the
    camera optical axis plus the camera's mounting offset (forward = 0 deg,
    aft = 180 deg). Positive is to starboard, negative to port. The plugin
    obtains the true bearing by adding own ``headingTrue``.
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
    # True when this target was not detected in the current frame and is being
    # "coasted" from its last detection + velocity by the track stabilizer, so
    # the box/info persists smoothly across short detector dropouts.
    coasting: bool = False


class FrameSize(BaseModel):
    w: int
    h: int


class Inference(BaseModel):
    backend: Backend
    latency_ms: float = 0.0


class DetectionEvent(BaseModel):
    """One event per processed frame, per camera.

    ``camera`` is a free-form name (``forward``/``aft`` by default, but any
    configured camera name is valid) so additional cameras don't break the wire
    contract.
    """

    schema_version: str = SCHEMA_VERSION
    camera: str
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

    active_camera: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    # Maximum detections to keep per frame before tracking/event generation.
    max_targets: Optional[int] = Field(None, ge=1, le=300)
    mode_hint: Optional[str] = None  # e.g. "underway" | "docking" | "anchored"
    # Canonical labels to surface (person | vessel | buoy). Empty list => all.
    labels: Optional[List[str]] = None
    # Master on/off: when False the camera workers release their capture devices
    # and stop reading/inferring entirely (no frames, no events) until re-enabled.
    enabled: Optional[bool] = None


class PtzRequest(BaseModel):
    """POST /ptz/{camera} — ONVIF PTZ control.

    ``action`` selects the operation; ``move`` uses the normalised pan/tilt/zoom
    velocities (-1..1, +pan = right, +tilt = up, +zoom = in).
    """

    action: str = "move"  # move | stop | home
    pan: float = Field(0.0, ge=-1.0, le=1.0)
    tilt: float = Field(0.0, ge=-1.0, le=1.0)
    zoom: float = Field(0.0, ge=-1.0, le=1.0)


class HealthResponse(BaseModel):
    status: str = "ok"
    mode: str
    backend: Backend
    cameras: List[str]
    uptime_s: float
    active_camera: Optional[str] = None
    camera_errors: dict = Field(default_factory=dict)
    # Whether detection is currently running (toggled via /control `enabled`).
    detection_enabled: bool = True
    # Active maximum detections/tracks kept per processed frame.
    max_targets: int = 20
    # Active canonical labels surfaced by the workers; null means all labels.
    labels: Optional[List[str]] = None
