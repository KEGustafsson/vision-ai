"""Configuration loading: layered YAML + environment overrides.

Precedence (lowest to highest): ``config/default.yaml`` -> the mode-specific
file (``config/<mode>.yaml``) -> ``VISION_*`` environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class CameraConfig(BaseModel):
    name: str
    url: Optional[str] = None          # RTSP url (jetson/cpu modes)
    hfov_deg: float = 90.0             # horizontal field of view
    height_m: float = 2.5             # camera height above waterline
    horizon_y: Optional[float] = None  # pixel row of horizon; None => auto/uncalibrated
    bearing_offset_deg: float = 0.0   # mounting offset from bow (forward=0, aft=180)
    # ONVIF PTZ control (Hikvision zoom domes). When ptz is true the web UI
    # shows a control pad. Host/credentials default to those embedded in `url`
    # (rtsp://user:pass@host) so they're configured once; override here only if
    # the ONVIF service is on a different host/port/account than the RTSP feed.
    ptz: bool = False
    onvif_host: Optional[str] = None
    onvif_port: int = 80
    onvif_user: Optional[str] = None
    onvif_password: Optional[str] = None
    # Display-only lens correction: straightens barrel distortion and levels a
    # tilted mount in the annotated MJPEG/snapshot ONLY. The detection event
    # (bearings/range/CPA) is still computed from the raw frame, so these are
    # cosmetic and do not need to be metrically calibrated. Replace with values
    # from a real checkerboard calibration when available.
    undistort: bool = False            # enable display correction for this camera
    undistort_k1: float = 0.0          # radial barrel coeff (negative corrects barrel)
    undistort_f_factor: float = 0.75   # assumed focal length as a fraction of width
    undistort_alpha: float = 0.0       # 0 crop to valid pixels, 1 keep all (black edges)
    undistort_rotation_deg: float = 0.0  # CCW image-plane rotation to level the horizon
    # EXPERIMENTAL: apply the correction to the frame BEFORE detection (the
    # detector + geometry then see the corrected image) instead of display-only.
    # Feeds YOLO straighter frames, but bearings/range now depend on the (still
    # eyeballed) coefficients AND on horizon_y being re-measured on the corrected
    # frame — so treat geometry as approximate until a real calibration exists.
    undistort_before_detect: bool = False
    # Run the undistort resample on the GPU (torch grid_sample) when CUDA is
    # available; falls back to CPU (cv2.remap) automatically if not.
    undistort_gpu: bool = True
    # DeepStream only: when undistort is on, correction runs on the GPU via the
    # nvdewarper element (zero-copy, in-NVMM, before inference). By default a
    # starter dewarper config is generated from undistort_k1/_f_factor/_rotation.
    # Point this at a hand-tuned nvdewarper config file to override the generated
    # one once you've calibrated on the hardware.
    dewarper_config: Optional[str] = None


class GeometryConfig(BaseModel):
    auto_horizon: bool = False
    # Known real-world widths (metres) per canonical label, for known-size ranging.
    known_widths_m: Dict[str, float] = Field(
        default_factory=lambda: {"person": 0.5, "buoy": 0.8, "vessel": 4.0}
    )


class DetectorConfig(BaseModel):
    backend: str = "mock"              # mock | torch-cpu | torch-cuda | tensorrt
    model_pt: str = "yolov8n.pt"
    model_engine: str = "yolov8n.engine"
    confidence: float = 0.35           # publish threshold (worker-side filter)
    # Lower floor fed to YOLO/ByteTrack so runtime /control can both raise AND
    # lower the effective confidence; the worker filters at `confidence`.
    track_floor: float = 0.1
    imgsz: int = 640
    # DeepStream only: the nvstreammux output resolution, which is also the
    # display/MJPEG resolution and the coordinate space of every reported bbox
    # and `horizon_y`. nvinfer rescales this to `imgsz` internally on the GPU,
    # so detection still runs at imgsz while display + geometry stay native.
    # Set to the camera's native stream resolution (these domes stream 1280x960)
    # so horizon_y/range calibration is in real pixels.
    mux_width: int = 1280
    mux_height: int = 960
    # DeepStream nvstreammux batched-push-timeout (ms): how long the muxer waits to
    # assemble a full batch before pushing what it has. Keep it SHORT — prompt
    # pushes let the two cameras pipeline independently and reach the full input
    # frame rate; a long timeout forces both into one synchronous batch and throttles
    # to the per-batch processing cost. Over-generation is prevented by the probe's
    # PTS guard (drops muxer frame repeats), not by this value.
    mux_batch_timeout_ms: int = 40
    # Maximum detections kept by YOLO/NMS and passed into tracking per frame.
    # Lower values reduce post-processing/tracker/event/overlay workload in busy scenes.
    max_det: int = 20
    tracker: str = "bytetrack.yaml"
    # Reject detections whose bbox covers more than this fraction of the frame.
    # The boat's own hull/superstructure fills most of the frame when a camera
    # is aimed inboard and is otherwise mis-detected as a nearby "vessel",
    # producing false dark-target/collision alerts. Real contacts of interest
    # (distant vessels, buoys, persons) occupy a small fraction. 1.0 disables.
    max_area_frac: float = 0.4
    # Suppress a detection whose bbox lies largely inside a larger detection's
    # bbox (e.g. a buoy/person on a vessel's deck, or a duplicate nested box):
    # if intersection / inner-box-area exceeds this fraction, the inner
    # detection is dropped and the larger containing one is kept. A
    # person-in-water is never dropped (MOB safety). 1.0 disables.
    contained_frac: float = 0.8
    # Drop any detection whose estimated range is below this many metres (own-hull
    # artifacts / very-near clutter). Applied EARLY — before events and the
    # annotated overlay — so neither surfaces a too-close object. person is exempt
    # (man-overboard must still be seen up close); detections with no range
    # estimate are kept. 0 disables. The SignalK plugin owns the operational value
    # and pushes it at runtime via POST /control (detector.minTargetRangeM).
    min_target_range_m: float = 0.0
    # Canonical labels to surface (person | vessel | buoy). None/empty => all.
    # The SignalK plugin overrides this at runtime via POST /control so the
    # operator can pick object types from the admin UI. Filtering here keeps
    # both the event stream and the annotated overlay limited to the selection.
    classes: Optional[List[str]] = None
    # Track stabilizer: damp per-frame detection flicker by giving each track a
    # short lifecycle (confidence hysteresis + coasting across dropouts +
    # appearance debounce). Applies to BOTH the event and the overlay.
    stabilize: bool = True
    # Frames a new track must be seen before it is shown (debounce false pops).
    stabilize_confirm_frames: int = 3
    # Frames a confirmed-but-undetected track is coasted before being dropped.
    # ~max_coast_frames / target_fps seconds of persistence across dropouts.
    stabilize_max_coast_frames: int = 8
    # Off-threshold = confidence * this ratio; below it a shown track turns off.
    stabilize_hysteresis_ratio: float = 0.6
    # EMA smoothing factor for per-track confidence (higher = more reactive).
    stabilize_ema_alpha: float = 0.4
    # Fraction of pixel velocity applied while coasting (0 = freeze the box at
    # its last spot, 1 = full extrapolation). Damps a noisy velocity so a
    # coasted (dashed) box doesn't drift fast off the object across dropouts.
    stabilize_coast_velocity_factor: float = 0.4
    # Batch both cameras into a single inference (needs a batch-capable engine).
    # Removes the one-camera-at-a-time detector serialization. Falls back to
    # per-camera inference automatically when the engine is batch=1.
    batch_cameras: bool = False
    # How long a camera waits to rendezvous with the other before inferring solo.
    batch_wait_ms: int = 20
    # DeepStream only: which detection model to run. Exactly ONE model runs at a
    # time (never both). Selects both the nvinfer config and the class map:
    #   "coco"                -> COCO YOLOv8n, 80 classes (person/vessel/buoy/...)
    #   "forward-watch"       -> forward-watch marine model, 6 classes
    #                            (ship/boat/debris/buoy/kayak/log)
    #   "marine-surveillance" -> Roboflow Marine Surveillance YOLOv8s, 7 classes
    #                            (boat/buoy/kayak/sailboat/speedboat/vessel/warship);
    #                            NO person -> no man-overboard. Train on-box via
    #                            training/train_marine_surveillance.py.
    # See app/detector/classmap.py (MODEL_PGIE_CONFIG) for the registry.
    model: str = "coco"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 7000
    target_fps: float = 10.0           # processing cadence cap
    jpeg_quality: int = 80
    event_buffer: int = 200
    # Allowed CORS origins; default permissive for dev. Restrict to the SignalK
    # origin(s) in production (the plugin proxies same-origin anyway).
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])


class Settings(BaseModel):
    mode: str = "mock"                 # mock | cpu | jetson
    mock_source: str = "synthetic"     # synthetic | <video path>
    cameras: List[CameraConfig]
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    def camera(self, name: str) -> CameraConfig:
        for c in self.cameras:
            if c.name == name:
                return c
        raise KeyError(f"unknown camera: {name}")


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _apply_env(raw: dict) -> dict:
    """Overlay VISION_* environment variables onto the raw config dict."""
    env = os.environ
    if "VISION_MODE" in env:
        raw["mode"] = env["VISION_MODE"]
    if "VISION_MOCK_SOURCE" in env:
        raw["mock_source"] = env["VISION_MOCK_SOURCE"]
    if "VISION_PORT" in env:
        raw.setdefault("server", {})["port"] = int(env["VISION_PORT"])
    det = raw.setdefault("detector", {})
    if "VISION_MODEL_PT" in env:
        det["model_pt"] = env["VISION_MODEL_PT"]
    if "VISION_MODEL_ENGINE" in env:
        det["model_engine"] = env["VISION_MODEL_ENGINE"]
    # DeepStream model selector (coco | forward-watch); see DetectorConfig.model.
    if "VISION_DETECTOR_MODEL" in env:
        det["model"] = env["VISION_DETECTOR_MODEL"]
    # Per-camera URL overrides
    cams = {c["name"]: c for c in raw.get("cameras", [])}
    if "VISION_CAMERA_FORWARD_URL" in env and "forward" in cams:
        cams["forward"]["url"] = env["VISION_CAMERA_FORWARD_URL"]
    if "VISION_CAMERA_AFT_URL" in env and "aft" in cams:
        cams["aft"]["url"] = env["VISION_CAMERA_AFT_URL"]
    return raw


def load_settings(mode: Optional[str] = None) -> Settings:
    base = _load_yaml(CONFIG_DIR / "default.yaml")
    mode = mode or os.environ.get("VISION_MODE") or base.get("mode", "mock")
    merged = _deep_merge(base, _load_yaml(CONFIG_DIR / f"{mode}.yaml"))
    merged["mode"] = mode
    merged = _apply_env(merged)
    return Settings(**merged)
