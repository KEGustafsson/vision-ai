"""Configuration loading: layered YAML + environment overrides.

Precedence (lowest to highest): ``config/default.yaml`` -> the mode-specific
file (``config/<mode>.yaml``) -> ``VISION_*`` environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

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
    model_pt: str = "yolo11n.pt"
    model_engine: str = "yolo11n.engine"
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
    # Sticky max-targets cap: when more targets are live than max_det allows, a
    # target emitted last frame keeps its slot unless a challenger's confidence
    # beats it by this margin. Stops two near-tied targets from swapping the
    # last slot every few frames (the loser blinks in and out of the overlay
    # and target list). person always ranks first regardless. 0 disables.
    max_det_sticky_margin: float = 0.05
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
    # MOB-critical override: person tracks confirm after this many frames instead
    # (default 1 = first frame). A person in the water must not be held back by the
    # generic false-positive debounce; the plugin's MOB persistence counter still
    # debounces the actual alarm, so confirming on frame 1 here is safe.
    stabilize_person_confirm_frames: int = 1
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
    # Box smoothing: every emitted box is the rolling AVERAGE of the track's
    # last stabilize_smooth_window raw boxes. Deliberately simple (operator
    # decision): no motion model, no prediction, no extent-holding — the box
    # just follows the detections, with per-frame jitter and shape flips
    # spread across the window instead of snapping. Bigger window = calmer but
    # laggier (a mover's box trails by about half the window). 1 disables.
    stabilize_smooth: bool = True
    stabilize_smooth_window: int = 5
    # Jump gate on the smoothed box: real objects don't teleport, so a raw box
    # whose center leaps more than this fraction of the box's larger dimension
    # per elapsed frame — or whose width/height changes by more than
    # (1 + this) per elapsed frame — is a FALSE measurement: it is not
    # averaged in and the held average is emitted instead. Judged only against
    # boxes already seen (no motion estimate). 0 disables the gate.
    stabilize_jump_tol: float = 0.35
    # An out-of-gate box observed this many CONSECUTIVE frames is real (the
    # target genuinely is elsewhere / another size): the average restarts
    # there. In-gate frames reset the count, so a recurring lone spike (glint,
    # marina cluster box) never accumulates acceptance.
    stabilize_jump_confirm_frames: int = 3
    # Track lock (the ByteTrack track_buffer / NvDCF shadow-tracking idea): a
    # track seen at least this many frames has proven itself real and may
    # coast stabilize_lock_coast_factor times longer than
    # stabilize_max_coast_frames before it is dropped, so an established
    # vessel rides out a longer dropout (wave occlusion, a wake burst) with
    # its box and id intact while a young track still dies fast. 0 disables.
    stabilize_lock_hits: int = 30
    stabilize_lock_coast_factor: float = 2.0
    # Weight each box in the smoothing window by its detection confidence
    # (StrongSORT's NSA-Kalman idea, arXiv:2202.13514, applied to the rolling
    # average): a marginal low-confidence measurement perturbs the shown box
    # less than a solid detection. false => plain unweighted average.
    stabilize_conf_weight: bool = True
    # Waterline re-identification: keep ONE detection id on a vessel whose box
    # alternates between partial and full extents (hull only <-> hull + mast).
    # The backend trackers (ByteTrack/NvDCF) associate by box IoU, so that shape
    # jump mints a fresh track id and the same target flickers between two ids;
    # re-id aliases the new id back when both boxes stand on the same waterline
    # footprint (see app/detector/tracker.py resolve()). person is always exempt
    # (two swimmers must never be fused into one MOB target). Trade-off: two
    # same-label targets that genuinely share a footprint (one passing close
    # behind another) can be fused while they overlap; the thresholds below
    # keep that window narrow.
    reid: bool = True
    # How many frames back a disappeared track can be re-identified (~20 s at
    # the measured ~6 FPS per camera), so a vessel that drops out re-acquires
    # its id instead of appearing as a new target. Aligned with the NvDCF
    # re-association search range (maxTrackletMatchingTimeSearchRange = 120 in
    # nvdcf_config.yml): the tracker and the waterline re-id give up on a
    # dropout at the same age. A NEW vessel arriving in the spot is kept from
    # inheriting the id by the width-similarity and motion-prediction gates
    # below (the buffered-gate widening is capped at reid_buffer_max_frac, so a
    # long gap does not judge loosely without bound), not by keeping this
    # window tight.
    reid_max_gap_frames: int = 120
    # Minimum horizontal overlap, as a fraction of the narrower box's width.
    reid_min_x_overlap: float = 0.5
    # Max bottom-edge (waterline) misalignment, as a fraction of the SHORTER
    # box's height (the hull) — a mast-height tolerance would bridge two
    # vertically separate targets.
    reid_bottom_tol_frac: float = 0.35
    # Max waterline-width mismatch (wider/narrower) between the candidate and
    # the stored footprint. A partial/full flip preserves the hull's width (a
    # mast adds height, not width), so a larger mismatch is a DIFFERENT vessel
    # that must not inherit the id.
    reid_max_width_ratio: float = 1.6
    # Buffered matching (C-BIoU, arXiv:2211.14317): the re-id gates RELAX as
    # the dropout grows — the footprint-overlap window and the waterline
    # tolerance widen by this fraction of the narrower box's dimension per
    # missed frame (velocity prediction is less certain the longer the target
    # was unseen), capped at reid_buffer_max_frac. A fresh partial/full flip
    # is still judged tightly. 0 disables the widening.
    reid_buffer_frac_per_frame: float = 0.03
    reid_buffer_max_frac: float = 0.25
    # Direction-consistency gate (OC-SORT's observation-centric momentum,
    # arXiv:2203.14360): a track moving at or above this speed (px/frame) is
    # only re-identified by a candidate displaced broadly ALONG its direction
    # of travel — a box appearing clearly BEHIND a mover is a different
    # vessel, even where the buffered gate would geometrically accept it.
    # Counterbalances the widened matching space above. 0 disables.
    reid_dir_min_speed_px: float = 2.0
    # How long an idle track's identity (velocity history, display id, wire
    # stable_id) is retained before being pruned. MUST cover the deepest
    # backend resurrection window — NvDCF shadow tracking holds a lost raw id
    # for maxShadowTrackingAge frames (240 in nvdcf_config.yml) and re-acquires
    # the vessel with the SAME raw id; retaining for less means that reborn
    # track finds its display id freed + quarantined and the same vessel
    # reblips on the chart under a fresh identity. Also must exceed
    # reid_max_gap_frames, or the waterline re-id's candidate footprints are
    # forgotten before the gap closes. ~43 s at the measured ~6 FPS per camera.
    track_memory_frames: int = 260
    # Batch both cameras into a single inference (needs a batch-capable engine).
    # Removes the one-camera-at-a-time detector serialization. Falls back to
    # per-camera inference automatically when the engine is batch=1.
    batch_cameras: bool = False
    # How long a camera waits to rendezvous with the other before inferring solo.
    batch_wait_ms: int = 20
    # Which detection model is loaded. Exactly ONE model runs at a time (never
    # both). Selects the class map used to decode raw class ids on EVERY
    # backend (raw ids collide across models: forward-watch 0 = ship, COCO 0 =
    # person, so decoding with the wrong table mislabels every detection) and,
    # on DeepStream, also the nvinfer config. On torch/tensorrt backends this
    # must match the weights loaded via model_pt/model_engine:
    #   "coco"                -> COCO, 80 classes (person/vessel/buoy/...); YOLO11n
    #                            on torch/tensorrt and on DeepStream
    #   "forward-watch"       -> forward-watch marine model, 6 classes
    #                            (ship/boat/debris/buoy/kayak/log)
    #   "marine-surveillance" -> Roboflow Marine Surveillance YOLOv8s, 7 classes
    #                            (boat/buoy/kayak/sailboat/speedboat/vessel/warship);
    #                            NO person -> no man-overboard. Train on-box via
    #                            training/train_marine_surveillance.py.
    # See app/detector/classmap.py (MODEL_PGIE_CONFIG) for the registry.
    # Constrained to the known registry keys so a typo fails fast at config
    # load instead of silently falling back to the COCO class map and
    # mislabeling every detection at runtime.
    model: Literal["coco", "forward-watch", "marine-surveillance"] = "coco"

    @model_validator(mode="after")
    def _identity_retention_covers_reid(self) -> "DetectorConfig":
        """Fail at config load, not silently at runtime: identity retention
        shorter than the re-id gap means the waterline re-id's candidate
        footprints are pruned before the gap closes — re-id quietly stops
        working for exactly the long dropouts it exists to bridge. (The other
        coupling, track_memory_frames >= NvDCF maxShadowTrackingAge, cannot be
        checked here: nvdcf_config.yml is an OpenCV-FileStorage file parsed by
        DeepStream itself, not by this config — it is documented on both
        sides instead.)"""
        if self.track_memory_frames < self.reid_max_gap_frames:
            raise ValueError(
                f"track_memory_frames ({self.track_memory_frames}) must be >= "
                f"reid_max_gap_frames ({self.reid_max_gap_frames}): idle-track "
                "identity would be pruned before the re-id window closes")
        return self


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 7000
    target_fps: float = 10.0           # processing cadence cap
    jpeg_quality: int = 80
    event_buffer: int = 200
    # Encode the annotated MJPEG/snapshot frames on the Jetson NVJPG hardware
    # block (GStreamer nvjpegenc) instead of CPU cv2.imencode, lifting the encode
    # out of the per-frame `post` cost. Needs python3-gi + GStreamer introspection
    # in the image (see Dockerfile); silently falls back to CPU when unavailable,
    # so it's safe to leave off on hosts without the binding.
    hw_jpeg: bool = False
    # Allowed CORS origins. Empty by default (no cross-origin access): the plugin
    # proxies the stream same-origin, so the browser never needs to reach the
    # container cross-origin. Set explicitly to the SignalK origin(s) only if you
    # deliberately expose the container to another origin. Avoid "*", which — with
    # no auth on the control endpoints — would let any web page drive the cameras.
    cors_origins: List[str] = Field(default_factory=list)
    # Cap concurrent MJPEG stream clients and WebSocket subscribers so an
    # unauthenticated peer can't exhaust the event loop / encode budget on a
    # resource-tight Jetson by opening connections without bound.
    max_stream_clients: int = Field(16, ge=1)
    max_ws_clients: int = Field(16, ge=1)


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
        try:
            raw.setdefault("server", {})["port"] = int(env["VISION_PORT"])
        except ValueError:
            raise ValueError(
                f"VISION_PORT must be an integer, got {env['VISION_PORT']!r}"
            )
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
