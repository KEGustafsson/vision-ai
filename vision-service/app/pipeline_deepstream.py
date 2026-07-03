"""DeepStream GPU pipeline — zero-copy decode → inference → tracking.

Architecture
============
All camera frames stay in NVMM (Jetson unified GPU/CPU memory) from RTSP
decode through TRT inference. The CPU never touches a pixel before inference.

    rtspsrc(N)
        → nvv4l2decoder       # hardware H.264/H.265 decode; output: NVMM NV12
        → nvvideoconvert      # colour-space: stays in NVMM
        → nvstreammux         # batch N cameras into one NVMM buffer; resizes to 640×640
        → nvinfer             # reads NVMM directly → TRT engine (FP16) → detections
        → nvtracker           # NvDCF on GPU → stable track IDs
        → nvvideoconvert      # → NVMM RGBA (for nvdsosd; probe maps it for horizon)
        → [pad probe]         # reads NvDsObjectMeta + attaches display meta (no pixel copy)
        → nvstreamdemux       # split batch back into per-camera NVMM buffers
        → (per camera) nvdsosd → nvvideoconvert(NVMM I420) → nvjpegenc → appsink
                              # per-camera OSD renders only that source's display meta

Display path is now fully zero-copy: pixels stay in NVMM through overlay + JPEG
encode and only the compressed bytes reach the CPU, on the per-camera appsink
(_on_jpeg_sample → LatestFrame). The single remaining host pixel access is the
auto-horizon surface map, throttled to ~1/s per camera (_horizon_for); explicit
horizon calibration avoids it entirely.

What is eliminated vs. pipeline.py
====================================
  Before: nvv4l2dec(NVMM) → nvvidconv(BGRx NVMM) → videoconvert(BGR CPU)
          → numpy → model.track() CUDA memcpy → TRT
  After:  nvv4l2dec(NVMM) → nvstreammux(NVMM batch) → nvinfer(TRT reads NVMM)

  GPU tracking:
    Before: ByteTrack in Python on CPU
    After:  NvDCF in nvtracker on GPU

  Batch serialisation:
    Before: CameraWorker threads serialize on a Python lock (one camera at a time)
    After:  nvstreammux batches both cameras; nvinfer infers the full batch in one
            kernel call; nvtracker updates both streams in parallel on GPU

Prerequisites
=============
  sudo apt-get install deepstream-7.1
  pip install /opt/nvidia/deepstream/deepstream/lib/pyds-*.whl

  # Build custom YOLOv8 nvinfer parser (C, one-time on Jetson):
  git clone https://github.com/marcoslucianops/DeepStream-Yolo /tmp/ds-yolo
  CUDA_VER=$(nvcc --version | grep -oP 'V\\K[0-9]+\\.[0-9]+') \\
      make -C /tmp/ds-yolo/nvdsinfer_custom_impl_Yolo
  cp /tmp/ds-yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so \\
      vision-service/deepstream/

  # Export raw YOLOv8n ONNX (no end-to-end NMS — deepstream-yolo requirement):
  python -c "from ultralytics import YOLO; YOLO('models/yolov8n.pt').export(
      format='onnx', simplify=True, opset=17, imgsz=640)"
  # nvinfer auto-builds yolov8n_ds.engine on first run; or pre-build with trtexec:
  # trtexec --onnx=models/yolov8n.onnx --fp16 --saveEngine=models/yolov8n_ds.engine \\
  #         --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 \\
  #         --maxShapes=images:2x3x640x640
"""

from __future__ import annotations

import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .api.osd import draw_event
from .config import CameraConfig, Settings
from .detector.base import RawTrack
from .detector.classmap import (
    MODEL_PGIE_CONFIG,
    is_person_in_water,
    label_for_model,
)
from .detector.stabilizer import TrackStabilizer, cap_targets_sticky
from .detector.tracker import VelocityTracker, reid_options
from .geometry import detect_horizon_y, estimate_bearing, estimate_range
from .pipeline import _drop_contained_targets  # shared geometry filter, same package
from .schemas import (
    Backend,
    BBox,
    CalibrationStatus,
    DetectionEvent,
    FrameSize,
    Geometry,
    Inference,
    PixelVelocity,
    RangeMethod,
    Target,
)
from .util import EventBuffer, LatestFrame

_STALL_TIMEOUT_S = 5.0  # mirror pipeline.py: flag a camera with no frames this long
# Self-heal threshold: if EVERY camera has been silent this long while detection
# is enabled, the fault is pipeline-level (stale RTSP sessions, stuck muxer —
# cases where the bus posts no ERROR so the supervisor won't act on its own).
# The watchdog quits the loop so the supervisor rebuilds with fresh RTSP
# connections. Comfortably above the RTSP preroll time so a slow start can't
# trigger a rebuild storm.
_STALL_REBUILD_S = 20.0
# Zero-copy keeps pixels in NVMM, so auto-horizon (which needs host pixels) is
# refreshed by mapping the RGBA surface at most this often per camera — rare
# enough that the per-frame path stays copy-free, frequent enough to track a
# horizon that drifts with the boat's pitch/roll.
_HORIZON_REFRESH_S = 1.0
_GST_CLOCK_TIME_NONE = 0xFFFF_FFFF_FFFF_FFFF  # GStreamer "invalid timestamp" sentinel
# Auto-restart after a fatal GStreamer error/EOS: a transient RTSP/decoder glitch
# must not take detection down until a manual container restart. The supervisor
# rebuilds the pipeline with exponential backoff and keeps trying indefinitely
# (a safety system should keep attempting to recover); the restart count and last
# error are surfaced in /health so a flapping feed is visible.
_RESTART_BACKOFF_INITIAL_S = 2.0
_RESTART_BACKOFF_MAX_S = 30.0
# Window after a rebuild during which /health still reports "degraded". Once a
# restart is older than this and the pipeline has stayed up, health returns to
# "ok" — a single recovered restart shouldn't mark the container unhealthy for
# its whole life. Comfortably longer than the max backoff so a flapping pipeline
# (which restarts again before the window closes) keeps reading degraded.
_RESTART_DEGRADED_WINDOW_S = 120.0
_DEEPSTREAM_DIR = Path(__file__).resolve().parent.parent / "deepstream"
_TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
_TRACKER_CFG_STOCK = Path(
    "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app"
    "/config_tracker_NvDCF_perf.yml"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _check_imports() -> None:
    """Raise ImportError with actionable install instructions if pyds/GI is absent."""
    try:
        import gi
        gi.require_version("Gst", "1.0")
        import pyds  # noqa: F401  # type: ignore[import]
        from gi.repository import Gst  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "DeepStream Python bindings (pyds) not available.\n"
            "Install on Jetson JetPack 6 + DeepStream 7.x:\n"
            "  sudo apt-get install deepstream-7.1\n"
            "  pip install /opt/nvidia/deepstream/deepstream/lib/pyds-*.whl\n"
            "See vision-service/config/deepstream.yaml for full setup instructions."
        ) from exc


def _tracker_cfg_path() -> str:
    """Return path to the NvDCF YAML config, preferring our tuned copy."""
    local = _DEEPSTREAM_DIR / "nvdcf_config.yml"
    if local.exists():
        return str(local)
    if _TRACKER_CFG_STOCK.exists():
        return str(_TRACKER_CFG_STOCK)
    raise FileNotFoundError(
        f"NvDCF tracker config not found at {local} or {_TRACKER_CFG_STOCK}. "
        "Either install DeepStream 7.x or create vision-service/deepstream/nvdcf_config.yml."
    )


# ── Per-camera lightweight proxy (exposed as pipeline.workers for /health) ──


@dataclass
class _CameraProxy:
    """Stand-in for CameraWorker; exposes .error for /health and /control."""
    name: str
    error: Optional[str] = None
    # Wall-clock of the last frame this camera produced; the watchdog flags a
    # wedged RTSP feed as an error after STALL_TIMEOUT_S (one dead camera in a
    # multi-cam batch otherwise goes unnoticed because the pipeline stays alive).
    last_frame_at: float = field(default_factory=time.time)


# ── Per-camera mutable inference state (accessed only from GLib probe thread) ──


@dataclass
class _StreamState:
    cam: CameraConfig
    settings: Settings
    stabilizer: Optional[TrackStabilizer]
    vel: VelocityTracker = field(default_factory=VelocityTracker)
    seq: int = 0
    confidence: float = 0.35
    allowed_labels: Optional[frozenset] = None
    min_target_range_m: float = 0.0
    # Last processed buffer PTS — used to drop muxer frame repeats so output never
    # exceeds the camera's real frame rate. dup_skipped counts those drops.
    last_pts: int = -1
    dup_skipped: int = 0
    # Throttled auto-horizon cache (zero-copy path): last detected horizon and the
    # monotonic time it was computed, so we only map the surface ~1/s.
    horizon_y_cached: Optional[float] = None
    last_horizon_t: float = 0.0
    # Track ids emitted in the last event, for the sticky max-targets cap.
    emitted_ids: set = field(default_factory=set)


# ── Main pipeline ─────────────────────────────────────────────────────────────


class DeepStreamPipeline:
    """Drop-in replacement for Pipeline using NVIDIA DeepStream.

    Exposes exactly the same interface as Pipeline (events, frames, workers,
    start/stop, set_* methods) so main.py can select either without changes to
    the rest of the service.
    """

    def __init__(self, settings: Settings, logger) -> None:
        self.settings = settings
        self._log = logger
        self.events = EventBuffer(maxlen=settings.server.event_buffer)
        self.frames = LatestFrame()
        # workers mirrors Pipeline.workers: name → proxy with .error for /health
        self.workers: Dict[str, _CameraProxy] = {
            cam.name: _CameraProxy(cam.name) for cam in settings.cameras
        }
        self.started_at = time.time()
        self.active_camera: str = settings.cameras[0].name if settings.cameras else "forward"
        self.mode_hint: Optional[str] = None
        self.enabled: bool = True

        # Per-camera inference state — written only during start(), read only
        # from the GLib probe callback thread after that.
        self._states: Dict[str, _StreamState] = {}

        # Map source_id (0, 1, …) → camera name; populated during start().
        self._src_idx_to_name: Dict[int, str] = {}

        # GStreamer pipeline + GLib main loop (in daemon thread)
        self._gst = None
        self._loop = None
        # Supervisor thread: runs the GLib loop and rebuilds the pipeline with
        # backoff after a fatal GStreamer error/EOS (see _supervise).
        self._supervisor: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._restart_count = 0
        self._last_error: Optional[str] = None
        # Why the GLib loop was quit deliberately (detection toggle, watchdog
        # stall rebuild) — None when it exited on its own (bus ERROR/EOS).
        # Written by _quit_loop / consumed once by _supervise.
        self._exit_reason: Optional[str] = None
        # Gates supervisor rebuilds: cleared while detection is disabled so the
        # pipeline stays fully torn down (a live RTSP pipeline can't sit PAUSED
        # — the sessions go stale and never deliver frames again on resume).
        self._enabled_evt = threading.Event()
        self._enabled_evt.set()
        # Monotonic timestamp of the most recent rebuild, so /health can treat a
        # restart as "degraded" only while it is recent (actively recovering)
        # rather than latching on the cumulative count for the container's life.
        self._last_restart_ts: Optional[float] = None

        # Holds generated nvdewarper config files; cleaned up on stop().
        self._dewarp_tmp: Optional[tempfile.TemporaryDirectory] = None

        # Runtime-adjustable via /control; guarded by _lock because they are
        # written from FastAPI/uvicorn threads and read from the GLib probe thread.
        self._lock = threading.Lock()
        self._confidence: float = settings.detector.confidence
        self._allowed_labels: Optional[frozenset] = (
            frozenset(settings.detector.classes) if settings.detector.classes else None
        )
        self._min_target_range_m: float = settings.detector.min_target_range_m
        self._max_det: int = settings.detector.max_det

    # ── Public Pipeline interface ─────────────────────────────────────────────

    def start(self) -> None:
        _check_imports()
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst

        Gst.init(None)
        self._stopping.clear()

        # First bring-up happens synchronously so a hard misconfiguration (bad
        # RTSP URL, missing plugin) still fails fast at startup. Subsequent
        # failures are recovered by the supervisor instead of going dark.
        self._bring_up(Gst, GLib)
        self._supervisor = threading.Thread(
            target=self._supervise, args=(Gst, GLib), daemon=True, name="ds-supervisor")
        self._supervisor.start()

    def _bring_up(self, Gst, GLib) -> None:
        """Build the pipeline, set it PLAYING, and create (but do not run) the
        GLib loop + stall watchdog. Raises on a hard state-change failure."""
        # FRESH per-stream inference state on every (re)build — never carry it
        # across pipelines. A rebuilt nvtracker restarts track IDs from scratch,
        # so stale TrackStabilizer state would hand a NEW track that reuses an
        # old ID the dead track's identity: already past the confirm debounce,
        # hysteresis latched ON (emits below the confidence gate), EMA seeded by
        # the old boat, stale coast velocity (observed live after a detection
        # off/on toggle: 0.19-0.29 confidence targets streaming immediately).
        # last_pts must reset too or the dup-frame guard can drop every frame of
        # a new pipeline whose PTS restarts lower. Runtime /control values are
        # preserved via their mirrors (self._confidence & co). Safe to swap
        # wholesale: no probe is running here (old pipeline is torn down).
        d = self.settings.detector
        for i, cam in enumerate(self.settings.cameras):
            self._src_idx_to_name[i] = cam.name
            stab = TrackStabilizer(
                confirm_frames=d.stabilize_confirm_frames,
                max_coast_frames=d.stabilize_max_coast_frames,
                hysteresis_ratio=d.stabilize_hysteresis_ratio,
                ema_alpha=d.stabilize_ema_alpha,
                coast_velocity_factor=d.stabilize_coast_velocity_factor,
                person_confirm_frames=d.stabilize_person_confirm_frames,
                smooth=d.stabilize_smooth,
                smooth_window=d.stabilize_smooth_window,
            ) if d.stabilize else None
            with self._lock:
                self._states[cam.name] = _StreamState(
                    cam=cam, settings=self.settings, stabilizer=stab,
                    confidence=self._confidence, allowed_labels=self._allowed_labels,
                    min_target_range_m=self._min_target_range_m,
                    # Display ids use the full 10..99 pool regardless of the
                    # max-targets cap: the cap bounds how many targets are
                    # SHOWN per frame, not what they are NAMED — a small pool
                    # sized to the cap recycles numbers so fast that a freed id
                    # lands on a different vessel within seconds.
                    vel=VelocityTracker(**reid_options(d)),
                )

        self._gst = self._build_pipeline(Gst)

        bus = self._gst.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        ret = self._gst.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(
                "DeepStream pipeline failed to enter PLAYING state. "
                "Check GStreamer plugin availability and RTSP camera URLs."
            )

        self._loop = GLib.MainLoop()
        # Stall watchdog: fires in the GLib loop thread (same thread as the probe
        # that writes last_frame_at, so no extra locking). Reset each camera's
        # clock to "now" first so the RTSP preroll grace period starts here.
        now = time.time()
        for proxy in self.workers.values():
            proxy.last_frame_at = now
            # Clear any error left from a prior fault (the GStreamer error stamped
            # on every camera by _on_bus_message, or a stale "no frames" flag);
            # the pipeline is healthy again here. Historical detail is kept in
            # self._last_error / /health's pipeline_last_error.
            proxy.error = None
        GLib.timeout_add_seconds(2, self._watchdog)
        # Respect a current disable across rebuilds: if detection was toggled off,
        # a recovered pipeline must come back PAUSED, not silently resume PLAYING.
        if not self.enabled:
            try:
                self._gst.set_state(Gst.State.PAUSED)
            except Exception:  # pragma: no cover - hardware dependent
                pass
        self._log.info(
            "DeepStream pipeline %s: %d camera(s) → nvinfer(%s) → nvtracker",
            "PLAYING" if self.enabled else "PAUSED (detection disabled)",
            len(self.settings.cameras), self.settings.detector.model,
        )

    def _tear_down(self, Gst) -> None:
        """Set the current pipeline to NULL and drop the loop reference. Safe to
        call repeatedly (used between supervised restarts and on stop)."""
        if self._gst is not None:
            try:
                self._gst.set_state(Gst.State.NULL)
            except Exception:
                pass
            self._gst = None
        self._loop = None

    def _supervise(self, Gst, GLib) -> None:
        """Run the GLib loop; on a fatal error/EOS, rebuild with backoff.

        start() has already brought up the first pipeline, so the first iteration
        runs that loop. When it exits (a bus ERROR/EOS quits it via
        _on_bus_message), we tear down and — unless stop() was called — rebuild
        and try again, never giving up so a transient RTSP/decoder fault can't
        leave detection permanently offline.
        """
        backoff = _RESTART_BACKOFF_INITIAL_S
        first = True
        while not self._stopping.is_set():
            if not first:
                # Stay fully torn down while detection is disabled; enable sets
                # the event and we rebuild with fresh RTSP sessions.
                while not self._stopping.is_set() and not self._enabled_evt.wait(timeout=1.0):
                    pass
                if self._stopping.is_set():
                    break
                try:
                    self._bring_up(Gst, GLib)
                    backoff = _RESTART_BACKOFF_INITIAL_S
                except Exception as exc:  # pragma: no cover - hardware dependent
                    self._last_error = str(exc)
                    self._log.error("DeepStream bring-up failed: %s", exc)
                    for proxy in self.workers.values():
                        proxy.error = f"pipeline down: {exc}"
                    if self._stopping.wait(timeout=backoff):
                        break
                    backoff = min(backoff * 2, _RESTART_BACKOFF_MAX_S)
                    continue
            first = False

            loop = self._loop
            if loop is not None:
                try:
                    loop.run()  # blocks until quit (error / EOS / stop)
                except Exception as exc:  # pragma: no cover - hardware dependent
                    self._last_error = str(exc)
                    self._log.error("DeepStream loop error: %s", exc)

            reason = self._exit_reason
            self._exit_reason = None
            self._tear_down(Gst)
            if self._stopping.is_set():
                break

            if reason in ("detection disabled", "detection re-enabled"):
                # Deliberate teardown from the /control toggle: not a fault. No
                # restart count, no error, no backoff — the loop top gates on
                # _enabled_evt and rebuilds as soon as detection is enabled.
                self._log.info("DeepStream pipeline torn down (%s)", reason)
                continue

            if reason:
                # Watchdog-initiated rebuild (silent stall): surface it in
                # /health like any other pipeline fault.
                self._last_error = reason
            self._restart_count += 1
            self._last_restart_ts = time.monotonic()
            self._log.error(
                "DeepStream pipeline exited (restart #%d, last_error=%s); "
                "rebuilding in %.0fs", self._restart_count, self._last_error, backoff)
            if self._stopping.wait(timeout=backoff):
                break
            backoff = min(backoff * 2, _RESTART_BACKOFF_MAX_S)

    def restart_info(self) -> Dict[str, object]:
        """Auto-restart telemetry for /health: how many times the GStreamer
        pipeline has been rebuilt after a fault, the last error seen, and whether
        the most recent rebuild is recent enough to still count as degraded."""
        ts = self._last_restart_ts
        recent = ts is not None and (time.monotonic() - ts) < _RESTART_DEGRADED_WINDOW_S
        return {
            "restarts": self._restart_count,
            "last_error": self._last_error,
            "recent": recent,
        }

    def stop(self) -> None:
        self._stopping.set()
        if self._loop is not None:
            self._loop.quit()
        if self._supervisor is not None:
            self._supervisor.join(timeout=5.0)
            self._supervisor = None
        # Ensure the pipeline is released even if the supervisor never ran.
        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
            self._tear_down(Gst)
        except Exception:
            pass
        if self._dewarp_tmp is not None:
            self._dewarp_tmp.cleanup()
            self._dewarp_tmp = None
        self._log.info("DeepStream pipeline stopped")

    def set_confidence(self, value: float) -> None:
        with self._lock:
            self._confidence = value
            for st in self._states.values():
                st.confidence = value

    def set_min_target_range(self, value: float) -> None:
        with self._lock:
            self._min_target_range_m = value
            for st in self._states.values():
                st.min_target_range_m = value

    def set_labels(self, labels: Optional[list]) -> None:
        fs = frozenset(labels) if labels else None
        with self._lock:
            self._allowed_labels = fs
            for st in self._states.values():
                st.allowed_labels = fs

    def labels(self) -> Optional[list]:
        with self._lock:
            vals = {st.allowed_labels for st in self._states.values()}
        if len(vals) != 1:
            return None
        only = next(iter(vals))
        return None if only is None else sorted(only)

    def set_max_targets(self, value: int) -> None:
        with self._lock:
            self._max_det = value
            self.settings.detector.max_det = value

    def max_targets(self) -> int:
        with self._lock:
            return self._max_det

    def set_active_camera(self, name: str) -> None:
        if name in self.workers:
            self.active_camera = name

    def set_enabled(self, value: bool) -> None:
        # /control re-POSTs the current state every plugin cycle — only act on a
        # real transition, otherwise we'd tear the pipeline down once a second.
        if value == self.enabled:
            return
        self.enabled = value
        # Disable must NOT merely PAUSE the pipeline: rtspsrc stops pulling from
        # the sockets, the domes' live sessions go stale, and on resume the
        # pipeline never delivers another frame (both cameras stall permanently
        # — observed live). Instead the supervisor tears the pipeline fully down
        # (NULL: decoders + nvinfer off the GPU, RTSP sessions closed — the
        # power/thermal saving PAUSED was after) and rebuilds it from scratch
        # with fresh RTSP connections on enable.
        if value:
            self.frames.resume()
            self._enabled_evt.set()
            # If a pipeline is somehow still up (e.g. disable landed mid-rebuild
            # and left a PAUSED graph), recycle it too rather than resume stale.
            self._quit_loop("detection re-enabled")
        else:
            self.frames.pause()
            self._enabled_evt.clear()
            self._quit_loop("detection disabled")

    def _quit_loop(self, reason: str) -> None:
        """Ask the supervisor to tear down the current pipeline: record why and
        quit the GLib loop. No-op (including the reason) when no loop is up —
        a reason recorded with nothing to quit would linger and mislabel the
        NEXT genuine crash as a deliberate teardown, skipping its backoff."""
        loop = self._loop
        if loop is None:
            return
        self._exit_reason = reason
        try:
            loop.quit()
        except Exception:  # pragma: no cover - hardware dependent
            pass

    def camera_errors(self) -> Dict[str, str]:
        return {name: p.error for name, p in self.workers.items() if p.error}

    def _watchdog(self) -> bool:
        """Flag any camera that has produced no frames for STALL_TIMEOUT_S.

        Runs on the GLib loop timer. Returns True to stay scheduled; stops once
        the loop is torn down (GLib drops the source when the loop is gone).
        """
        if self._loop is None or not self._loop.is_running():
            return False
        if not self.enabled:
            # Detection disabled: the pipeline is being torn down (or sits in a
            # brief PAUSED window if the disable raced a rebuild) — silence is
            # expected, don't flag cameras or self-heal.
            return True
        now = time.time()
        stalls = {name: now - p.last_frame_at for name, p in self.workers.items()}
        for name, proxy in self.workers.items():
            if stalls[name] > _STALL_TIMEOUT_S:
                # Don't clobber a hard GStreamer error already recorded on the bus.
                if not proxy.error or proxy.error.startswith("no frames"):
                    proxy.error = (f"no frames for {int(stalls[name])}s "
                                   "(camera/RTSP stalled)")
        # Self-heal: EVERY camera silent at once is a pipeline-level fault the
        # bus never reported (stale RTSP sessions, stuck muxer) — the supervisor
        # would otherwise wait forever. Quit the loop so it rebuilds with fresh
        # RTSP connections. A single stalled camera stays flagged only:
        # rebuilding both streams won't revive a dead dome.
        if self.workers and all(s > _STALL_REBUILD_S for s in stalls.values()):
            self._log.error(
                "DeepStream watchdog: all cameras stalled > %.0fs — rebuilding pipeline",
                _STALL_REBUILD_S)
            self._quit_loop(f"all cameras stalled > {int(_STALL_REBUILD_S)}s")
            return False
        return True

    def _dewarper_config_for(self, cam: CameraConfig, w: int, h: int) -> Optional[str]:
        """Return a path to an nvdewarper config for this camera, or None.

        If the camera sets an explicit ``dewarper_config`` we use it verbatim.
        Otherwise we synthesise a single-surface *perspective* (projection-type=3)
        config from the same knobs the CPU Undistorter uses — focal length as a
        fraction of width, a single radial term ``k1``, and an image-plane roll to
        level the horizon. nvdewarper applies this on the GPU in NVMM before the
        mux, so inference and geometry run on the corrected frame.

        NOTE: the key names mirror NVIDIA's ``config_dewarper_perspective.txt``
        sample. The coefficients are the existing *eyeballed* values — verify and
        re-tune on the Jetson against a checkerboard; nvdewarper's distortion model
        is not guaranteed bit-identical to OpenCV's. Returns None when undistort is
        off so the caller leaves the camera branch uncorrected.
        """
        if not cam.undistort:
            return None
        if cam.dewarper_config:
            p = Path(cam.dewarper_config)
            if p.exists():
                return str(p)
            self._log.warning(
                "camera %s: dewarper_config %s not found; generating one",
                cam.name, cam.dewarper_config)

        if self._dewarp_tmp is None:
            self._dewarp_tmp = tempfile.TemporaryDirectory(prefix="vision-dewarp-")

        focal = cam.undistort_f_factor * w
        cfg = (
            "# Auto-generated nvdewarper perspective config — re-tune on hardware.\n"
            "# Keys follow NVIDIA's config_dewarper_perspective.txt; nvdewarper warns\n"
            "# (non-fatally) on unrecognised keys, so keep this to known-valid ones.\n"
            "[property]\n"
            f"output-width={w}\n"
            f"output-height={h}\n"
            "num-batch-buffers=1\n"
            "\n"
            "[surface0]\n"
            "projection-type=3\n"          # 3 = PerspectivePerspective
            "surface-index=0\n"
            f"width={w}\n"
            f"height={h}\n"
            # Optical center is assumed at the image centre by nvdewarper.
            f"focal-length={focal:.3f}\n"
            # Single radial term (k1); negative corrects barrel — matches Undistorter.
            f"distortion={cam.undistort_k1:.5f};0.0;0.0;0.0;0.0\n"
            # CCW image-plane rotation to level a tilted mount.
            f"roll={cam.undistort_rotation_deg:.3f}\n"
        )
        path = Path(self._dewarp_tmp.name) / f"dewarper_{cam.name}.txt"
        path.write_text(cfg)
        return str(path)

    # ── GStreamer pipeline construction ───────────────────────────────────────

    def _build_pipeline(self, Gst):
        """Create, configure, and link all GStreamer/DeepStream elements."""
        pipeline = Gst.Pipeline.new("vision-ai-ds")
        cams = self.settings.cameras
        n = len(cams)
        # nvstreammux output = display + geometry resolution (native camera res).
        # nvinfer rescales this to imgsz internally on the GPU for inference, and
        # DeepStream maps detection coords back into this WxH space, so bboxes and
        # horizon_y stay in native pixels. (Inference still happens at imgsz.)
        W = self.settings.detector.mux_width
        H = self.settings.detector.mux_height

        # Exactly one detection model runs at a time, selected by detector.model.
        model = self.settings.detector.model
        cfg_name = MODEL_PGIE_CONFIG.get(model)
        if cfg_name is None:
            raise ValueError(
                f"detector.model={model!r} is not supported. "
                f"Choose one of: {sorted(MODEL_PGIE_CONFIG)}."
            )
        pgie_cfg = str(_DEEPSTREAM_DIR / cfg_name)
        if not Path(pgie_cfg).exists():
            raise FileNotFoundError(
                f"nvinfer config for model {model!r} not found at {pgie_cfg}."
            )
        tracker_cfg = _tracker_cfg_path()

        def make(factory: str, name: str, **props):
            el = Gst.ElementFactory.make(factory, name)
            if el is None:
                raise RuntimeError(
                    f"Cannot create GStreamer element '{factory}' (name={name!r}). "
                    "Verify DeepStream 7.x is installed and GST_PLUGIN_PATH is set."
                )
            for k, v in props.items():
                el.set_property(k.replace("_", "-"), v)
            pipeline.add(el)
            return el

        # nvstreammux: batches all camera feeds into a single NVMM buffer.
        # batched-push-timeout (µs): how long to wait for a full batch before
        # pushing a partial one. Kept LONGER than the camera frame interval so the
        # muxer waits for real frames rather than timing out early and repeating
        # the last frame to fill the batch (which would push output above the input
        # rate). The probe also drops any residual PTS repeat as a hard guard.
        push_timeout_us = max(self.settings.detector.mux_batch_timeout_ms, 1) * 1000
        mux = make("nvstreammux", "mux",
                   batch_size=n, width=W, height=H,
                   batched_push_timeout=push_timeout_us,
                   live_source=1,
                   enable_padding=1)  # letterbox: preserve aspect ratio with black bars

        for i, cam in enumerate(cams):
            if not cam.url:
                raise ValueError(
                    f"camera {cam.name!r}: RTSP URL is required for DeepStream mode. "
                    f"Set via VISION_CAMERA_{cam.name.upper()}_URL or url: in deepstream.yaml."
                )

            # All Hikvision domes use H.264; add H.265 support by extending
            # CameraConfig with a `codec` field if needed.
            depay = make("rtph264depay", f"depay{i}")
            parser = make("h264parse", f"parse{i}")
            decoder = make("nvv4l2decoder", f"dec{i}")

            # GPU lens correction (optional, in NVMM): nvdewarper straightens the
            # frame before the mux, so inference + geometry see the corrected image
            # (same as undistort_before_detect). It needs RGBA in/out; without it we
            # stay on the cheaper NV12 path. If the element can't be created we fall
            # back to no-dewarp here; note a *bad config* instead surfaces later as a
            # bus error at PLAYING (nvdewarper validates the config at state change).
            dewarp = None
            dewarp_cfg = self._dewarper_config_for(cam, W, H)
            if dewarp_cfg:
                try:
                    dewarp = make("nvdewarper", f"dewarp{i}",
                                  config_file=dewarp_cfg,
                                  source_id=i,
                                  num_batch_buffers=1)
                    self._log.info("camera %s: GPU dewarp via nvdewarper (%s)",
                                   cam.name, dewarp_cfg)
                except Exception as exc:
                    dewarp = None
                    self._log.warning(
                        "camera %s: nvdewarper unavailable (%s); running uncorrected",
                        cam.name, exc)

            fmt = "RGBA" if dewarp is not None else "NV12"
            conv = make("nvvideoconvert", f"conv{i}")
            caps = make("capsfilter", f"caps{i}")
            caps.set_property(
                "caps",
                Gst.Caps.from_string(f"video/x-raw(memory:NVMM),format={fmt}"),
            )
            # Queue between converter and mux: decouples camera timing so one
            # slow camera does not stall the other.
            que = make("queue", f"q{i}")
            que.set_property("max-size-buffers", 4)
            que.set_property("leaky", 2)  # GST_QUEUE_LEAK_DOWNSTREAM — drop oldest

            # Static links: depay → parse → decode → conv → caps [→ dewarp] → queue
            chain = [(depay, parser), (parser, decoder), (decoder, conv), (conv, caps)]
            chain += [(caps, dewarp), (dewarp, que)] if dewarp is not None else [(caps, que)]
            for src_el, dst_el in chain:
                if not src_el.link(dst_el):
                    raise RuntimeError(
                        f"Failed to link {src_el.get_name()} → {dst_el.get_name()}"
                    )

            # queue → mux.sink_N (request pad)
            mux_sink = mux.get_request_pad(f"sink_{i}")
            q_src = que.get_static_pad("src")
            if q_src.link(mux_sink) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Failed to link q{i}.src → mux.sink_{i}")

            # rtspsrc produces dynamic pads; connect via signal once the stream
            # prerolls (usually within 1–2 seconds of PLAYING).
            src = make("rtspsrc", f"src{i}",
                       location=cam.url,
                       latency=50)
            src.set_property("protocols", 4)          # GST_RTSP_LOWER_TRANS_TCP
            src.set_property("drop-on-latency", True)
            src.connect("pad-added", self._on_src_pad_added, depay)

        # nvinfer: TensorRT inference. Reads the batched NVMM buffer directly —
        # no host-side copy. Uses custom deepstream-yolo parser for YOLOv8 output.
        # The config (and thus the model) is selected by detector.model above.
        pgie = make("nvinfer", "pgie",
                    config_file_path=pgie_cfg,
                    batch_size=n)
        self._log.info("nvinfer model=%s config=%s", model, cfg_name)

        # nvtracker: NvDCF GPU tracker. Assigns stable object_id (track ID) to
        # each detection across frames. Runs entirely on the GPU.
        # Tracker works at its own internal resolution (GPU cost scales with it);
        # keep it decoupled from the larger native display res. 640x384 matches
        # the inference scale and keeps NvDCF cheap. Coords are still reported in
        # the streammux WxH space, so this does not affect bbox geometry.
        tracker = make("nvtracker", "tracker",
                       ll_lib_file=_TRACKER_LIB,
                       ll_config_file=tracker_cfg,
                       tracker_width=self.settings.detector.imgsz,
                       tracker_height=384,  # power-of-2 height for GPU efficiency
                       # 0 = don't let the tracker stamp "<label> <id>" object text
                       # onto the OSD: we draw every label ourselves from the
                       # stabilized event (draw_event), so this would double up.
                       display_tracking_id=0)

        # ── Zero-copy NVMM display + HW JPEG path ─────────────────────────────
        # Everything from here stays in NVMM until nvjpegenc emits compressed
        # bytes: the frame pixels never round-trip to host. nvvideoconvert lifts
        # the tracker's NV12 to RGBA (needed by the per-camera nvdsosd, and what
        # the probe maps for the throttled auto-horizon read); the probe attaches
        # display meta per frame; nvstreamdemux splits the batch into per-camera
        # buffers; each branch has its OWN nvdsosd (a single OSD before the demux
        # renders every source's meta onto one surface — boxes/HUD leak between
        # cameras), then converts to NVMM I420 and HW-encodes to JPEG.
        dispconv = make("nvvideoconvert", "dispconv")
        dispcaps = make("capsfilter", "dispcaps")
        dispcaps.set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"),
        )
        demux = make("nvstreamdemux", "demux")

        # Main batched chain: mux → pgie → tracker → dispconv → dispcaps(RGBA)
        #                     → nvstreamdemux
        for src_el, dst_el in [
            (mux, pgie), (pgie, tracker),
            (tracker, dispconv), (dispconv, dispcaps), (dispcaps, demux),
        ]:
            if not src_el.link(dst_el):
                raise RuntimeError(
                    f"Failed to link {src_el.get_name()} → {dst_el.get_name()}"
                )

        # Probe on the RGBA capsfilter src pad: fires after tracking + conversion
        # and BEFORE the demux, so the display meta it attaches to each frame_meta
        # travels with that frame to its per-camera nvdsosd. It reads NvDsObjectMeta
        # (small) to build events and, at most once per _HORIZON_REFRESH_S, maps the
        # RGBA surface to re-detect the horizon.
        probe_pad = dispcaps.get_static_pad("src")
        probe_pad.add_probe(Gst.PadProbeType.BUFFER, self._probe_callback, None)

        # Per-camera tail: demux.src_N → queue → nvdsosd → nvvideoconvert → NVMM
        # I420 → nvjpegenc → appsink. The per-camera nvdsosd renders only that
        # source's display meta; appsink hands finished JPEG bytes to the camera's
        # LatestFrame via _on_jpeg_sample (generic GObject signals, so the GstApp
        # typelib is not required).
        for i, cam in enumerate(cams):
            que = make("queue", f"encq{i}")
            que.set_property("max-size-buffers", 2)
            que.set_property("leaky", 2)  # drop oldest: MJPEG is latest-frame-wins
            osd = make("nvdsosd", f"osd{i}")
            osd.set_property("process-mode", 1)  # 1 = GPU (cairo) — supports text
            jconv = make("nvvideoconvert", f"jconv{i}")
            jcaps = make("capsfilter", f"jcaps{i}")
            jcaps.set_property(
                "caps",
                Gst.Caps.from_string("video/x-raw(memory:NVMM),format=I420"),
            )
            enc = make("nvjpegenc", f"nvjpegenc{i}",
                       quality=self.settings.server.jpeg_quality)
            appsink = make("appsink", f"appsink{i}",
                           emit_signals=True, sync=False,
                           max_buffers=1, drop=True)
            appsink.connect("new-sample", self._on_jpeg_sample, cam.name)

            demux_src = demux.get_request_pad(f"src_{i}")
            q_sink = que.get_static_pad("sink")
            if demux_src.link(q_sink) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Failed to link demux.src_{i} → encq{i}")
            for src_el, dst_el in [(que, osd), (osd, jconv), (jconv, jcaps),
                                   (jcaps, enc), (enc, appsink)]:
                if not src_el.link(dst_el):
                    raise RuntimeError(
                        f"Failed to link {src_el.get_name()} → {dst_el.get_name()}"
                    )

        return pipeline

    def _on_src_pad_added(self, src_elem, new_pad, depay) -> None:
        """Link the rtspsrc dynamic pad to the depayloader sink.

        rtspsrc may emit multiple pads (audio + video). We only link the one
        that carries RTP video.
        """
        sink = depay.get_static_pad("sink")
        if sink.is_linked():
            return
        caps = new_pad.get_current_caps() or new_pad.query_caps(None)
        if not caps:
            return
        struct_name = caps.get_structure(0).get_name()
        if not struct_name.startswith("application/x-rtp"):
            return
        media = caps.get_structure(0).get_string("media") or ""
        if media and media != "video":
            return
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        ret = new_pad.link(sink)
        if ret != Gst.PadLinkReturn.OK:
            self._log.warning(
                "rtspsrc pad link returned %s for %s", ret, src_elem.get_name()
            )

    # ── appsink callback: finished HW-JPEG bytes per camera ──────────────────

    def _on_jpeg_sample(self, appsink, cam_name):
        """Pull a HW-encoded JPEG from a per-camera appsink and store it.

        Runs in the appsink's streaming thread. The bytes are the only thing that
        ever leaves NVMM on the display path; LatestFrame.set() is a no-op while
        paused (detection disabled), so a frame can't resurface after a disable.
        """
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        gbuf = sample.get_buffer()
        ok, info = gbuf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK
        try:
            jpeg = bytes(info.data)
        finally:
            gbuf.unmap(info)
        if jpeg:
            self.frames.set(cam_name, jpeg)
            proxy = self.workers.get(cam_name)
            if proxy is not None:
                proxy.last_frame_at = time.time()
                if proxy.error and proxy.error.startswith("no frames"):
                    proxy.error = None  # recovered from a stall
        return Gst.FlowReturn.OK

    # ── Probe callback (GLib thread, every batch buffer) ─────────────────────

    def _probe_callback(self, pad, info, user_data):
        """Extract detections from NvDsObjectMeta and emit DetectionEvents.

        Called in the GLib main loop thread for every batch pushed by nvtracker.
        Only NvDsObjectMeta structs (small) are read — no pixel data is copied
        to host until the display path below (one copy per camera, after inference).

        Returns Gst.PadProbeReturn.OK (= 1) to let the buffer continue.
        """
        try:
            import pyds  # type: ignore[import]
        except ImportError:
            return 1  # Gst.PadProbeReturn.OK

        if not self.enabled:
            return 1

        gst_buf = info.get_buffer()
        if gst_buf is None:
            return 1

        try:
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buf))
        except Exception:
            return 1
        if batch_meta is None:
            return 1

        t0 = time.perf_counter()

        # Sentinel for untracked objects (pyds.UNTRACKED_OBJECT_ID on DS 7.x)
        untracked = getattr(pyds, "UNTRACKED_OBJECT_ID", 0xFFFF_FFFF_FFFF_FFFF)

        # Active model selects the class map (only one model runs at a time).
        model = self.settings.detector.model

        with self._lock:
            conf_thresh = self._confidence
            allowed = self._allowed_labels
            max_det = self._max_det

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            src_idx = frame_meta.source_id
            cam_name = self._src_idx_to_name.get(src_idx)
            if cam_name is None:
                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break
                continue

            state = self._states.get(cam_name)
            if state is None:
                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break
                continue

            # Respect the input frame rate as the ceiling: if the muxer handed us a
            # frame whose PTS is not newer than the last one we processed for this
            # camera, it is a repeat (the camera produced nothing new) — skip it so
            # we never emit more events/MJPEG frames than the camera actually sent.
            pts = frame_meta.buf_pts
            if pts != _GST_CLOCK_TIME_NONE and pts <= state.last_pts:
                state.dup_skipped += 1
                if state.dup_skipped % 200 == 1:
                    self._log.debug("camera %s: dropped %d muxer frame repeat(s)",
                                    cam_name, state.dup_skipped)
                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break
                continue
            if pts != _GST_CLOCK_TIME_NONE:
                state.last_pts = pts

            state.seq += 1
            # Detection coords + display surface are in the streammux (native)
            # resolution, NOT imgsz — see _build_pipeline. Geometry (bearing,
            # range, horizon_y) and the bbox overlay all key off this WxH.
            W = self.settings.detector.mux_width
            H = self.settings.detector.mux_height

            # ── Extract raw detections from NvDsObjectMeta ─────────────────
            # NvDsObjectMeta contains the bbox, class_id, confidence, and
            # tracker object_id populated by nvinfer + nvtracker. Reading this
            # struct does NOT touch frame pixel data — it is metadata-only.
            raw_tracks: List[RawTrack] = []
            active_ids: set = set()

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                r = obj.rect_params
                # Suppress nvdsosd's default per-object box AND text: we draw every
                # box/label ourselves from the stabilized event (which also includes
                # coasted tracks that have no obj meta), so the raw nvinfer/tracker
                # rect+label must not be rendered too. border_width=0 drops the box;
                # blanking display_text + bg drops the "<label> <id>" stamp.
                r.border_width = 0
                obj.text_params.display_text = ""
                obj.text_params.set_bg_clr = 0
                cls_id = int(obj.class_id)
                conf = float(obj.confidence)
                tid_raw = int(obj.object_id)
                tid = None if tid_raw == untracked else tid_raw
                lbl, cls_id = label_for_model(model, cls_id)
                vx = vy = 0.0
                age = 0
                disp = tid

                if tid is not None:
                    # Waterline re-id first: a partial re-detection (hull only)
                    # of a known target continues that track instead of minting
                    # a new id. Velocity is anchored at the bbox bottom-center
                    # (the waterline), which stays put when the detected extent
                    # flips partial <-> full (see VelocityTracker.update).
                    tid = state.vel.resolve(
                        tid, state.seq, r.left, r.top, r.width, r.height, lbl)
                    cx = r.left + r.width / 2.0
                    vx, vy, age = state.vel.update(
                        tid, state.seq, cx, r.top + r.height)
                    disp = state.vel.display_id(tid)
                    active_ids.add(tid)

                raw_tracks.append(RawTrack(
                    track_id=disp,
                    cls=cls_id,
                    label=lbl,
                    confidence=conf,
                    x=r.left, y=r.top, w=r.width, h=r.height,
                    vx=vx, vy=vy,
                    age_frames=age,
                ))

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            state.vel.prune(active_ids, state.seq)

            # ── Horizon (throttled, zero-copy friendly) ─────────────────────
            # Explicit calibration wins. Otherwise auto-horizon needs host pixels,
            # so we map the RGBA surface at most once per _HORIZON_REFRESH_S and
            # cache the result; every other frame reuses the cached value and the
            # pipeline stays copy-free.
            horizon_y = self._horizon_for(state, pyds, gst_buf, frame_meta)

            # ── Stabilizer (Python, per-camera) ────────────────────────────
            # NvDCF already provides track IDs; the Python stabilizer adds
            # confidence hysteresis and coasting on top.
            if state.stabilizer is not None:
                raw_tracks = state.stabilizer.update(raw_tracks, state.seq, conf_thresh)

            # ── Build DetectionEvent ────────────────────────────────────────
            latency_ms = (time.perf_counter() - t0) * 1000.0
            event = self._build_event(
                state, raw_tracks, W, H,
                conf_thresh, allowed, max_det,
                latency_ms, frame_meta.frame_num,
                horizon_y=horizon_y,
            )

            self.events.publish(event.model_dump(mode="json"))

            # ── Overlay on the GPU ───────────────────────────────────────────
            # Attach display meta for nvdsosd (downstream) to burn onto the NVMM
            # surface. No pixels touched here; the encoded JPEG arrives later on
            # the per-camera appsink (_on_jpeg_sample), which updates last_frame_at.
            draw_event(pyds, batch_meta, frame_meta, event)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return 1  # Gst.PadProbeReturn.OK

    # ── Horizon (throttled host-pixel access) ──────────────────────────────────

    def _horizon_for(self, state, pyds, gst_buf, frame_meta) -> Optional[float]:
        """Horizon line for this frame. Explicit calibration wins; otherwise
        auto-detect from the RGBA surface, but only once per _HORIZON_REFRESH_S
        (the cached value is reused in between so the per-frame path is copy-free).

        This is the ONLY host pixel access left on the display path; it runs about
        once a second per camera, not per frame, so the zero-copy property holds.
        """
        cam = state.cam
        if cam.horizon_y is not None:
            return cam.horizon_y
        if not self.settings.geometry.auto_horizon:
            return None
        now = time.time()
        if (state.horizon_y_cached is not None
                and now - state.last_horizon_t < _HORIZON_REFRESH_S):
            return state.horizon_y_cached

        # Stamp the throttle now, before attempting the map, so a persistent
        # failure backs off to ~1/s too instead of retrying (and logging) every
        # frame; the cached value (if any) is preserved on error.
        state.last_horizon_t = now
        try:
            import numpy as np
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buf), frame_meta.batch_id)
            rgba = np.array(n_frame, copy=True)            # cache-coherent on Jetson
            bgr = rgba[:, :, :3][:, :, ::-1].copy()        # RGBA → BGR, drop alpha
            state.horizon_y_cached = detect_horizon_y(bgr)
        except Exception as exc:
            self._log.debug("horizon refresh failed (%s): %s", cam.name, exc)
        finally:
            # Release the mapping (Jetson leaks NvBufSurface maps otherwise).
            _unmap = getattr(pyds, "unmap_nvds_buf_surface", None)
            if _unmap is not None:
                try:
                    _unmap(hash(gst_buf), frame_meta.batch_id)
                except Exception:
                    pass
        return state.horizon_y_cached

    # ── Event construction ────────────────────────────────────────────────────

    def _build_event(
        self,
        state: _StreamState,
        tracks: List[RawTrack],
        W: int, H: int,
        conf_thresh: float,
        allowed: Optional[frozenset],
        max_det: int,
        latency_ms: float,
        frame_num: int,
        horizon_y: Optional[float] = None,
    ) -> DetectionEvent:
        cam = state.cam

        # Horizon is resolved by the caller (_horizon_for): explicit calibration,
        # else a throttled auto-detect, else None.
        calib = (
            CalibrationStatus.ok if cam.horizon_y is not None
            else CalibrationStatus.auto if horizon_y is not None
            else CalibrationStatus.uncalibrated
        )

        frame_area = float(W * H) or 1.0
        max_area = self.settings.detector.max_area_frac

        targets: List[Target] = []
        for tr in tracks:
            # When stabilizer is off, apply the plain threshold gate here.
            if state.stabilizer is None and tr.confidence < conf_thresh:
                continue
            if allowed is not None and tr.label not in allowed:
                continue
            # person is EXEMPT from the oversized-box drop: a man-overboard close
            # to the hull legitimately fills much of the frame, and dropping it
            # before the is_person_in_water classification below would lose the
            # most safety-critical detection (mirrors pipeline.py and the
            # person-exempt min-range filter).
            if tr.label != "person" and (tr.w * tr.h) / frame_area > max_area:
                continue

            brg = estimate_bearing(tr, cam, W)
            rng, method, rconf = estimate_range(
                tr, cam, self.settings.geometry, W, H, horizon_y)

            # Minimum-range gate (own-hull / very-near clutter), applied EARLY so
            # neither the event nor the overlay shows a too-close object. person is
            # exempt (MOB must be seen up close); unknown range is kept. The value
            # is owned by the SignalK plugin (detector.minTargetRangeM via /control).
            if (state.min_target_range_m > 0 and tr.label != "person"
                    and rng is not None and 0 < rng < state.min_target_range_m):
                continue

            piw = is_person_in_water(tr.label, tr.y + tr.h, horizon_y)
            targets.append(Target(
                track_id=tr.track_id,
                label=tr.label,
                coco_class=tr.cls,
                confidence=tr.confidence,
                bbox=BBox(x=tr.x, y=tr.y, w=tr.w, h=tr.h),
                is_person_in_water=piw,
                geometry=Geometry(
                    relative_bearing_deg=brg,
                    range_m=rng,
                    range_method=RangeMethod(method) if method else None,
                    range_confidence=rconf,
                ),
                pixel_velocity=PixelVelocity(vx=tr.vx, vy=tr.vy),
                age_frames=tr.age_frames,
                coasting=tr.coasting,
            ))

        targets = _drop_contained_targets(targets, self.settings.detector.contained_frac)
        targets = cap_targets_sticky(
            targets, max_det, state.emitted_ids,
            self.settings.detector.max_det_sticky_margin)
        state.emitted_ids = {t.track_id for t in targets if t.track_id is not None}

        return DetectionEvent(
            camera=cam.name,
            timestamp=_now_iso(),
            frame_seq=frame_num,
            frame_size=FrameSize(w=W, h=H),
            horizon_y=horizon_y,
            inference=Inference(backend=Backend.deepstream, latency_ms=latency_ms),
            calibration_status=calib,
            targets=targets,
        )

    # ── GLib bus message handler ──────────────────────────────────────────────

    def _on_bus_message(self, bus, message) -> None:
        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except ImportError:
            return

        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self._log.error(
                "DeepStream GStreamer error: %s\n  debug: %s", err.message, debug)
            self._last_error = err.message
            for proxy in self.workers.values():
                proxy.error = f"GStreamer error: {err.message}"
            # Quitting the loop hands control to the supervisor, which rebuilds the
            # pipeline with backoff instead of leaving detection down (see _supervise).
            if self._loop:
                self._loop.quit()
        elif t == Gst.MessageType.WARNING:
            warn, _ = message.parse_warning()
            self._log.warning("DeepStream GStreamer warning: %s", warn.message)
        elif t == Gst.MessageType.EOS:
            self._log.info("DeepStream pipeline EOS")
            if self._loop:
                self._loop.quit()
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self._gst:
                _old, new, _pending = message.parse_state_changed()
                self._log.debug(
                    "Pipeline state → %s",
                    Gst.Element.state_get_name(new),
                )
