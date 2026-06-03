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
        → [pad probe]         # reads NvDsObjectMeta (metadata only — no pixel copy)
        → fakesink

Display path (one CPU copy per camera per displayed frame, after inference):

    probe  →  pyds.get_nvds_buf_surface()  →  np.array() copy to CPU
           →  annotate()  →  encode_jpeg()  →  LatestFrame

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

import logging
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .api.overlay import annotate, encode_jpeg
from .config import CameraConfig, Settings
from .detector.base import RawTrack
from .detector.classmap import is_person_in_water, label_for
from .detector.stabilizer import TrackStabilizer
from .detector.tracker import VelocityTracker
from .geometry import detect_horizon_y, estimate_bearing, estimate_range
from .pipeline import _drop_contained_targets  # shared geometry filter, same package
from .schemas import (
    Backend, BBox, CalibrationStatus, DetectionEvent, FrameSize,
    Geometry, Inference, PixelVelocity, RangeMethod, Target,
)
from .util import EventBuffer, LatestFrame

_STALL_TIMEOUT_S = 5.0  # mirror pipeline.py: flag a camera with no frames this long
_GST_CLOCK_TIME_NONE = 0xFFFF_FFFF_FFFF_FFFF  # GStreamer "invalid timestamp" sentinel
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
        from gi.repository import Gst  # noqa: F401
        import pyds  # noqa: F401  # type: ignore[import]
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
    # Last processed buffer PTS — used to drop muxer frame repeats so output never
    # exceeds the camera's real frame rate. dup_skipped counts those drops.
    last_pts: int = -1
    dup_skipped: int = 0


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
        self._loop_thread: Optional[threading.Thread] = None

        # Holds generated nvdewarper config files; cleaned up on stop().
        self._dewarp_tmp: Optional[tempfile.TemporaryDirectory] = None

        # Runtime-adjustable via /control; guarded by _lock because they are
        # written from FastAPI/uvicorn threads and read from the GLib probe thread.
        self._lock = threading.Lock()
        self._confidence: float = settings.detector.confidence
        self._allowed_labels: Optional[frozenset] = (
            frozenset(settings.detector.classes) if settings.detector.classes else None
        )
        self._max_det: int = settings.detector.max_det

    # ── Public Pipeline interface ─────────────────────────────────────────────

    def start(self) -> None:
        _check_imports()
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst, GLib

        Gst.init(None)

        d = self.settings.detector
        for i, cam in enumerate(self.settings.cameras):
            self._src_idx_to_name[i] = cam.name
            stab = TrackStabilizer(
                confirm_frames=d.stabilize_confirm_frames,
                max_coast_frames=d.stabilize_max_coast_frames,
                hysteresis_ratio=d.stabilize_hysteresis_ratio,
                ema_alpha=d.stabilize_ema_alpha,
                coast_velocity_factor=d.stabilize_coast_velocity_factor,
            ) if d.stabilize else None
            self._states[cam.name] = _StreamState(
                cam=cam, settings=self.settings, stabilizer=stab,
                confidence=self._confidence, allowed_labels=self._allowed_labels,
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
        GLib.timeout_add_seconds(2, self._watchdog)
        self._loop_thread = threading.Thread(
            target=self._loop.run, daemon=True, name="glib-main")
        self._loop_thread.start()
        self._log.info(
            "DeepStream pipeline PLAYING: %d camera(s) → nvinfer → nvtracker",
            len(self.settings.cameras),
        )

    def stop(self) -> None:
        if self._loop is not None:
            self._loop.quit()
        if self._gst is not None:
            try:
                import gi
                gi.require_version("Gst", "1.0")
                from gi.repository import Gst
                self._gst.set_state(Gst.State.NULL)
            except Exception:
                pass
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=3.0)
        if self._dewarp_tmp is not None:
            self._dewarp_tmp.cleanup()
            self._dewarp_tmp = None
        self._log.info("DeepStream pipeline stopped")

    def set_confidence(self, value: float) -> None:
        with self._lock:
            self._confidence = value
            for st in self._states.values():
                st.confidence = value

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
        self.enabled = value
        if value:
            self.frames.resume()
        else:
            self.frames.pause()

    def camera_errors(self) -> Dict[str, str]:
        return {name: p.error for name, p in self.workers.items() if p.error}

    def _watchdog(self) -> bool:
        """Flag any camera that has produced no frames for STALL_TIMEOUT_S.

        Runs on the GLib loop timer. Returns True to stay scheduled; stops once
        the loop is torn down (GLib drops the source when the loop is gone).
        """
        if self._loop is None or not self._loop.is_running():
            return False
        now = time.time()
        for proxy in self.workers.values():
            stalled = now - proxy.last_frame_at
            if stalled > _STALL_TIMEOUT_S:
                # Don't clobber a hard GStreamer error already recorded on the bus.
                if not proxy.error or proxy.error.startswith("no frames"):
                    proxy.error = (f"no frames for {int(stalled)}s "
                                   "(camera/RTSP stalled)")
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

        pgie_cfg = str(_DEEPSTREAM_DIR / "pgie_yolov8n.txt")
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
        pgie = make("nvinfer", "pgie",
                    config_file_path=pgie_cfg,
                    batch_size=n)

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
                       display_tracking_id=1)

        # Display conversion: the probe extracts pixels with
        # pyds.get_nvds_buf_surface(), which requires an RGBA surface. The
        # tracker emits NV12, so insert nvvideoconvert → RGBA (kept in NVMM;
        # get_nvds_buf_surface maps it to the CPU on Jetson). Without this stage
        # the surface read fails and the MJPEG/snapshot frames come out blank.
        dispconv = make("nvvideoconvert", "dispconv")
        dispcaps = make("capsfilter", "dispcaps")
        dispcaps.set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"),
        )

        # fakesink: swallows frames after conversion; we consume everything via
        # the probe on the converter's src pad.
        sink = make("fakesink", "fsink")
        sink.set_property("async", False)
        sink.set_property("sync", False)

        # Main chain: mux → pgie → tracker → dispconv → dispcaps(RGBA) → sink
        for src_el, dst_el in [
            (mux, pgie), (pgie, tracker),
            (tracker, dispconv), (dispconv, dispcaps), (dispcaps, sink),
        ]:
            if not src_el.link(dst_el):
                raise RuntimeError(
                    f"Failed to link {src_el.get_name()} → {dst_el.get_name()}"
                )

        # Probe on the RGBA capsfilter src pad: fires after tracking + conversion.
        # This is where we read NvDsObjectMeta and extract the RGBA frame pixels
        # for MJPEG annotation.
        probe_pad = dispcaps.get_static_pad("src")
        probe_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            self._probe_callback,
            None,
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
                cls_id = int(obj.class_id)
                conf = float(obj.confidence)
                tid_raw = int(obj.object_id)
                tid = None if tid_raw == untracked else tid_raw
                lbl = label_for(cls_id)
                vx = vy = 0.0
                age = 0

                if tid is not None:
                    cx = r.left + r.width / 2.0
                    cy = r.top + r.height / 2.0
                    vx, vy, age = state.vel.update(tid, state.seq, cx, cy)
                    active_ids.add(tid)

                raw_tracks.append(RawTrack(
                    track_id=tid,
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

            # ── Display frame: one CPU copy per camera, after inference ─────
            # pyds.get_nvds_buf_surface() returns a numpy array view of the NVMM
            # RGBA surface. On Jetson unified memory, np.array(…, copy=True) is a
            # cache-coherent access (not a DMA transfer) — still far cheaper than
            # the pipeline.py path that copied BGR frames BEFORE inference.
            disp_img = None
            try:
                import numpy as np
                n_frame = pyds.get_nvds_buf_surface(hash(gst_buf), frame_meta.batch_id)
                frame_rgba = np.array(n_frame, copy=True)  # NVMM → CPU
                # RGBA → BGR: reverse first 3 channels (drop alpha)
                disp_img = frame_rgba[:, :, :3][:, :, ::-1].copy()
            except Exception as exc:
                self._log.debug("display frame extraction failed (%s): %s", cam_name, exc)
            finally:
                # Release the surface mapping (Jetson leaks NvBufSurface maps across
                # frames otherwise); no-op if this pyds build lacks the symbol.
                _unmap = getattr(pyds, "unmap_nvds_buf_surface", None)
                if _unmap is not None:
                    try:
                        _unmap(hash(gst_buf), frame_meta.batch_id)
                    except Exception:
                        pass

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
                disp_img=disp_img,
            )

            self.events.publish(event.model_dump(mode="json"))

            # ── Annotate + store MJPEG ──────────────────────────────────────
            if disp_img is not None:
                jpeg = encode_jpeg(
                    annotate(disp_img, event), self.settings.server.jpeg_quality)
                if jpeg:
                    self.frames.set(cam_name, jpeg)

            if cam_name in self.workers:
                proxy = self.workers[cam_name]
                proxy.last_frame_at = time.time()
                if proxy.error and proxy.error.startswith("no frames"):
                    proxy.error = None  # recovered from a stall

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return 1  # Gst.PadProbeReturn.OK

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
        disp_img=None,
    ) -> DetectionEvent:
        cam = state.cam

        # Horizon: prefer explicit calibration, fall back to auto-detect
        # (auto-detect needs the display frame; skip it if unavailable).
        horizon_y: Optional[float] = cam.horizon_y
        if horizon_y is None and self.settings.geometry.auto_horizon and disp_img is not None:
            horizon_y = detect_horizon_y(disp_img)

        calib = (
            CalibrationStatus.ok if cam.horizon_y is not None
            else CalibrationStatus.auto if horizon_y is not None
            else CalibrationStatus.uncalibrated
        )

        frame_area = float(W * H) or 1.0
        max_area = self.settings.detector.max_area_frac
        own_hull_min = self.settings.detector.own_hull_min_range_m

        targets: List[Target] = []
        for tr in tracks:
            # When stabilizer is off, apply the plain threshold gate here.
            if state.stabilizer is None and tr.confidence < conf_thresh:
                continue
            if allowed is not None and tr.label not in allowed:
                continue
            if (tr.w * tr.h) / frame_area > max_area:
                continue

            brg = estimate_bearing(tr, cam, W)
            rng, method, rconf = estimate_range(
                tr, cam, self.settings.geometry, W, H, horizon_y)

            if tr.label == "vessel" and rng is not None and 0 < rng < own_hull_min:
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
        targets = sorted(targets, key=lambda t: t.confidence, reverse=True)[:max_det]

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
            for proxy in self.workers.values():
                proxy.error = f"GStreamer error: {err.message}"
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
