"""Per-camera processing pipeline: read -> detect+track -> geometry -> emit.

Each camera runs in its own daemon thread at a capped cadence. Output goes to
two decoupled buffers: the detection :class:`EventBuffer` (WebSocket) and the
:class:`LatestFrame` annotated-JPEG store (MJPEG). The API layer reads from the
buffers and never blocks the inference loop.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from .api.overlay import annotate, encode_jpeg
from .camera import create_source
from .config import CameraConfig, Settings
from .detector import create_detector
from .detector.classmap import is_person_in_water
from .geometry import detect_horizon_y, estimate_bearing, estimate_range
from .schemas import (
    BBox, Backend, CalibrationStatus, DetectionEvent, FrameSize, Geometry,
    Inference, PixelVelocity, RangeMethod, Target,
)
from .util import EventBuffer, LatestFrame


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# A camera whose source returns no frames for this long is reported as an error
# in /health, instead of silently disappearing (read()==None sets no error).
STALL_TIMEOUT_S = 5.0


class CameraWorker(threading.Thread):
    def __init__(self, cam: CameraConfig, settings: Settings,
                 events: EventBuffer, frames: LatestFrame, logger, detector):
        super().__init__(daemon=True, name=f"cam-{cam.name}")
        self._cam = cam
        self._settings = settings
        self._events = events
        self._frames = frames
        self._log = logger
        # One detector instance is shared across all camera workers (single
        # model / CUDA context); it isolates tracker state per camera name.
        self._detector = detector
        self._stop = threading.Event()
        self._source = None
        # Runtime-adjustable via /control.
        self.confidence = settings.detector.confidence
        self.error: str | None = None

    def stop(self) -> None:
        self._stop.set()

    def close_source(self) -> None:
        """Release the capture device; safe to call from another thread once the
        worker loop has exited (used as a fallback if join() times out)."""
        if self._source is not None:
            try:
                self._source.close()
            except Exception:  # pragma: no cover
                pass
            self._source = None

    def run(self) -> None:
        if self._detector is None:
            self.error = "detector unavailable (init failed)"
            self._log.error("camera %s: no detector", self._cam.name)
            return
        try:
            self._source = create_source(self._cam, self._settings)
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            self.error = f"init failed: {exc}"
            self._log.error("camera %s init failed: %s", self._cam.name, exc)
            return

        period = 1.0 / max(self._settings.server.target_fps, 0.1)
        backend = Backend(self._detector.backend)
        stalled_since: float | None = None
        try:
            while not self._stop.is_set():
                t0 = time.perf_counter()
                try:
                    frame = self._source.read()
                    if frame is None:
                        # No frame: flag a sustained stall so a wedged RTSP feed
                        # shows up in /health rather than vanishing silently.
                        if stalled_since is None:
                            stalled_since = t0
                        elif t0 - stalled_since > STALL_TIMEOUT_S:
                            self.error = (f"no frames for {int(t0 - stalled_since)}s "
                                          "(camera/RTSP stalled)")
                        time.sleep(0.1)
                        continue
                    stalled_since = None

                    tracks = self._detector.detect_and_track(frame, self._cam.name)
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    event = self._build_event(frame, tracks, backend, latency_ms)

                    payload = event.model_dump(mode="json")
                    self._events.publish(payload)
                    jpeg = encode_jpeg(annotate(frame.image, event),
                                       self._settings.server.jpeg_quality)
                    if jpeg:
                        self._frames.set(self._cam.name, jpeg)
                    self.error = None
                except Exception as exc:
                    # One bad frame must not kill the camera; log and carry on.
                    self.error = f"frame error: {exc}"
                    self._log.error("camera %s frame error: %s", self._cam.name, exc)
                    time.sleep(0.2)

                dt = time.perf_counter() - t0
                if dt < period:
                    time.sleep(period - dt)
        finally:
            self.close_source()

    def _resolve_horizon(self, frame) -> float | None:
        if self._cam.horizon_y is not None:
            return self._cam.horizon_y
        if self._settings.geometry.auto_horizon:
            return detect_horizon_y(frame.image)
        return None

    def _build_event(self, frame, tracks, backend: Backend, latency_ms: float) -> DetectionEvent:
        h, w = frame.image.shape[:2]
        horizon_y = self._resolve_horizon(frame)
        calib = (CalibrationStatus.ok if self._cam.horizon_y is not None
                 else CalibrationStatus.auto if horizon_y is not None
                 else CalibrationStatus.uncalibrated)

        targets = []
        max_area_frac = self._settings.detector.max_area_frac
        frame_area = float(w * h) or 1.0
        for tr in tracks:
            if tr.confidence < self.confidence:
                continue
            # Drop oversized detections (own hull / very-near structure): they
            # swamp the frame and create phantom dark-target/collision alerts.
            if (tr.w * tr.h) / frame_area > max_area_frac:
                continue
            brg = estimate_bearing(tr, self._cam, w)
            rng, method, rconf = estimate_range(
                tr, self._cam, self._settings.geometry, w, h, horizon_y)
            # Use the waterline (bbox bottom) consistently with range estimation.
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
            ))

        return DetectionEvent(
            camera=self._cam.name,
            timestamp=_now_iso(),
            frame_seq=frame.seq,
            frame_size=FrameSize(w=w, h=h),
            horizon_y=horizon_y,
            inference=Inference(backend=backend, latency_ms=latency_ms),
            calibration_status=calib,
            targets=targets,
        )


class Pipeline:
    def __init__(self, settings: Settings, logger):
        self.settings = settings
        self._log = logger
        self.events = EventBuffer(maxlen=settings.server.event_buffer)
        self.frames = LatestFrame()
        self.workers: dict[str, CameraWorker] = {}
        self.detector = None  # shared across workers; created in start()
        self.started_at = time.time()
        self.active_camera: str = settings.cameras[0].name if settings.cameras else "forward"
        self.mode_hint: str | None = None

    def start(self) -> None:
        # One detector for all cameras: a single model load / CUDA context. Two
        # concurrent TensorRT inits on the Orin Nano race the allocator and one
        # camera dies (NvMap ENOMEM); sharing avoids that entirely.
        detector = None
        try:
            detector = create_detector(self.settings)
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            self._log.error("detector init failed: %s", exc)
        self.detector = detector
        for cam in self.settings.cameras:
            w = CameraWorker(cam, self.settings, self.events, self.frames, self._log, detector)
            self.workers[cam.name] = w
            w.start()
            self._log.info("started camera worker: %s", cam.name)

    def stop(self) -> None:
        for w in self.workers.values():
            w.stop()
        for w in self.workers.values():
            w.join(timeout=2.0)
            # If the read() call was blocked and the join timed out, release the
            # capture device anyway so we don't leak it across restarts.
            if w.is_alive():
                w.close_source()

    def set_confidence(self, value: float) -> None:
        for w in self.workers.values():
            w.confidence = value

    def set_active_camera(self, name: str) -> None:
        if name in self.workers:
            self.active_camera = name

    def camera_errors(self) -> dict:
        return {name: w.error for name, w in self.workers.items() if w.error}
