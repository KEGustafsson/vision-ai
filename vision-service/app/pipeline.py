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


class CameraWorker(threading.Thread):
    def __init__(self, cam: CameraConfig, settings: Settings,
                 events: EventBuffer, frames: LatestFrame, logger):
        super().__init__(daemon=True, name=f"cam-{cam.name}")
        self._cam = cam
        self._settings = settings
        self._events = events
        self._frames = frames
        self._log = logger
        self._stop = threading.Event()
        self._source = None
        self._detector = None
        # Runtime-adjustable via /control.
        self.confidence = settings.detector.confidence
        self.last_event: dict | None = None

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            self._source = create_source(self._cam, self._settings)
            self._detector = create_detector(self._settings)
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            self._log.error("camera %s init failed: %s", self._cam.name, exc)
            return

        period = 1.0 / max(self._settings.server.target_fps, 0.1)
        backend = Backend(self._detector.backend)
        while not self._stop.is_set():
            t0 = time.perf_counter()
            frame = self._source.read()
            if frame is None:
                time.sleep(0.1)
                continue

            tracks = self._detector.detect_and_track(frame)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            event = self._build_event(frame, tracks, backend, latency_ms)

            self._events.publish(event.model_dump(mode="json"))
            self.last_event = event.model_dump(mode="json")
            jpeg = encode_jpeg(annotate(frame.image, event),
                               self._settings.server.jpeg_quality)
            if jpeg:
                self._frames.set(self._cam.name, jpeg)

            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)

        if self._source:
            self._source.close()

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
        for tr in tracks:
            if tr.confidence < self.confidence:
                continue
            brg = estimate_bearing(tr, self._cam, w)
            rng, method, rconf = estimate_range(
                tr, self._cam, self._settings.geometry, w, h, horizon_y)
            piw = is_person_in_water(tr.label, tr.cy, horizon_y)
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
        self.started_at = time.time()

    def start(self) -> None:
        for cam in self.settings.cameras:
            w = CameraWorker(cam, self.settings, self.events, self.frames, self._log)
            self.workers[cam.name] = w
            w.start()
            self._log.info("started camera worker: %s", cam.name)

    def stop(self) -> None:
        for w in self.workers.values():
            w.stop()
        for w in self.workers.values():
            w.join(timeout=2.0)

    def set_confidence(self, value: float) -> None:
        for w in self.workers.values():
            w.confidence = value
