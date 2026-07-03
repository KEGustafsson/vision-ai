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

from .api.jpeg import make_jpeg_encoder
from .api.overlay import annotate
from .api.undistort import Undistorter
from .camera import create_source
from .config import CameraConfig, Settings
from .detector import create_detector
from .detector.classmap import is_person_in_water
from .detector.dedup import TargetDeduper
from .detector.stabilizer import TrackStabilizer
from .geometry import detect_horizon_y, estimate_bearing, estimate_range
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
        # Master on/off (set => running). When cleared the worker releases its
        # capture device and idles; toggled from SignalK via /control `enabled`.
        self._enabled = threading.Event()
        self._enabled.set()
        self._source = None
        # Lazily built once the frame size is known; only when undistort is on.
        self._undistorter: Undistorter | None = None
        # Per-worker JPEG encoder (the HW GStreamer pipeline isn't thread-safe, so
        # each camera owns its own). Falls back to CPU cv2.imencode when nvjpegenc
        # is unavailable, so a stale hw_jpeg flag can't take the stream down.
        self._encoder = make_jpeg_encoder(
            quality=settings.server.jpeg_quality,
            hw=settings.server.hw_jpeg,
            logger=logger,
        )
        # Per-camera flicker damping: keeps a detected track alive (coasted)
        # across short dropouts instead of blinking it off. State is per camera.
        d = settings.detector
        self._stabilizer = TrackStabilizer(
            confirm_frames=d.stabilize_confirm_frames,
            max_coast_frames=d.stabilize_max_coast_frames,
            hysteresis_ratio=d.stabilize_hysteresis_ratio,
            ema_alpha=d.stabilize_ema_alpha,
            coast_velocity_factor=d.stabilize_coast_velocity_factor,
            person_confirm_frames=d.stabilize_person_confirm_frames,
            bbox_ema_alpha=d.stabilize_bbox_ema_alpha,
        ) if d.stabilize else None
        # Per-camera same-vessel duplicate suppression (sticky loser→winner
        # state, so it must not be shared across cameras). Hold pairings past
        # the coast window so a briefly-coasted winner can't lose its claim.
        self._deduper = TargetDeduper(
            vessel_ios=d.duplicate_vessel_ios,
            contained_frac=d.contained_frac,
            hold_frames=2 * d.stabilize_max_coast_frames,
        )
        # Runtime-adjustable via /control.
        self.confidence = settings.detector.confidence
        # Drop detections closer than this (m); seeded from config, owned by the
        # SignalK plugin (minTargetRangeM) which pushes it via /control. 0 => off.
        self.min_target_range_m = settings.detector.min_target_range_m
        # Canonical labels to surface; None => all. Seeded from config, then
        # driven by the SignalK plugin's object-type selection via /control.
        self.allowed_labels: set | None = (
            set(settings.detector.classes) if settings.detector.classes else None)
        self.error: str | None = None

    def stop(self) -> None:
        self._stop.set()
        # Unblock the run loop if it's parked in the disabled-idle wait.
        self._enabled.set()

    def set_enabled(self, value: bool) -> None:
        if value:
            self._enabled.set()
        else:
            self._enabled.clear()

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

        period = 1.0 / max(self._settings.server.target_fps, 0.1)
        backend = Backend(self._detector.backend)
        stalled_since: float | None = None
        try:
            while not self._stop.is_set():
                t0 = time.perf_counter()

                # Detection disabled from SignalK: release the capture device so
                # we stop decoding the feed entirely, drop the last frame, and
                # park until re-enabled (wakes within 0.5s of a toggle).
                if not self._enabled.is_set():
                    if self._source is not None:
                        self.close_source()
                    self._frames.clear(self._cam.name)
                    self.error = None
                    stalled_since = None
                    self._enabled.wait(timeout=0.5)
                    continue

                # Lazily (re)open the source — on first enable and after a
                # disable/enable cycle. A failed open is retried, not fatal, so a
                # camera that's briefly unreachable recovers on its own.
                if self._source is None:
                    try:
                        self._source = create_source(self._cam, self._settings)
                    except Exception as exc:  # pragma: no cover - hardware dependent
                        self.error = f"init failed: {exc}"
                        self._log.error("camera %s init failed: %s", self._cam.name, exc)
                        if self._stop.wait(timeout=2.0):
                            break
                        continue

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

                    # EXPERIMENTAL: correct the frame before detection so the
                    # detector + geometry see the straightened image. Otherwise
                    # correction is display-only (see _for_display).
                    if self._cam.undistort and self._cam.undistort_before_detect:
                        h, w = frame.image.shape[:2]
                        frame.image = self._undistorter_for(w, h).image(frame.image)

                    tracks = self._detector.detect_and_track(
                        frame, self._cam.name, max_det=self._settings.detector.max_det)
                    if self._stabilizer is not None:
                        # Coast/hysteresis/debounce. conf_on tracks the runtime
                        # publish threshold so /control confidence still applies.
                        tracks = self._stabilizer.update(
                            tracks, frame.seq, self.confidence)
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    event = self._build_event(frame, tracks, backend, latency_ms)

                    payload = event.model_dump(mode="json")
                    self._events.publish(payload)
                    # Display-only lens correction: undistort the shown frame and
                    # remap the overlay to match. The published event above keeps
                    # the raw-frame geometry, so bearings/range are unaffected.
                    disp_img, disp_event = self._for_display(frame.image, event)
                    jpeg = self._encoder.encode(annotate(disp_img, disp_event))
                    # set() is a no-op while the store is paused (detection off),
                    # so a frame encoded just before a disable cannot resurface.
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
            self._encoder.close()

    def _undistorter_for(self, w, h) -> Undistorter:
        """Lazily build + cache this camera's undistorter for the frame size."""
        u = self._undistorter
        if u is None or u.size != (w, h):
            u = self._undistorter = Undistorter(self._cam, w, h)
            self._log.info("camera %s undistort backend: %s (before_detect=%s)",
                           self._cam.name, u.backend, self._cam.undistort_before_detect)
        return u

    def _for_display(self, image, event: DetectionEvent):
        """Return (display_image, display_event) with optional cosmetic lens
        correction applied to both, so the overlay still lands correctly. When
        the camera has no undistort config — or it was already corrected before
        detection — this is a cheap pass-through."""
        if not self._cam.undistort or self._cam.undistort_before_detect:
            return image, event
        h, w = image.shape[:2]
        u = self._undistorter_for(w, h)
        disp = u.image(image)
        ev = event.model_copy(deep=True)
        if ev.horizon_y is not None:
            ev.horizon_y = u.horizon_y(ev.horizon_y, w)
        for t in ev.targets:
            bx, by, bw, bh = u.bbox(t.bbox.x, t.bbox.y, t.bbox.w, t.bbox.h)
            t.bbox.x, t.bbox.y, t.bbox.w, t.bbox.h = bx, by, bw, bh
        return disp, ev

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
            # The stabilizer already applied the confidence gate (with
            # hysteresis); only filter here when it's disabled.
            if self._stabilizer is None and tr.confidence < self.confidence:
                continue
            # Only surface the operator-selected object types (set via the
            # SignalK plugin). None => all. Filtering here also keeps the
            # annotated overlay limited to the selected classes.
            if self.allowed_labels is not None and tr.label not in self.allowed_labels:
                continue
            # Drop oversized detections (own hull / very-near structure): they
            # swamp the frame and create phantom dark-target/collision alerts.
            # person is EXEMPT: a man-overboard close to the hull legitimately
            # fills much of the frame, and dropping it here — before the
            # is_person_in_water classification below — would silently lose the
            # most safety-critical detection. (Consistent with the min-range
            # filter, which also exempts person.)
            if tr.label != "person" and (tr.w * tr.h) / frame_area > max_area_frac:
                continue
            brg = estimate_bearing(tr, self._cam, w)
            rng, method, rconf = estimate_range(
                tr, self._cam, self._settings.geometry, w, h, horizon_y)
            # Minimum-range gate (own-hull / very-near clutter), applied EARLY so
            # neither the event nor the overlay shows a too-close object. person is
            # exempt (MOB must be seen up close); unknown range is kept. The value
            # is owned by the SignalK plugin (detector.minTargetRangeM via /control).
            if (self.min_target_range_m > 0 and tr.label != "person"
                    and rng is not None and 0 < rng < self.min_target_range_m):
                continue
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
                coasting=tr.coasting,
            ))
        # Collapse duplicate detections of one physical vessel (hull vs
        # hull+mast double-fires) and drop boxes nested inside a larger
        # detection (deck clutter), before ranking/capping.
        targets = self._deduper.update(targets, frame.seq)
        targets = sorted(
            targets, key=lambda t: t.confidence, reverse=True
        )[:self._settings.detector.max_det]

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
        self.enabled: bool = True

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

    def set_min_target_range(self, value: float) -> None:
        for w in self.workers.values():
            w.min_target_range_m = value

    def set_labels(self, labels: list | None) -> None:
        """Restrict surfaced detections to these canonical labels. An empty list
        or None means "all" (a safe default — never blacks out detection)."""
        allowed = set(labels) if labels else None
        for w in self.workers.values():
            w.allowed_labels = allowed

    def labels(self) -> list | None:
        vals = []
        for w in self.workers.values():
            vals.append(None if w.allowed_labels is None else tuple(sorted(w.allowed_labels)))
        unique = set(vals)
        if len(unique) != 1:
            return None
        only = unique.pop()
        return None if only is None else list(only)

    def set_max_targets(self, value: int) -> None:
        self.settings.detector.max_det = value

    def max_targets(self) -> int:
        return self.settings.detector.max_det

    def set_active_camera(self, name: str) -> None:
        if name in self.workers:
            self.active_camera = name

    def set_enabled(self, value: bool) -> None:
        """Master on/off. Disabling pauses every camera worker (capture released,
        no inference); enabling resumes them. Idempotent."""
        self.enabled = value
        # Flip the frame store first: pause() atomically blocks further writes
        # and drops retained frames, so a worker finishing a frame concurrently
        # with this disable can't re-publish a stale image (its set() is rejected
        # under the same lock). resume() re-allows writes on enable.
        if value:
            self.frames.resume()
        else:
            self.frames.pause()
        for w in self.workers.values():
            w.set_enabled(value)

    def camera_errors(self) -> dict:
        return {name: w.error for name, w in self.workers.items() if w.error}
