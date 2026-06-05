"""JPEG encoders for the annotated MJPEG/snapshot frames.

Two backends behind a common ``encode(bgr) -> bytes`` interface:

* :class:`CpuJpegEncoder` — ``cv2.imencode`` on the CPU (the default, always
  available).
* :class:`HwJpegEncoder` — the Jetson NVJPG hardware block via a GStreamer
  ``appsrc ! videoconvert ! nvjpegenc ! appsink`` pipeline. Offloads the encode
  (a large slice of the per-frame ``post`` cost) off the CPU. Needs PyGObject
  (``python3-gi``) + GStreamer introspection in the image; raises on
  import/build failure so the factory falls back to the CPU encoder.

Use :func:`make_jpeg_encoder` to pick the backend from settings; it never raises
— an unavailable/broken HW path silently degrades to CPU so a misconfigured flag
can't take the stream down. Each encoder instance is single-threaded (the HW
GStreamer pipeline is not thread-safe): give each producer its own encoder.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np


class CpuJpegEncoder:
    backend = "cpu"

    def __init__(self, quality: int = 80):
        self._params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]

    def encode(self, image: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", image, self._params)
        return buf.tobytes() if ok else b""

    def close(self) -> None:  # pragma: no cover - trivial
        pass


class HwJpegEncoder:
    """Hardware JPEG encode through ``nvjpegenc`` (Jetson NVJPG engine), with the
    colorspace conversion offloaded to the VIC instead of the CPU.

    The chain is ``appsrc(BGRx) ! nvvidconv ! NVMM I420 ! nvjpegenc ! appsink``.
    ``nvvidconv`` runs the BGRx->I420 conversion and the NVMM upload on the VIC
    hardware (the slice that ``videoconvert`` used to burn on the CPU), so the
    only residual CPU work is padding the 3-channel BGR frame to 4-channel BGRx —
    a cheap channel expand, not a YUV conversion — because ``nvvidconv`` does not
    accept packed 24-bit BGR. The encode itself runs on the NVJPG block.

    The pipeline is built lazily on the first frame and rebuilt if the frame size
    changes (mirrors :class:`~app.api.undistort.Undistorter`).
    """

    backend = "nvjpegenc"

    def __init__(self, quality: int = 80):
        import gi  # raises ImportError when python3-gi isn't installed

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        if not Gst.is_initialized():
            Gst.init(None)
        # Fail fast (so make_jpeg_encoder falls back to CPU) when the NVIDIA
        # plugins are absent — e.g. PyGObject is installed but we're not on a
        # Jetson with the multimedia stack. Otherwise the missing element would
        # only surface as a per-frame error on the first encode.
        for el in ("nvvidconv", "nvjpegenc"):
            if Gst.ElementFactory.find(el) is None:
                raise RuntimeError(f"GStreamer element {el!r} not available")
        self._Gst = Gst
        self._quality = int(quality)
        self._lock = threading.Lock()
        self._size: tuple[int, int] | None = None
        self._pipeline = None
        self._src = None
        self._sink = None

    def _build(self, w: int, h: int) -> None:
        Gst = self._Gst
        desc = (
            "appsrc name=src is-live=true do-timestamp=true format=time "
            f"caps=video/x-raw,format=BGRx,width={w},height={h},framerate=0/1 ! "
            "nvvidconv ! video/x-raw(memory:NVMM),format=I420 ! "
            f"nvjpegenc quality={self._quality} ! "
            "appsink name=sink emit-signals=false sync=false max-buffers=1 drop=false"
        )
        pipe = Gst.parse_launch(desc)
        if pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            pipe.set_state(Gst.State.NULL)
            raise RuntimeError("nvjpegenc pipeline failed to start")
        self._pipeline = pipe
        self._src = pipe.get_by_name("src")
        self._sink = pipe.get_by_name("sink")
        self._size = (w, h)

    def encode(self, image: np.ndarray) -> bytes:
        Gst = self._Gst
        h, w = image.shape[:2]
        # Pad BGR->BGRx so nvvidconv accepts it; the YUV conversion stays on the
        # VIC. The padded byte order (B,G,R,X) matches GStreamer's BGRx.
        bgrx = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        with self._lock:
            if self._size != (w, h):
                self._teardown()
                self._build(w, h)
            buf = Gst.Buffer.new_wrapped(bgrx.tobytes())  # cvtColor output is contiguous
            if self._src.emit("push-buffer", buf) != Gst.FlowReturn.OK:
                raise RuntimeError("nvjpegenc appsrc rejected the frame")
            # Block (bounded) for the encoded frame; 1s is generous vs ~few-ms HW.
            sample = self._sink.emit("try-pull-sample", Gst.SECOND)
            if sample is None:
                raise RuntimeError("nvjpegenc produced no sample")
            gbuf = sample.get_buffer()
            ok, info = gbuf.map(Gst.MapFlags.READ)
            if not ok:
                raise RuntimeError("nvjpegenc buffer map failed")
            try:
                return bytes(info.data)
            finally:
                gbuf.unmap(info)

    def _teardown(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(self._Gst.State.NULL)
            except Exception:  # pragma: no cover
                pass
        self._pipeline = self._src = self._sink = None
        self._size = None

    def close(self) -> None:
        with self._lock:
            self._teardown()


class _ResilientHwJpegEncoder:
    """Wrap the HW encoder so a *runtime* failure degrades to CPU once and stays
    there. The factory only catches construction failures, but the GStreamer
    pipeline is built lazily on the first frame (and rebuilt on a size change),
    so ``_build``/encode can still raise later — without this the MJPEG stream
    would error every frame instead of falling back."""

    def __init__(self, hw, quality: int, logger=None):
        self._enc = hw
        self._quality = quality
        self._logger = logger
        self.backend = hw.backend

    def encode(self, image: np.ndarray) -> bytes:
        if self.backend == "cpu":
            return self._enc.encode(image)
        try:
            return self._enc.encode(image)
        except Exception as exc:
            if self._logger:
                self._logger.warning(
                    "hw_jpeg runtime failure (%s); falling back to cpu", exc)
            try:
                self._enc.close()
            except Exception:  # pragma: no cover - best-effort teardown
                pass
            self._enc = CpuJpegEncoder(self._quality)
            self.backend = self._enc.backend
            return self._enc.encode(image)

    def close(self) -> None:
        self._enc.close()


def make_jpeg_encoder(quality: int = 80, hw: bool = False, logger=None):
    """Return a JPEG encoder. Falls back to CPU if the HW path is unavailable,
    so a stale/over-eager ``hw_jpeg`` flag never takes the stream down — at
    construction (no gi/plugin) and, via the wrapper, on a later runtime failure."""
    if hw:
        try:
            enc = HwJpegEncoder(quality)
            if logger:
                logger.info("jpeg encoder: nvjpegenc (hardware)")
            return _ResilientHwJpegEncoder(enc, quality, logger)
        except Exception as exc:  # ImportError (no gi) or pipeline build failure
            if logger:
                logger.warning("hw_jpeg requested but unavailable (%s); using cpu", exc)
    return CpuJpegEncoder(quality)
