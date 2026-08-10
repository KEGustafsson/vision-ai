"""Global image motion from NVIDIA Optical Flow Accelerator (OFA) vectors.

The Jetson Orin SoC carries a dedicated optical-flow hardware block (OFA). In
DeepStream it is driven by the ``nvof`` element, which attaches a map of
block-level flow vectors to each frame as ``NvDsOpticalFlowMeta`` user meta.
This module turns that vector map into **one robust per-camera image-motion
estimate** and keeps the per-camera diagnostic state behind it.

Nothing here imports pyds, GStreamer or CUDA: the DeepStream plumbing lives in
``app/pipeline_deepstream.py`` and hands us a plain array of raw vectors, so the
maths below is unit-testable on any machine.

Vector representation
=====================
``nvof`` emits one ``NvOFFlowVector`` per 4x4 pixel block (the only grid size
DeepStream currently supports), each a pair of **signed 16-bit S10.5 fixed-point**
components — 5 fractional bits, so the pixel value is ``raw / 32.0``
(``raw 32 -> 1.0 px``). The pyds helper ``get_optical_flow_vectors()`` widens the
raw int16s to float32 but does **not** scale them, so the conversion is ours to
apply (see ``raw_to_px``).

NVIDIA notes that the quantisation floor means a genuinely static scene reports
+-0.5 px rather than exactly 0, so treat sub-pixel magnitudes as noise.

Sign convention
===============
Components are reported in the OFA's own convention, unscaled and unflipped —
this module never negates them. The mapping from ``global_dx`` sign to "image
content moved left/right" is NOT assumed here; it must be measured on the
hardware with a controlled pan (see ``docs/jetson-deepstream.md``).

Robust estimate
===============
The median — not the mean — of the valid vectors is used. On the water a large
minority of blocks is genuinely wrong or genuinely moving differently from the
camera: swell, spray, wakes, sun glitter, reflections and independently moving
vessels. A mean is dragged by those outliers; the median reports the motion of
the dominant (background) part of the image, which is what "global image motion"
means. First-implementation filtering is deliberately simple and deterministic:
drop malformed and non-finite vectors, drop absurd magnitudes, take the median.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

# NvOF flow components are S10.5 fixed point: 5 fractional bits -> 32 units/px.
OF_FIXED_POINT_DIVISOR = 32.0

# Reject vectors longer than this many pixels per frame before taking the median.
# At the cameras' ~6 FPS a real scene element crossing 128 px between frames is
# a quarter of the frame width; anything beyond that is a broken vector, not
# motion we want to average in. Generous on purpose — the median already handles
# outliers, this only keeps wild values out of the ordering.
DEFAULT_MAX_MAGNITUDE_PX = 128.0

# How old the newest flow estimate may be before a camera is reported "stale".
DEFAULT_STALE_AFTER_S = 2.0


class OpticalFlowState(str, Enum):
    """Diagnostic state of OFA for one camera."""

    disabled = "disabled"      # feature off in config — no nvof element exists
    no_data = "no_data"        # enabled, but no flow metadata has arrived yet
    active = "active"          # flow metadata received recently
    stale = "stale"            # enabled, last metadata older than the threshold
    error = "error"            # nvof unavailable / metadata parsing failed


def raw_to_px(raw: float) -> float:
    """Convert one raw S10.5 OFA component to pixels (raw 32 -> 1.0 px)."""
    return float(raw) / OF_FIXED_POINT_DIVISOR


@dataclass(frozen=True)
class OpticalFlowStats:
    """One camera's global image motion for one frame.

    ``global_dx``/``global_dy`` are in **pixels per frame interval**, in the
    OFA's own sign convention (see module docstring). ``confidence`` is the
    fraction of the frame's vectors that survived filtering — a coverage
    measure, not an accuracy claim.
    """

    global_dx: float = 0.0
    global_dy: float = 0.0
    vector_count: int = 0
    confidence: float = 0.0
    updated_at: float = 0.0  # time.monotonic() when this estimate was computed
    frame_num: int = 0       # NvDsOpticalFlowMeta.frame_num, for traceability

    @property
    def valid(self) -> bool:
        """True when at least one vector survived filtering."""
        return self.vector_count > 0


def _is_ndarray(obj: Any) -> bool:
    """Duck-type a numpy array without importing numpy (mock mode has no need
    to pay for the import here; the DeepStream path always passes an array)."""
    return hasattr(obj, "dtype") and hasattr(obj, "reshape") and hasattr(obj, "size")


def _stats(xs, ys, total: int, updated_at: float, frame_num: int) -> OpticalFlowStats:
    kept = len(xs)
    if kept == 0:
        return OpticalFlowStats(updated_at=updated_at, frame_num=frame_num)
    return OpticalFlowStats(
        global_dx=float(statistics.median(xs)),
        global_dy=float(statistics.median(ys)),
        vector_count=kept,
        confidence=(kept / total) if total else 0.0,
        updated_at=updated_at,
        frame_num=frame_num,
    )


def _estimate_ndarray(arr, max_magnitude_px: float,
                      updated_at: float, frame_num: int) -> OpticalFlowStats:
    """Vectorised path for the flat float32 array pyds hands us.

    This is metadata, not pixels: one small array of block vectors per frame
    (~77k values at 1280x960 with the 4x4 grid), so the numpy work is compact
    and bounded — no full-frame host processing is involved.
    """
    import numpy as np

    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    pairs = flat[: (flat.size // 2) * 2].reshape(-1, 2) / OF_FIXED_POINT_DIVISOR
    total = int(pairs.shape[0])
    if total == 0:
        return OpticalFlowStats(updated_at=updated_at, frame_num=frame_num)

    keep = np.isfinite(pairs).all(axis=1)
    # Zero the non-finite rows before hypot so NaN/Inf can't poison the
    # magnitude test (they are already excluded by `keep`).
    safe = np.where(keep[:, None], pairs, 0.0)
    if max_magnitude_px > 0:
        keep &= np.hypot(safe[:, 0], safe[:, 1]) <= max_magnitude_px

    good = safe[keep]
    kept = int(good.shape[0])
    if kept == 0:
        return OpticalFlowStats(updated_at=updated_at, frame_num=frame_num)
    return OpticalFlowStats(
        global_dx=float(np.median(good[:, 0])),
        global_dy=float(np.median(good[:, 1])),
        vector_count=kept,
        confidence=kept / total,
        updated_at=updated_at,
        frame_num=frame_num,
    )


def estimate_global_motion(
    vectors: Any,
    *,
    max_magnitude_px: float = DEFAULT_MAX_MAGNITUDE_PX,
    updated_at: Optional[float] = None,
    frame_num: int = 0,
) -> OpticalFlowStats:
    """Median global image motion from raw OFA flow vectors.

    ``vectors`` is either the flat float32 array returned by
    ``pyds.get_optical_flow_vectors()`` (raw S10.5 components, interleaved
    x, y, x, y, ...) or any iterable of ``(raw_x, raw_y)`` pairs — the latter is
    what the unit tests use. Values are converted to pixels here.

    Never raises: malformed entries, non-finite values and absurd magnitudes are
    dropped, and an input with nothing usable yields ``vector_count == 0``
    (``stats.valid`` False) rather than an exception, because a probe callback
    must not be able to take the pipeline down.
    """
    now = time.monotonic() if updated_at is None else updated_at
    if vectors is None:
        return OpticalFlowStats(updated_at=now, frame_num=frame_num)

    if _is_ndarray(vectors):
        try:
            return _estimate_ndarray(vectors, max_magnitude_px, now, frame_num)
        except Exception:
            # Unexpected array shape/dtype: report "no estimate", never raise.
            return OpticalFlowStats(updated_at=now, frame_num=frame_num)

    xs: list = []
    ys: list = []
    total = 0
    for item in vectors:
        total += 1
        try:
            raw_x, raw_y = item[0], item[1]
        except Exception:
            continue  # malformed vector (too short, not indexable, ...)
        try:
            dx = float(raw_x) / OF_FIXED_POINT_DIVISOR
            dy = float(raw_y) / OF_FIXED_POINT_DIVISOR
        except (TypeError, ValueError):
            continue  # non-numeric component
        if not (math.isfinite(dx) and math.isfinite(dy)):
            continue
        if max_magnitude_px > 0 and math.hypot(dx, dy) > max_magnitude_px:
            continue
        xs.append(dx)
        ys.append(dy)

    return _stats(xs, ys, total, now, frame_num)


@dataclass
class CameraFlowState:
    """Latest OFA result and diagnostic state for ONE camera.

    Per-camera by construction: cameras never share an instance, so one feed
    stalling or reconnecting cannot corrupt another's history.

    Threading: the fields are written only by the DeepStream probe (one
    streaming thread) and read by FastAPI worker threads through
    :meth:`snapshot`. Each write replaces an immutable value in a single
    attribute assignment, which is atomic under the GIL, so the per-frame path
    stays lock-free — matching the rest of the DeepStream probe, which keeps
    locks off the hot path.
    """

    name: str
    enabled: bool = False
    stats: Optional[OpticalFlowStats] = None
    error: Optional[str] = None

    def update(self, stats: OpticalFlowStats) -> None:
        """Record a fresh estimate (and clear any previous error)."""
        self.stats = stats
        self.error = None

    def fail(self, message: str) -> bool:
        """Record an error. Returns True only when the message is new, so the
        caller can log once instead of once per frame."""
        changed = self.error != message
        self.error = message
        return changed

    def reset(self) -> None:
        """Forget flow history — called on every pipeline (re)build.

        Optical flow is temporal: after a rebuild, an RTSP reconnect or a
        detection off/on toggle the first frame legitimately has no flow, and
        the estimate that preceded the gap describes a different camera epoch.
        """
        self.stats = None
        self.error = None

    def state(self, now: float, stale_after_s: float = DEFAULT_STALE_AFTER_S) -> OpticalFlowState:
        if not self.enabled:
            return OpticalFlowState.disabled
        if self.error:
            return OpticalFlowState.error
        if self.stats is None:
            return OpticalFlowState.no_data
        if stale_after_s > 0 and (now - self.stats.updated_at) > stale_after_s:
            return OpticalFlowState.stale
        return OpticalFlowState.active

    def snapshot(self, now: Optional[float] = None,
                 stale_after_s: float = DEFAULT_STALE_AFTER_S) -> Dict[str, Any]:
        """Diagnostics for /health (snake_case, matching the other fields)."""
        if now is None:
            now = time.monotonic()
        state = self.state(now, stale_after_s)
        st = self.stats
        return {
            "enabled": self.enabled,
            "state": state.value,
            "active": state is OpticalFlowState.active,
            "global_dx": None if st is None else round(st.global_dx, 3),
            "global_dy": None if st is None else round(st.global_dy, 3),
            "vectors": 0 if st is None else st.vector_count,
            "confidence": 0.0 if st is None else round(st.confidence, 3),
            "age_ms": None if st is None else int(max(0.0, now - st.updated_at) * 1000),
            "error": self.error,
        }
