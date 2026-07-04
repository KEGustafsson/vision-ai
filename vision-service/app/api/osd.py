"""GPU overlay drawing for the DeepStream zero-copy path.

This is the ``nvdsosd`` counterpart of :func:`app.api.overlay.annotate`. Instead
of drawing boxes/labels onto a host-copied numpy frame with OpenCV, it populates
``NvDsDisplayMeta`` (rectangles, lines, text) that ``nvdsosd`` renders on the GPU
directly on the NVMM surface — so the pixels never leave NVMM before ``nvjpegenc``
encodes them. The visual result mirrors ``annotate()`` as closely as the nvdsosd
primitives allow; the one unavoidable difference is the coasted-box "dashed"
border, which nvdsosd has no native support for and is emulated with short line
segments.

Call :func:`draw_event` from a pad probe placed upstream of ``nvdsosd`` (it must
run before the OSD element consumes the buffer). ``pyds`` is passed in so this
module imports cleanly off-Jetson (e.g. for unit tests of the colour maths).
"""

from __future__ import annotations

from ..schemas import DetectionEvent, Target

# nvdsosd packs at most this many elements of each kind into one display meta
# (NvDsDisplayMeta's fixed-size arrays). We roll over to a fresh display meta
# when any of rects/lines/labels would overflow.
_MAX_ELEMENTS = 16

# Severity colours as RGBA floats (0..1), matching overlay.py's BGR constants:
#   _GREEN (0,200,0)  _AMBER (0,165,255)  _RED (0,0,255)
_GREEN = (0.0, 200 / 255, 0.0, 1.0)
_AMBER = (1.0, 165 / 255, 0.0, 1.0)
_RED = (1.0, 0.0, 0.0, 1.0)
_WHITE = (1.0, 1.0, 1.0, 1.0)
_HORIZON = (200 / 255, 200 / 255, 200 / 255, 1.0)
_BLACK = (0.0, 0.0, 0.0, 1.0)

_FONT = "Serif"  # always available to nvdsosd's pango backend


def _severity_colour(t: Target) -> tuple:
    if t.is_person_in_water:
        return _RED
    if t.geometry.range_m is not None and t.geometry.range_m < 100:
        return _AMBER
    return _GREEN


def _dim(colour: tuple) -> tuple:
    """0.6× brightness, matching overlay.py's coasted-box dimming."""
    r, g, b, a = colour
    return (r * 0.6, g * 0.6, b * 0.6, a)


class _MetaWriter:
    """Acquire NvDsDisplayMeta from the batch pool on demand and roll over to a
    fresh one before any element array overflows ``_MAX_ELEMENTS``."""

    def __init__(self, pyds, batch_meta, frame_meta):
        self._pyds = pyds
        self._bm = batch_meta
        self._fm = frame_meta
        self._dm = None

    def _slot(self, *, rects=0, lines=0, labels=0):
        d = self._dm
        if d is None or (d.num_rects + rects > _MAX_ELEMENTS
                         or d.num_lines + lines > _MAX_ELEMENTS
                         or d.num_labels + labels > _MAX_ELEMENTS):
            self.flush()
            d = self._dm = self._pyds.nvds_acquire_display_meta_from_pool(self._bm)
            d.num_rects = d.num_lines = d.num_labels = 0
        return d

    def rect(self, x, y, w, h, colour, border_width=2):
        d = self._slot(rects=1)
        r = d.rect_params[d.num_rects]
        r.left, r.top, r.width, r.height = float(x), float(y), float(w), float(h)
        r.border_width = border_width
        r.border_color.set(*colour)
        r.has_bg_color = 0
        d.num_rects += 1

    def line(self, x1, y1, x2, y2, colour, width=2):
        d = self._slot(lines=1)
        ln = d.line_params[d.num_lines]
        # NvOSD_LineParams fields are unsigned; a coasting/extrapolated box
        # that has drifted off the left/top edge produces a negative
        # coordinate that pyds rejects outright (raises, skipping the whole
        # frame's overlay). Clamp to the canvas edge instead.
        ln.x1, ln.y1, ln.x2, ln.y2 = (max(0, int(x1)), max(0, int(y1)),
                                       max(0, int(x2)), max(0, int(y2)))
        ln.line_width = width
        ln.line_color.set(*colour)
        d.num_lines += 1

    def text(self, s, x, y, colour, font_size=11):
        d = self._slot(labels=1)
        t = d.text_params[d.num_labels]
        t.display_text = s
        t.x_offset = max(0, int(x))
        t.y_offset = max(0, int(y))
        t.font_params.font_name = _FONT
        t.font_params.font_size = font_size
        t.font_params.font_color.set(*colour)
        t.set_bg_clr = 1
        t.text_bg_clr.set(*_BLACK)  # black plate behind text, like overlay.py
        d.num_labels += 1

    def flush(self):
        if self._dm is not None:
            self._pyds.nvds_add_display_meta_to_frame(self._fm, self._dm)
            self._dm = None


def _dashed_edges(x0, y0, x1, y1, dash=10):
    """Yield (x1,y1,x2,y2) segments approximating a dashed rectangle border —
    nvdsosd has no dashed primitive (mirrors overlay._dashed_rect)."""
    for ax, ay, bx, by in ((x0, y0, x1, y0), (x1, y0, x1, y1),
                           (x1, y1, x0, y1), (x0, y1, x0, y0)):
        length = int(max(abs(bx - ax), abs(by - ay)))
        if length == 0:
            continue
        for s in range(0, length, dash * 2):
            t0, t1 = s / length, min((s + dash) / length, 1.0)
            yield (ax + (bx - ax) * t0, ay + (by - ay) * t0,
                   ax + (bx - ax) * t1, ay + (by - ay) * t1)


def draw_event(pyds, batch_meta, frame_meta, event: DetectionEvent) -> None:
    """Add display meta to *frame_meta* reproducing :func:`overlay.annotate`.

    Must run upstream of nvdsosd. No pixels are touched — only metadata."""
    mw = _MetaWriter(pyds, batch_meta, frame_meta)
    width = event.frame_size.w

    if event.horizon_y is not None:
        y = int(event.horizon_y)
        mw.line(0, y, width, y, _HORIZON, width=1)

    for t in event.targets:
        colour = _severity_colour(t)
        bx, by = int(t.bbox.x), int(t.bbox.y)
        bw, bh = int(t.bbox.w), int(t.bbox.h)
        brg = f"{t.geometry.relative_bearing_deg:+.0f}deg"
        rng = f"{t.geometry.range_m:.0f}m" if t.geometry.range_m is not None else "?"
        tid = f"#{t.track_id}" if t.track_id is not None else ""
        label = f"{t.label}{tid} {brg} {rng}"
        if t.is_person_in_water:
            label = "MOB! " + label
        ty = max(by - 18, 0)
        if t.coasting:
            dim = _dim(colour)
            for x1, y1, x2, y2 in _dashed_edges(bx, by, bx + bw, by + bh):
                mw.line(x1, y1, x2, y2, dim)
            mw.text(label + " ~", bx, ty, dim)
        else:
            mw.rect(bx, by, bw, bh, colour)
            mw.text(label, bx, ty, colour)

    hud = (f"{event.camera}  {event.inference.backend.value}  "
           f"{event.inference.latency_ms:.0f}ms  n={len(event.targets)}")
    mw.text(hud, 10, 4, _WHITE, font_size=13)
    mw.text(_format_timestamp(event.timestamp), 10, 30, _WHITE, font_size=13)

    mw.flush()


def _format_timestamp(ts: str) -> str:
    """``2026-05-31T12:34:56.789Z`` -> ``2026-05-31 12:34:56 UTC`` (as overlay.py)."""
    date, _, rest = ts.partition("T")
    clock = rest.rstrip("Z").split(".", 1)[0].split("+", 1)[0]
    return f"{date} {clock} UTC".strip()
