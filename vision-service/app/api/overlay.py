"""Draw detection boxes + bearing/range/label/track onto a frame, colour-coded
by a coarse severity (person-in-water > close range > normal)."""

from __future__ import annotations

import cv2
import numpy as np

from ..schemas import DetectionEvent, Target

_GREEN = (0, 200, 0)
_AMBER = (0, 165, 255)
_RED = (0, 0, 255)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BG_COLOUR = (0, 0, 0)  # black plate behind text for contrast over sky/water


def _draw_label(image: np.ndarray, text: str, org: tuple, fg: tuple,
                scale: float = 0.5, thickness: int = 1) -> None:
    """Draw *text* on a filled black plate so it stays readable over any
    background. *org* is the text baseline-left (same anchor as cv2.putText);
    the plate and text are clamped to stay fully on-frame."""
    x, y = int(org[0]), int(org[1])
    (tw, th), base = cv2.getTextSize(text, _FONT, scale, thickness)
    h, w = image.shape[:2]
    pad = 3
    # Keep the whole label (plate included) inside the frame.
    x = max(0, min(x, w - tw - 2 * pad))
    y = max(th + pad, min(y, h - base - pad))
    cv2.rectangle(image, (x, y - th - pad), (x + tw + 2 * pad, y + base + pad),
                  _BG_COLOUR, cv2.FILLED)
    cv2.putText(image, text, (x + pad, y), _FONT, scale, fg, thickness, cv2.LINE_AA)


def _format_timestamp(ts: str) -> str:
    """Turn the event's ISO-8601 UTC timestamp (e.g. ``2026-05-31T12:34:56.789Z``)
    into a compact, human-readable ``2026-05-31 12:34:56 UTC`` for the overlay."""
    date, _, rest = ts.partition("T")
    clock = rest.rstrip("Z").split(".", 1)[0].split("+", 1)[0]
    return f"{date} {clock} UTC".strip()


def _dashed_rect(image: np.ndarray, p0: tuple, p1: tuple, colour: tuple,
                 thickness: int = 2, dash: int = 10) -> None:
    """Axis-aligned dashed rectangle — marks a coasted (predicted) box."""
    x0, y0 = p0
    x1, y1 = p1
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x1, y0, x1, y1),
                             (x1, y1, x0, y1), (x0, y1, x0, y0)):
        length = int(max(abs(bx - ax), abs(by - ay)))
        if length == 0:
            continue
        for s in range(0, length, dash * 2):
            t0, t1 = s / length, min((s + dash) / length, 1.0)
            cv2.line(image,
                     (int(ax + (bx - ax) * t0), int(ay + (by - ay) * t0)),
                     (int(ax + (bx - ax) * t1), int(ay + (by - ay) * t1)),
                     colour, thickness, cv2.LINE_AA)


def _severity_colour(t: Target) -> tuple:
    if t.is_person_in_water:
        return _RED
    if t.geometry.range_m is not None and t.geometry.range_m < 100:
        return _AMBER
    return _GREEN


def annotate(image: np.ndarray, event: DetectionEvent) -> np.ndarray:
    img = image.copy()
    if event.horizon_y is not None:
        y = int(event.horizon_y)
        cv2.line(img, (0, y), (img.shape[1], y), (200, 200, 200), 1)

    for t in event.targets:
        colour = _severity_colour(t)
        x, y, w, h = int(t.bbox.x), int(t.bbox.y), int(t.bbox.w), int(t.bbox.h)
        brg = f"{t.geometry.relative_bearing_deg:+.0f}deg"
        rng = f"{t.geometry.range_m:.0f}m" if t.geometry.range_m is not None else "?"
        tid = f"#{t.track_id}" if t.track_id is not None else ""
        label = f"{t.label}{tid} {brg} {rng}"
        if t.is_person_in_water:
            label = "MOB! " + label
        if t.coasting:
            # Predicted (not detected this frame): dashed + dimmed colour so the
            # box/info persist without claiming a fresh detection.
            dim = tuple(int(c * 0.6) for c in colour)
            _dashed_rect(img, (x, y), (x + w, y + h), dim)
            _draw_label(img, label + " ~", (x, max(y - 6, 12)), dim)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
            _draw_label(img, label, (x, max(y - 6, 12)), colour)

    hud = f"{event.camera}  {event.inference.backend.value}  {event.inference.latency_ms:.0f}ms  n={len(event.targets)}"
    _draw_label(img, hud, (10, 24), (255, 255, 255), scale=0.6)

    # Stamp the capture time on every frame, just below the HUD line in the
    # top-left corner, so any streamed or saved frame carries its own timestamp.
    _draw_label(img, _format_timestamp(event.timestamp),
                (10, 50), (255, 255, 255), scale=0.6)
    return img


def encode_jpeg(image: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""
