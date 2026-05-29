"""Draw detection boxes + bearing/range/label/track onto a frame, colour-coded
by a coarse severity (person-in-water > close range > normal)."""

from __future__ import annotations

import cv2
import numpy as np

from ..schemas import DetectionEvent, Target

_GREEN = (0, 200, 0)
_AMBER = (0, 165, 255)
_RED = (0, 0, 255)


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
        cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
        brg = f"{t.geometry.relative_bearing_deg:+.0f}°"
        rng = f"{t.geometry.range_m:.0f}m" if t.geometry.range_m is not None else "?"
        tid = f"#{t.track_id}" if t.track_id is not None else ""
        label = f"{t.label}{tid} {brg} {rng}"
        if t.is_person_in_water:
            label = "MOB! " + label
        cv2.putText(img, label, (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA)

    hud = f"{event.camera}  {event.inference.backend.value}  {event.inference.latency_ms:.0f}ms  n={len(event.targets)}"
    cv2.putText(img, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def encode_jpeg(image: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""
