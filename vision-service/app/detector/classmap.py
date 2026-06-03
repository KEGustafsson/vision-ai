"""Map raw model class ids to canonical marine labels, and apply the
person-in-water (man-overboard candidate) rule.

Exactly ONE detection model is active at a time (selected via
``detector.model``). Two models are supported:

  ``"coco"`` — **COCO YOLOv8n** (80 classes, the default)
    COCO ids relevant to the marine domain: 0=person, 8=boat.  Class id 80
    is an extension we reserve for "buoy".

  ``"forward-watch"`` — **forward-watch** (6 marine-specific classes)
    0=ship, 1=boat, 2=debris, 3=buoy, 4=kayak, 5=log.
    Raw class ids are remapped to 81–86 so the ``coco_class`` field on the
    wire stays unambiguous regardless of which model produced the detection.
"""

from __future__ import annotations

from typing import Optional, Tuple

MODEL_COCO = "coco"
MODEL_FORWARD_WATCH = "forward-watch"

# Model name -> nvinfer config file (relative to the deepstream/ directory).
MODEL_PGIE_CONFIG = {
    MODEL_COCO: "pgie_yolov8n.txt",
    MODEL_FORWARD_WATCH: "pgie_forward_watch.txt",
}

# Canonical labels each model can produce (deduplicated, sorted).
MODEL_LABELS = {
    MODEL_COCO: sorted({"person", "vessel", "buoy"}),
    MODEL_FORWARD_WATCH: sorted({"vessel", "buoy", "debris", "kayak", "log"}),
}

# Minimal COCO id -> name table for the classes we care to surface.
_COCO = {
    0: "person",
    8: "boat",
    33: "kite",      # occasionally fires on sails; remapped below
}

# Canonical marine label per (possibly remapped) class id.
_MARINE = {
    0: "person",
    8: "vessel",
    80: "buoy",
}

# forward-watch raw class id → canonical marine label.
_FW_LABEL = {
    0: "vessel",   # ship
    1: "vessel",   # boat
    2: "debris",
    3: "buoy",
    4: "kayak",
    5: "log",
}

# forward-watch raw class id → wire-safe synthetic id (avoids COCO collision).
_FW_CLS_REMAP = {
    0: 81,   # ship
    1: 82,   # boat
    2: 83,   # debris
    3: 84,   # buoy
    4: 85,   # kayak
    5: 86,   # log
}

# Inverse of _FW_CLS_REMAP, for label_for() lookups off the synthetic id.
_FW_CLS_REMAP_INV = {v: k for k, v in _FW_CLS_REMAP.items()}


def label_for(cls: int) -> str:
    """Canonical label for a (possibly already-remapped) class id.

    Accepts both COCO ids and the forward-watch synthetic ids (81+) so callers
    holding only the wire ``coco_class`` can still resolve a label.
    """
    if cls in _MARINE:
        return _MARINE[cls]
    if cls in _FW_CLS_REMAP_INV:
        return _FW_LABEL.get(_FW_CLS_REMAP_INV[cls], f"class_{cls}")
    return _COCO.get(cls, f"class_{cls}")


def label_for_model(model: str, raw_cls: int) -> Tuple[str, int]:
    """Return ``(canonical_label, effective_cls_id)`` for a raw detection.

    ``model`` is the active ``detector.model`` value. For the COCO model the
    effective class id *is* the raw COCO id; for forward-watch the raw id is
    remapped to a synthetic id (81+) so the ``coco_class`` field on the wire
    never collides with a real COCO id.
    """
    if model == MODEL_FORWARD_WATCH:
        label = _FW_LABEL.get(raw_cls, f"class_{raw_cls}")
        eff = _FW_CLS_REMAP.get(raw_cls, 80 + raw_cls)
        return label, eff
    # Default: COCO path.
    label = _MARINE.get(raw_cls, _COCO.get(raw_cls, f"class_{raw_cls}"))
    return label, raw_cls


def is_person_in_water(label: str, waterline_y: float, horizon_y: Optional[float]) -> bool:
    """A person whose waterline (bbox bottom) sits *below* the horizon is in the
    water.

    Without a calibrated horizon we cannot make the call, so we return False
    (the plugin still receives the person detection and can decide).
    """
    if label != "person":
        return False
    if horizon_y is None:
        return False
    return waterline_y > horizon_y
