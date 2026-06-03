"""Map raw model class ids to canonical marine labels, and apply the
person-in-water (man-overboard candidate) rule.

Exactly ONE detection model is active at a time (selected via
``detector.model``). Supported models:

  ``"coco"`` — **COCO YOLOv8n** (80 classes, the default)
    COCO ids relevant to the marine domain: 0=person, 8=boat.  Class id 80
    is an extension we reserve for "buoy".

  ``"forward-watch"`` — **forward-watch** (6 marine-specific classes)
    0=ship, 1=boat, 2=debris, 3=buoy, 4=kayak, 5=log.

  ``"marine-surveillance"`` — **Roboflow Marine Surveillance YOLOv8s**
    (7 marine classes, kept DISTINCT): boat, buoy, kayak, sailboat, speedboat,
    vessel, warship. Trained on-box from the Roboflow dataset; see
    training/train_marine_surveillance.py. NO person class (no man-overboard).

Each non-COCO model's raw class ids are remapped to a private synthetic-id band
(forward-watch → 81–86, marine-surveillance → 87–93) so the ``coco_class`` field
on the wire stays unambiguous regardless of which model produced the detection.
"""

from __future__ import annotations

from typing import Optional, Tuple

MODEL_COCO = "coco"
MODEL_FORWARD_WATCH = "forward-watch"
MODEL_MARINE_SURVEILLANCE = "marine-surveillance"

# Model name -> nvinfer config file (relative to the deepstream/ directory).
MODEL_PGIE_CONFIG = {
    MODEL_COCO: "pgie_yolov8n.txt",
    MODEL_FORWARD_WATCH: "pgie_forward_watch.txt",
    MODEL_MARINE_SURVEILLANCE: "pgie_marine_surveillance.txt",
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

# marine-surveillance raw class id → canonical label. The 7 classes are kept
# DISTINCT (no collapsing into "vessel"). Class order is the dataset's
# data.yaml ``names`` — Roboflow defaults to ALPHABETICAL, which is asserted by
# training/train_marine_surveillance.py before export, so this stays in sync.
_MS_LABEL = {
    0: "boat",
    1: "buoy",
    2: "kayak",
    3: "sailboat",
    4: "speedboat",
    5: "vessel",
    6: "warship",
}

# marine-surveillance raw class id → synthetic id (87–93, after forward-watch).
_MS_CLS_REMAP = {i: 87 + i for i in range(7)}

# Registry of non-COCO models: name -> (raw-id→label, raw-id→synthetic-id).
_MODEL_TABLES = {
    MODEL_FORWARD_WATCH: (_FW_LABEL, _FW_CLS_REMAP),
    MODEL_MARINE_SURVEILLANCE: (_MS_LABEL, _MS_CLS_REMAP),
}

# Canonical labels each model can produce — derived from the label tables above.
MODEL_LABELS = {
    MODEL_COCO: sorted(set(_MARINE.values())),
    **{name: sorted(set(lbl.values())) for name, (lbl, _) in _MODEL_TABLES.items()},
}

# Global inverse: synthetic id → canonical label, across every non-COCO model.
_SYNTH_INV = {
    eff: lbl[raw]
    for lbl, remap in _MODEL_TABLES.values()
    for raw, eff in remap.items()
}


def label_for(cls: int) -> str:
    """Canonical label for a (possibly already-remapped) class id.

    Accepts COCO ids and any model's synthetic ids (81+) so callers holding only
    the wire ``coco_class`` can still resolve a label.
    """
    if cls in _MARINE:
        return _MARINE[cls]
    if cls in _SYNTH_INV:
        return _SYNTH_INV[cls]
    return _COCO.get(cls, f"class_{cls}")


def label_for_model(model: str, raw_cls: int) -> Tuple[str, int]:
    """Return ``(canonical_label, effective_cls_id)`` for a raw detection.

    ``model`` is the active ``detector.model`` value. For the COCO model the
    effective class id *is* the raw COCO id; for every other model the raw id is
    remapped into that model's private synthetic-id band so the ``coco_class``
    field on the wire never collides with a real COCO id.
    """
    if model in _MODEL_TABLES:
        labels, remap = _MODEL_TABLES[model]
        label = labels.get(raw_cls, f"class_{raw_cls}")
        eff = remap.get(raw_cls, min(remap.values()) + raw_cls)
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
