"""Map raw model class ids to canonical marine labels, and apply the
person-in-water (man-overboard candidate) rule.

COCO ids relevant to the marine domain: 0=person, 8=boat. Class id 80 is an
extension we reserve for "buoy" (from a maritime-trained model or the synthetic
source). Everything else falls through to its COCO name where known.
"""

from __future__ import annotations

from typing import Optional

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


def label_for(cls: int) -> str:
    if cls in _MARINE:
        return _MARINE[cls]
    return _COCO.get(cls, f"class_{cls}")


def is_person_in_water(label: str, cy: float, horizon_y: Optional[float]) -> bool:
    """A person whose centroid sits *below* the horizon is in the water.

    Without a calibrated horizon we cannot make the call, so we return False
    (the plugin still receives the person detection and can decide).
    """
    if label != "person":
        return False
    if horizon_y is None:
        return False
    return cy > horizon_y
