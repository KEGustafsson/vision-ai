"""Detector factory keyed on the configured backend."""

from __future__ import annotations

import os
from pathlib import Path

from ..config import Settings
from .base import Detector, RawTrack

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def _resolve(model: str) -> str:
    p = Path(model)
    if p.is_absolute() or p.exists():
        return str(p)
    return str(MODELS_DIR / model)


def create_detector(settings: Settings) -> Detector:
    backend = settings.detector.backend
    det = settings.detector

    if backend == "mock":
        from .mock import MockDetector
        return MockDetector()

    # Feed YOLO the lower floor (not the publish threshold) so the worker-side
    # filter at det.confidence is authoritative and runtime /control is symmetric.
    floor = min(det.track_floor, det.confidence)

    if backend in ("torch-cpu", "torch-cuda"):
        from .yolo_torch import YoloTorchDetector
        device = "cpu" if backend == "torch-cpu" else "0"
        return YoloTorchDetector(_resolve(det.model_pt), device=device,
                                 confidence=floor, imgsz=det.imgsz,
                                 tracker_cfg=det.tracker, backend_name=backend)

    if backend == "tensorrt":
        from .yolo_trt import YoloTrtDetector
        return YoloTrtDetector(_resolve(det.model_engine), confidence=floor,
                               imgsz=det.imgsz, tracker_cfg=det.tracker)

    raise ValueError(f"unknown detector backend: {backend}")


__all__ = ["Detector", "RawTrack", "create_detector"]
