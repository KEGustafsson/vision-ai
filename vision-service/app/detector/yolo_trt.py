"""TensorRT YOLOv8 backend for the Jetson Orin Nano.

Ultralytics loads a ``.engine`` file the same way as a ``.pt`` file, so this is
a thin specialisation of :class:`YoloTorchDetector` pinned to device 0 with the
exported engine. Build the engine on the Jetson with ``scripts/export_engine.py``.
"""

from __future__ import annotations

from .yolo_torch import YoloTorchDetector


class YoloTrtDetector(YoloTorchDetector):
    def __init__(self, engine: str, confidence: float = 0.35, imgsz: int = 640,
                 tracker_cfg: str = "bytetrack.yaml", batch_cameras: bool = False,
                 batch_wait_ms: int = 20):
        super().__init__(weights=engine, device="0", confidence=confidence,
                         imgsz=imgsz, tracker_cfg=tracker_cfg, backend_name="tensorrt",
                         batch_cameras=batch_cameras, batch_wait_ms=batch_wait_ms)
