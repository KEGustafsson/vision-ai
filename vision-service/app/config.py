"""Configuration loading: layered YAML + environment overrides.

Precedence (lowest to highest): ``config/default.yaml`` -> the mode-specific
file (``config/<mode>.yaml``) -> ``VISION_*`` environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class CameraConfig(BaseModel):
    name: str
    url: Optional[str] = None          # RTSP url (jetson/cpu modes)
    hfov_deg: float = 90.0             # horizontal field of view
    height_m: float = 2.5             # camera height above waterline
    horizon_y: Optional[float] = None  # pixel row of horizon; None => auto/uncalibrated
    bearing_offset_deg: float = 0.0   # mounting offset from bow (forward=0, aft=180)


class GeometryConfig(BaseModel):
    auto_horizon: bool = False
    # Known real-world widths (metres) per canonical label, for known-size ranging.
    known_widths_m: Dict[str, float] = Field(
        default_factory=lambda: {"person": 0.5, "buoy": 0.8, "vessel": 4.0}
    )


class DetectorConfig(BaseModel):
    backend: str = "mock"              # mock | torch-cpu | torch-cuda | tensorrt
    model_pt: str = "yolov8n.pt"
    model_engine: str = "yolov8n.engine"
    confidence: float = 0.35
    imgsz: int = 640
    tracker: str = "bytetrack.yaml"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    target_fps: float = 10.0           # processing cadence cap
    jpeg_quality: int = 80
    event_buffer: int = 200


class Settings(BaseModel):
    mode: str = "mock"                 # mock | cpu | jetson
    mock_source: str = "synthetic"     # synthetic | <video path>
    cameras: List[CameraConfig]
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    def camera(self, name: str) -> CameraConfig:
        for c in self.cameras:
            if c.name == name:
                return c
        raise KeyError(f"unknown camera: {name}")


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _apply_env(raw: dict) -> dict:
    """Overlay VISION_* environment variables onto the raw config dict."""
    env = os.environ
    if "VISION_MODE" in env:
        raw["mode"] = env["VISION_MODE"]
    if "VISION_MOCK_SOURCE" in env:
        raw["mock_source"] = env["VISION_MOCK_SOURCE"]
    if "VISION_PORT" in env:
        raw.setdefault("server", {})["port"] = int(env["VISION_PORT"])
    det = raw.setdefault("detector", {})
    if "VISION_MODEL_PT" in env:
        det["model_pt"] = env["VISION_MODEL_PT"]
    if "VISION_MODEL_ENGINE" in env:
        det["model_engine"] = env["VISION_MODEL_ENGINE"]
    # Per-camera URL overrides
    cams = {c["name"]: c for c in raw.get("cameras", [])}
    if "VISION_CAMERA_FORWARD_URL" in env and "forward" in cams:
        cams["forward"]["url"] = env["VISION_CAMERA_FORWARD_URL"]
    if "VISION_CAMERA_AFT_URL" in env and "aft" in cams:
        cams["aft"]["url"] = env["VISION_CAMERA_AFT_URL"]
    return raw


def load_settings(mode: Optional[str] = None) -> Settings:
    base = _load_yaml(CONFIG_DIR / "default.yaml")
    mode = mode or os.environ.get("VISION_MODE") or base.get("mode", "mock")
    merged = _deep_merge(base, _load_yaml(CONFIG_DIR / f"{mode}.yaml"))
    merged["mode"] = mode
    merged = _apply_env(merged)
    return Settings(**merged)
