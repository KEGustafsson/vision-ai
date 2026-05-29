"""Camera source factory — picks a backend from the active mode/config."""

from __future__ import annotations

import os

from ..config import CameraConfig, Settings
from .base import Frame, FrameSource, TruthBox


def create_source(cfg: CameraConfig, settings: Settings) -> FrameSource:
    mode = settings.mode
    if mode == "mock":
        source = settings.mock_source
        if source and source != "synthetic" and os.path.exists(source):
            from .video_file import VideoFileSource
            return VideoFileSource(cfg.name, source)
        from .synthetic import SyntheticSource
        # Drop the man-overboard actor into the aft camera for demoability.
        return SyntheticSource(
            cfg.name, with_mob=(cfg.name == "aft"), fps=settings.server.target_fps
        )

    if mode == "cpu":
        if settings.mock_source and settings.mock_source != "synthetic" \
                and os.path.exists(settings.mock_source):
            from .video_file import VideoFileSource
            return VideoFileSource(cfg.name, settings.mock_source)
        if cfg.url:
            from .rtsp_cpu import RtspCpuSource
            return RtspCpuSource(cfg.name, cfg.url)
        from .synthetic import SyntheticSource
        return SyntheticSource(cfg.name, with_mob=(cfg.name == "aft"))

    if mode == "jetson":
        if not cfg.url:
            raise RuntimeError(f"camera {cfg.name} has no RTSP url configured")
        from .rtsp_gstreamer import RtspGstreamerSource
        return RtspGstreamerSource(cfg.name, cfg.url)

    raise ValueError(f"unknown mode: {mode}")


__all__ = ["Frame", "FrameSource", "TruthBox", "create_source"]
