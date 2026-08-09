"""Image-motion measurement (NVIDIA Optical Flow Accelerator).

Hardware-independent statistics only: the DeepStream/pyds plumbing lives in
``app/pipeline_deepstream.py``, everything that can be unit tested without a
Jetson lives here.
"""

from .optical_flow import (
    DEFAULT_MAX_MAGNITUDE_PX,
    OF_FIXED_POINT_DIVISOR,
    CameraFlowState,
    OpticalFlowState,
    OpticalFlowStats,
    estimate_global_motion,
    raw_to_px,
)

__all__ = [
    "CameraFlowState",
    "DEFAULT_MAX_MAGNITUDE_PX",
    "OF_FIXED_POINT_DIVISOR",
    "OpticalFlowState",
    "OpticalFlowStats",
    "estimate_global_motion",
    "raw_to_px",
]
