"""Display-only lens correction for the annotated stream.

Straightens barrel distortion and levels a tilted mount in the MJPEG/snapshot
JPEG ONLY. Detection geometry (bearing/range/CPA) is computed upstream from the
raw frame and is never routed through here, so this is purely cosmetic and does
not require a metric calibration.

The remap is precomputed once per (camera, frame-size) into a lookup table, so
per-frame cost is one ``cv2.remap`` (and one rotation when the mount is tilted).
Detection-box and horizon coordinates are mapped through the *same* transform so
the overlay still lands on the correct pixels after correction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from ..config import CameraConfig


class Undistorter:
    def __init__(self, cam: "CameraConfig", width: int, height: int):
        self.size = (width, height)
        f = cam.undistort_f_factor * width
        self.K = np.array([[f, 0, width / 2.0],
                           [0, f, height / 2.0],
                           [0, 0, 1]], dtype=np.float64)
        self.D = np.array([cam.undistort_k1, 0, 0, 0, 0], dtype=np.float64)
        self.newK, _ = cv2.getOptimalNewCameraMatrix(
            self.K, self.D, self.size, cam.undistort_alpha, self.size)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, self.D, None, self.newK, self.size, cv2.CV_16SC2)
        self.rotation_deg = float(cam.undistort_rotation_deg)
        # Forward (src->dst) rotation about the image centre; the same matrix is
        # applied to the image (warpAffine) and to points (M . [x, y, 1]).
        self.R = (cv2.getRotationMatrix2D((width / 2.0, height / 2.0),
                                          self.rotation_deg, 1.0)
                  if self.rotation_deg else None)

    def image(self, img: np.ndarray) -> np.ndarray:
        out = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
        if self.R is not None:
            out = cv2.warpAffine(out, self.R, self.size, flags=cv2.INTER_LINEAR)
        return out

    def points(self, pts: np.ndarray) -> np.ndarray:
        """Map raw-frame pixel coords (Nx2) into corrected-display coords."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts, self.K, self.D, P=self.newK).reshape(-1, 2)
        if self.R is not None:
            aug = np.hstack([und, np.ones((len(und), 1))])
            und = (self.R @ aug.T).T
        return und

    def bbox(self, x: float, y: float, w: float, h: float) -> tuple:
        """Map a raw-frame bbox to an axis-aligned bbox in display coords."""
        corners = self.points([(x, y), (x + w, y), (x, y + h), (x + w, y + h)])
        x0, y0 = corners.min(axis=0)
        x1, y1 = corners.max(axis=0)
        return float(x0), float(y0), float(x1 - x0), float(y1 - y0)

    def horizon_y(self, y: float, width: int) -> float:
        """Map the horizon row (sampled at frame centre) to a display row."""
        return float(self.points([(width / 2.0, y)])[0, 1])
