"""Lens correction (barrel + mount-level rotation) for the vision pipeline.

Used two ways (see pipeline): display-only on the annotated JPEG, or — when
``undistort_before_detect`` is set — on the frame fed to the detector.

Barrel undistortion and the leveling rotation are fused into a *single* sampling
map (raw-source coord per output pixel), so each frame is one resample instead
of a remap followed by a separate rotation warp. That map runs either on the CPU
(``cv2.remap``) or, when torch CUDA is available, on the GPU via
``grid_sample`` — the full-frame resample is ~63 ms on this Jetson's CPU but a
few ms on the Orin GPU.

Detection-box and horizon coordinates are mapped through the same transform
(``points``) so an overlay drawn in raw-frame coords still lands correctly after
correction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from ..config import CameraConfig

# Sentinel source coord for output pixels that fall outside the source image
# after the rotation: far out of range so both backends pad them black.
_OOB = -1.0e4


class Undistorter:
    def __init__(self, cam: "CameraConfig", width: int, height: int,
                 use_gpu: bool | None = None):
        self.size = (width, height)
        f = cam.undistort_f_factor * width
        self.K = np.array([[f, 0, width / 2.0],
                           [0, f, height / 2.0],
                           [0, 0, 1]], dtype=np.float64)
        self.D = np.array([cam.undistort_k1, 0, 0, 0, 0], dtype=np.float64)
        self.newK, _ = cv2.getOptimalNewCameraMatrix(
            self.K, self.D, self.size, cam.undistort_alpha, self.size)
        self.rotation_deg = float(cam.undistort_rotation_deg)
        # Rotation about the image centre (== principal point, since K puts it
        # there). Same matrix maps points() so the overlay stays aligned.
        self.R = (cv2.getRotationMatrix2D((width / 2.0, height / 2.0),
                                          self.rotation_deg, 1.0)
                  if self.rotation_deg else None)

        # Fused float map: for each OUTPUT pixel, the source coord in the raw
        # frame (barrel + rotation in one). m1f/m2f are undistort-only; if there
        # is a rotation, resample them through the inverse rotation so the single
        # map reproduces "undistort then rotate" exactly.
        m1f, m2f = cv2.initUndistortRectifyMap(
            self.K, self.D, None, self.newK, self.size, cv2.CV_32FC1)
        if self.R is not None:
            minv = cv2.invertAffineTransform(self.R)
            ys, xs = np.indices((height, width), dtype=np.float32)
            ux = minv[0, 0] * xs + minv[0, 1] * ys + minv[0, 2]
            uy = minv[1, 0] * xs + minv[1, 1] * ys + minv[1, 2]
            self.map_x = cv2.remap(m1f, ux, uy, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=_OOB)
            self.map_y = cv2.remap(m2f, ux, uy, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=_OOB)
        else:
            self.map_x, self.map_y = m1f, m2f

        # Optional GPU backend (torch grid_sample). Falls back to CPU silently if
        # torch/CUDA isn't present, so dev/mock hosts keep working.
        self._torch = None
        self._grid = None
        if use_gpu is None:
            use_gpu = bool(getattr(cam, "undistort_gpu", True))
        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    gx = self.map_x / (width - 1) * 2.0 - 1.0
                    gy = self.map_y / (height - 1) * 2.0 - 1.0
                    grid = np.stack([gx, gy], axis=-1)[None]  # 1,H,W,2
                    self._grid = torch.from_numpy(grid).float().cuda()
                    self._torch = torch
            except Exception:
                self._torch = None  # CPU fallback

    @property
    def backend(self) -> str:
        return "gpu" if self._torch is not None else "cpu"

    def image(self, img: np.ndarray) -> np.ndarray:
        if self._torch is not None:
            torch = self._torch
            with torch.no_grad():
                t = (torch.from_numpy(np.ascontiguousarray(img)).cuda()
                     .permute(2, 0, 1).unsqueeze(0).float())
                out = torch.nn.functional.grid_sample(
                    t, self._grid, mode="bilinear",
                    padding_mode="zeros", align_corners=True)
                return (out.squeeze(0).permute(1, 2, 0)
                        .clamp_(0, 255).to(torch.uint8).cpu().numpy())
        return cv2.remap(img, self.map_x, self.map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

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
