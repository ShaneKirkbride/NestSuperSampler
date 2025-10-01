"""Frame post-processing operators applied after super-resolution."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .config import PipelineConfig
from .interfaces import IPostProcessor


class Sharpen(IPostProcessor):
    """Edge-aware unsharp masking applied to BGR frames."""

    def __init__(
        self,
        alpha: float = 0.35,
        sigma: float = 1.0,
        edge_clip: float = 0.015,
    ) -> None:
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.edge_clip = float(edge_clip)

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        base = frame_bgr.astype(np.float32) / 255.0

        ycrcb = cv2.cvtColor(base, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]

        laplacian = cv2.Laplacian(y_channel, cv2.CV_32F, ksize=3)
        edge_mag = np.abs(laplacian)
        max_val = float(edge_mag.max())
        if max_val > 1e-6:
            edge_norm = edge_mag / max_val
        else:
            edge_norm = np.zeros_like(edge_mag)

        clip = float(np.clip(self.edge_clip, 0.0, 1.0))
        denom = max(1e-6, 1.0 - clip)
        mask = np.clip((edge_norm - clip) / denom, 0.0, 1.0)

        blur_sigma = max(self.sigma, 1e-6)
        blurred = cv2.GaussianBlur(base, (0, 0), blur_sigma)
        sharpened = np.clip(base + self.alpha * (base - blurred), 0.0, 1.0)

        mask = mask[:, :, None]
        output = sharpened * mask + base * (1.0 - mask)
        return np.clip(output * 255.0, 0.0, 255.0).astype(np.uint8)


def build_postprocessor(cfg: PipelineConfig) -> Optional[IPostProcessor]:
    """Create the configured post-processor, if any."""

    if not cfg.enable_sharpen:
        return None
    return Sharpen(cfg.sharpen_alpha, cfg.sharpen_sigma, cfg.sharpen_edge_clip)

