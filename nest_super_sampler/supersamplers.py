"""Super-resolution implementations."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from .config import PipelineConfig, SuperResAlgo
from .interfaces import ISuperSampler


class BicubicSuperSampler(ISuperSampler):
    """Fallback sampler using OpenCV bicubic resize."""

    def __init__(self, scale: int) -> None:
        if scale <= 0:
            raise ValueError("scale must be > 0")
        self._scale = scale

    def upscale(self, img_bgr: np.ndarray) -> np.ndarray:
        height, width = img_bgr.shape[:2]
        return cv2.resize(
            img_bgr,
            (width * self._scale, height * self._scale),
            interpolation=cv2.INTER_CUBIC,
        )


class DNNSuperSampler(ISuperSampler):
    """DNN-based super-resolution using ``cv2.dnn_superres``."""

    def __init__(self, algo: SuperResAlgo, scale: int, model_dir: Path) -> None:
        if scale <= 0:
            raise ValueError("scale must be > 0")
        try:
            from cv2.dnn_superres import DnnSuperResImpl_create
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "OpenCV contrib (dnn_superres) not available. Install opencv-contrib-python."
            ) from exc
        self._sr = DnnSuperResImpl_create()
        algo_name = algo.value.lower()
        self._sr.setModel(algo_name, scale)
        model_filename = f"{algo_name.upper()}_x{scale}.pb"
        model_path = (model_dir / model_filename).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model {model_path}")
        self._sr.readModel(str(model_path))

    def upscale(self, img_bgr: np.ndarray) -> np.ndarray:
        return self._sr.upsample(img_bgr)


def build_supersampler(cfg: PipelineConfig) -> ISuperSampler:
    """Create the appropriate super sampler based on :class:`PipelineConfig`."""

    if cfg.algo == SuperResAlgo.BICUBIC:
        return BicubicSuperSampler(cfg.upscale_factor)
    if cfg.model_dir is None:
        if cfg.fail_on_missing_superres:
            raise ValueError("model_dir must be set for DNN superres.")
        return BicubicSuperSampler(cfg.upscale_factor)
    try:
        return DNNSuperSampler(cfg.algo, cfg.upscale_factor, cfg.model_dir)
    except Exception as exc:  # pragma: no cover - safety fallback
        if cfg.fail_on_missing_superres:
            raise
        print(f"[WARN] Superres failed ({exc}), fallback bicubic", file=sys.stderr)
        return BicubicSuperSampler(cfg.upscale_factor)
