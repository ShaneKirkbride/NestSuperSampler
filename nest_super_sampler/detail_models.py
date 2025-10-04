"""Daytime detail transfer models for guiding night restoration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np

from .video_reader import OpenCVVideoReader

_SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)


def _mean_gradient(gray: np.ndarray) -> float:
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return float(np.mean(magnitude))


def _mean_laplacian(gray: np.ndarray) -> float:
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(np.mean(np.abs(laplacian)))


def _collect_reference_frames(path: Path, max_frames: int = 180) -> List[np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Reference media not found: {path}")
    if path.is_dir():
        frames: List[np.ndarray] = []
        for image_path in sorted(path.iterdir()):
            if image_path.suffix.lower() not in _SUPPORTED_IMAGE_SUFFIXES:
                continue
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        if not frames:
            raise ValueError(f"No readable images found in {path}")
        return frames
    reader = OpenCVVideoReader(path)
    info = reader.info()
    total_frames = int(info.get("frame_count", 0) or 0)
    stride = max(1, total_frames // max_frames) if total_frames else 5
    collected: List[np.ndarray] = []
    for idx, _, frame in reader.frames():
        if idx % stride != 0:
            continue
        collected.append(frame)
        if len(collected) >= max_frames:
            break
    if not collected:
        raise ValueError(f"No frames collected from video {path}")
    return collected


@dataclass(frozen=True)
class DaytimeDetailModel:
    """Encapsulate gradient statistics from sharp daytime captures."""

    mean_gradient: float
    mean_laplacian_abs: float
    base_alpha: float
    base_sigma: float

    def estimate_parameters(self, gradient: float, laplacian_abs: float) -> tuple[float, float]:
        """Return adaptive (alpha, sigma) sharpening parameters."""

        gradient_ratio = self.mean_gradient / max(gradient, 1e-3)
        laplacian_ratio = self.mean_laplacian_abs / max(laplacian_abs, 1e-3)
        blend = 0.55 * gradient_ratio + 0.45 * laplacian_ratio
        gain = float(np.clip(blend, 0.85, 3.0))
        alpha = float(np.clip(self.base_alpha * gain, 0.35, 2.5))
        sigma = float(np.clip(self.base_sigma / np.sqrt(gain), 0.45, 1.8))
        return alpha, sigma

    def to_dict(self) -> dict:
        return {
            "mean_gradient": self.mean_gradient,
            "mean_laplacian_abs": self.mean_laplacian_abs,
            "base_alpha": self.base_alpha,
            "base_sigma": self.base_sigma,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DaytimeDetailModel":
        return cls(
            mean_gradient=float(payload["mean_gradient"]),
            mean_laplacian_abs=float(payload["mean_laplacian_abs"]),
            base_alpha=float(payload["base_alpha"]),
            base_sigma=float(payload["base_sigma"]),
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            mean_gradient=self.mean_gradient,
            mean_laplacian_abs=self.mean_laplacian_abs,
            base_alpha=self.base_alpha,
            base_sigma=self.base_sigma,
        )

    @classmethod
    def load(cls, path: Path) -> "DaytimeDetailModel":
        if not path.exists():
            raise FileNotFoundError(f"Daytime model not found: {path}")
        data = np.load(path)
        return cls(
            mean_gradient=float(data["mean_gradient"]),
            mean_laplacian_abs=float(data["mean_laplacian_abs"]),
            base_alpha=float(data["base_alpha"]),
            base_sigma=float(data["base_sigma"]),
        )


def fit_daytime_detail_model(frames: Iterable[np.ndarray]) -> DaytimeDetailModel:
    gradients: List[float] = []
    laplacians: List[float] = []
    for frame in frames:
        gradient, laplacian = analyze_frame(frame)
        gradients.append(gradient)
        laplacians.append(laplacian)
    if not gradients:
        raise ValueError("At least one frame is required to build a daytime detail model")
    mean_grad = float(np.mean(gradients))
    mean_lap = float(np.mean(laplacians))
    base_alpha = float(np.clip(0.45 + 0.0025 * mean_lap, 0.45, 1.8))
    base_sigma = float(np.clip(1.1 - 0.002 * mean_grad, 0.55, 1.4))
    return DaytimeDetailModel(mean_grad, mean_lap, base_alpha, base_sigma)


def fit_daytime_model_from_path(path: Path, max_frames: int = 180) -> DaytimeDetailModel:
    frames = _collect_reference_frames(path, max_frames=max_frames)
    return fit_daytime_detail_model(frames)


def analyze_frame(frame_bgr: np.ndarray) -> tuple[float, float]:
    """Return (gradient_mean, laplacian_abs_mean) metrics for a frame."""

    gray = _to_gray(frame_bgr)
    return _mean_gradient(gray), _mean_laplacian(gray)
