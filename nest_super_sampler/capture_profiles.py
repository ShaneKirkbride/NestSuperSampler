"""Capture metadata helpers to drive adaptive processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import LightingCondition, PipelineConfig


@dataclass(frozen=True)
class CaptureProfile:
    """Encapsulate capture metadata used by adaptive filters."""

    fps: float
    lighting: LightingCondition
    has_daytime_reference: bool = False

    def motion_blur_strength(self) -> float:
        """Return a normalized blur strength estimate in the range [0.5, 3.5]."""

        reference_fps = 30.0
        fps_term = reference_fps / max(self.fps, 1.0)
        lighting_term = {
            LightingCondition.DAYTIME: 0.75,
            LightingCondition.NIGHT: 1.35,
        }.get(self.lighting, 1.0)
        strength = fps_term * lighting_term
        return max(0.5, min(strength, 3.5))

    def exposure_time_ms(self) -> float:
        """Estimate the effective shutter time in milliseconds."""

        base = 1000.0 / max(self.fps, 0.5)
        lighting_multiplier = 1.0 if self.lighting is LightingCondition.DAYTIME else 1.6
        return float(min(1000.0, base * lighting_multiplier))

    def motion_kernel_length(self, pixel_velocity_px: float = 12.0) -> int:
        """Return an odd kernel length approximating the motion blur extent."""

        blur_extent_px = pixel_velocity_px * (self.exposure_time_ms() / 1000.0)
        length = int(round(max(3.0, blur_extent_px)))
        if length % 2 == 0:
            length += 1
        return length

    def motion_kernel(self, pixel_velocity_px: float = 12.0) -> np.ndarray:
        """Return a 2D horizontal motion blur kernel informed by FPS and lighting."""

        length = self.motion_kernel_length(pixel_velocity_px)
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0 / length
        return kernel

    def wiener_balance(self) -> float:
        """Return a noise balance parameter for Wiener deconvolution."""

        base = 0.0025 + 0.0015 * self.motion_blur_strength()
        if self.lighting is LightingCondition.NIGHT:
            base *= 1.45
        if self.has_daytime_reference:
            base *= 0.75
        return float(base)

    def detail_gain(self) -> float:
        """Return a multiplier used to recover edge details."""

        base_gain = 0.8
        gain = base_gain + 0.25 * self.motion_blur_strength()
        if self.has_daytime_reference:
            gain += 0.15
        return min(1.6, gain)


def derive_capture_profile(cfg: PipelineConfig, info: Dict[str, float]) -> CaptureProfile:
    """Create a :class:`CaptureProfile` from CLI configuration and reader info."""

    fps = cfg.capture_fps_hint or info.get("fps", 0.0) or 30.0
    if cfg.lighting is LightingCondition.AUTO:
        lighting = LightingCondition.DAYTIME if fps >= 20.0 else LightingCondition.NIGHT
    else:
        lighting = cfg.lighting
    return CaptureProfile(
        fps=fps,
        lighting=lighting,
        has_daytime_reference=cfg.enable_daytime_detail_transfer,
    )
