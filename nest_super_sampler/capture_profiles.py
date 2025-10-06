"""Capture metadata helpers to drive adaptive processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .config import LightingCondition, PipelineConfig


@dataclass(frozen=True)
class CaptureProfile:
    """Encapsulate capture metadata used by adaptive filters."""

    fps: float
    lighting: LightingCondition

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

    def detail_gain(self) -> float:
        """Return a multiplier used to recover edge details."""

        base_gain = 0.8
        return min(1.6, base_gain + 0.25 * self.motion_blur_strength())


def derive_capture_profile(cfg: PipelineConfig, info: Dict[str, float]) -> CaptureProfile:
    """Create a :class:`CaptureProfile` from CLI configuration and reader info."""

    fps = cfg.capture_fps_hint or info.get("fps", 0.0) or 30.0
    if cfg.lighting is LightingCondition.AUTO:
        lighting = LightingCondition.DAYTIME if fps >= 20.0 else LightingCondition.NIGHT
    else:
        lighting = cfg.lighting
    return CaptureProfile(fps=fps, lighting=lighting)
