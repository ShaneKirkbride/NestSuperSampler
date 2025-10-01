"""Configuration models for the Nest Super Sampler pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class SuperResAlgo(str, Enum):
    """Supported super-resolution algorithms."""

    EDSR = "edsr"
    ESPCN = "espcn"
    FSRCNN = "fsrcnn"
    LAPSRN = "lapsrn"
    BICUBIC = "bicubic"  # fallback


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable container with pipeline configuration options."""

    # IO
    input_video: Path
    output_dir: Path

    # Sampling
    sample_every_n_frames: int = 15
    enable_scene_change_sampling: bool = True
    scene_hist_threshold: float = 0.30
    scene_min_gap_frames: int = 20

    # Super-resolution
    upscale_factor: int = 2
    algo: SuperResAlgo = SuperResAlgo.EDSR
    model_dir: Optional[Path] = None

    # Export
    image_format: str = "png"
    jpg_quality: int = 95
    write_full_size_only: bool = True

    # Limits
    max_frames: Optional[int] = None
    fail_on_missing_superres: bool = False

    # Preprocessing toggles
    enable_glare: bool = False
    enable_denoise: bool = False
    enable_deblur: bool = False

    # Glare params
    glare_v_thresh: float = 0.90
    glare_s_thresh: float = 0.40
    glare_knee_tau: float = 0.75
    glare_knee_strength: float = 3.0
    glare_dilate_px: int = 2

    # Denoise params
    denoise_h_luma: int = 7
    denoise_h_chroma: int = 5
    denoise_template: int = 7
    denoise_search: int = 21

    # Deblur params
    deblur_alpha: float = 0.6
    deblur_sigma: float = 1.2

    # Sharpen params
    enable_sharpen: bool = False
    sharpen_alpha: float = 0.35
    sharpen_sigma: float = 1.0
    sharpen_edge_clip: float = 0.015
