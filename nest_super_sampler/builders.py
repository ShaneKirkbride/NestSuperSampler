"""Factory helpers for assembling the pipeline from configuration."""

from __future__ import annotations

from .config import PipelineConfig
from .pipeline import SuperSamplingPipeline
from .preprocessors import build_preprocessor
from .samplers import CompositeSampler, NthFrameSampler, SceneChangeSampler
from .sinks import DiskImageSink
from .supersamplers import build_supersampler
from .video_reader import OpenCVVideoReader


def build_sampler(cfg: PipelineConfig) -> CompositeSampler:
    """Instantiate the frame sampler pipeline."""

    samplers = [NthFrameSampler(cfg.sample_every_n_frames)]
    if cfg.enable_scene_change_sampling:
        samplers.append(
            SceneChangeSampler(cfg.scene_hist_threshold, cfg.scene_min_gap_frames)
        )
    return CompositeSampler(samplers)


def build_pipeline(cfg: PipelineConfig) -> SuperSamplingPipeline:
    """Assemble the full :class:`SuperSamplingPipeline`."""

    reader = OpenCVVideoReader(cfg.input_video)
    sampler = build_sampler(cfg)
    supersampler = build_supersampler(cfg)
    sink = DiskImageSink(
        cfg.output_dir,
        cfg.image_format,
        cfg.jpg_quality,
        cfg.write_full_size_only,
    )
    preprocessor = build_preprocessor(
        cfg.enable_glare,
        cfg.enable_denoise,
        cfg.enable_deblur,
        glare_v_thresh=cfg.glare_v_thresh,
        glare_s_thresh=cfg.glare_s_thresh,
        glare_knee_tau=cfg.glare_knee_tau,
        glare_knee_strength=cfg.glare_knee_strength,
        glare_dilate_px=cfg.glare_dilate_px,
        denoise_h_luma=cfg.denoise_h_luma,
        denoise_h_chroma=cfg.denoise_h_chroma,
        denoise_template=cfg.denoise_template,
        denoise_search=cfg.denoise_search,
        deblur_alpha=cfg.deblur_alpha,
        deblur_sigma=cfg.deblur_sigma,
    )
    return SuperSamplingPipeline(
        reader,
        sampler,
        supersampler,
        sink,
        cfg.max_frames,
        preprocessor=preprocessor,
    )
