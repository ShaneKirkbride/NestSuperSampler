"""Factory helpers for assembling the pipeline from configuration."""

from __future__ import annotations

from typing import Optional

from .config import PipelineConfig
from .capture_profiles import derive_capture_profile
from .detail_models import DaytimeDetailModel, fit_daytime_model_from_path
from .pipeline import SuperSamplingPipeline
from .postprocessors import build_postprocessor
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
    capture_profile = derive_capture_profile(cfg, reader.info())
    daytime_model: Optional[DaytimeDetailModel] = None
    if cfg.enable_daytime_detail_transfer:
        if cfg.daytime_model_path is not None:
            daytime_model = DaytimeDetailModel.load(cfg.daytime_model_path)
        elif cfg.daytime_reference_media is not None:
            daytime_model = fit_daytime_model_from_path(cfg.daytime_reference_media)
        else:
            raise ValueError(
                "Daytime transfer enabled but no --daytime-model or --daytime-reference provided"
            )
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
        cfg.enable_motion_compensation,
        cfg.enable_daytime_detail_transfer,
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
        motion_profile=capture_profile if cfg.enable_motion_compensation else None,
        daytime_model=daytime_model,
    )
    postprocessor = build_postprocessor(cfg)
    return SuperSamplingPipeline(
        reader,
        sampler,
        supersampler,
        sink,
        cfg.max_frames,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
