import cv2
import numpy as np

from nest_super_sampler.preprocessors import (
    Deblurrer,
    Denoiser,
    FrameProcessingPipeline,
    GlareReducer,
    build_preprocessor,
)


def _make_bright_image() -> np.ndarray:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :] = [10, 10, 250]
    return img


def test_glare_reducer_lowers_highlights() -> None:
    reducer = GlareReducer(v_thresh=0.5, s_thresh=1.0, knee_tau=0.2, knee_strength=10.0, dilate_px=0)
    src = _make_bright_image()
    result = reducer.process(src)
    assert result.dtype == np.uint8
    assert result[..., 2].mean() < src[..., 2].mean()


def test_denoiser_runs_without_error() -> None:
    denoiser = Denoiser(h_luma=3, h_chroma=3, template=3, search=7)
    noisy = np.random.randint(0, 255, size=(12, 12, 3), dtype=np.uint8)
    result = denoiser.process(noisy)
    assert result.shape == noisy.shape


def test_deblurrer_increases_contrast() -> None:
    deblurrer = Deblurrer(alpha=0.8, sigma=0.5)
    frame = np.tile(np.linspace(0, 255, 20, dtype=np.uint8), (20, 1))
    frame = cv2.merge([frame, frame, frame])
    sharpened = deblurrer.process(frame)
    assert sharpened.dtype == np.uint8
    assert sharpened.var() >= frame.var()


def test_frame_processing_pipeline_chains_processors() -> None:
    class AddOneProcessor:
        def process(self, frame_bgr: np.ndarray) -> np.ndarray:
            return frame_bgr + 1

    pipeline = FrameProcessingPipeline([AddOneProcessor(), AddOneProcessor()])
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    result = pipeline.process(frame)
    assert np.array_equal(result, frame + 2)


def test_build_preprocessor_returns_none_when_disabled() -> None:
    pre = build_preprocessor(
        enable_glare=False,
        enable_denoise=False,
        enable_deblur=False,
        glare_v_thresh=0.9,
        glare_s_thresh=0.4,
        glare_knee_tau=0.75,
        glare_knee_strength=3.0,
        glare_dilate_px=2,
        denoise_h_luma=7,
        denoise_h_chroma=5,
        denoise_template=7,
        denoise_search=21,
        deblur_alpha=0.6,
        deblur_sigma=1.2,
    )
    assert pre is None


def test_build_preprocessor_returns_pipeline_when_enabled() -> None:
    pre = build_preprocessor(
        enable_glare=True,
        enable_denoise=True,
        enable_deblur=True,
        glare_v_thresh=0.9,
        glare_s_thresh=0.4,
        glare_knee_tau=0.75,
        glare_knee_strength=3.0,
        glare_dilate_px=2,
        denoise_h_luma=7,
        denoise_h_chroma=5,
        denoise_template=7,
        denoise_search=21,
        deblur_alpha=0.6,
        deblur_sigma=1.2,
    )
    assert isinstance(pre, FrameProcessingPipeline)
