import cv2
import numpy as np

from nest_super_sampler.preprocessors import (
    DayNightDetailTransfer,
    Deblurrer,
    Denoiser,
    FrameProcessingPipeline,
    GlareReducer,
    build_preprocessor,
    MotionAwareDeblurrer,
)
from nest_super_sampler.capture_profiles import CaptureProfile
from nest_super_sampler.config import LightingCondition
from nest_super_sampler.detail_models import (
    DaytimeDetailModel,
    analyze_frame,
    fit_daytime_detail_model,
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
        enable_motion_compensation=False,
        enable_daytime_detail_transfer=False,
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
        enable_motion_compensation=True,
        enable_daytime_detail_transfer=False,
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
        motion_profile=CaptureProfile(fps=30.0, lighting=LightingCondition.DAYTIME),
    )
    assert isinstance(pre, FrameProcessingPipeline)


def test_motion_aware_deblurrer_enhances_edges() -> None:
    profile = CaptureProfile(fps=15.0, lighting=LightingCondition.DAYTIME)
    processor = MotionAwareDeblurrer(profile)
    frame = np.full((20, 20, 3), 127, dtype=np.uint8)
    cv2.line(frame, (0, 10), (19, 10), (127, 127, 127), 3)
    blurred = cv2.GaussianBlur(frame, (0, 0), 2.5)
    restored = processor.process(blurred)
    assert restored.dtype == np.uint8
    assert restored.var() >= blurred.var()


def test_daytime_detail_model_roundtrip(tmp_path) -> None:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.rectangle(frame, (4, 4), (28, 28), (255, 255, 255), 2)
    model = fit_daytime_detail_model([frame])
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = DaytimeDetailModel.load(path)
    assert loaded == model


def test_daynight_detail_transfer_increases_gradient() -> None:
    reference = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.rectangle(reference, (6, 6), (26, 26), (255, 255, 255), 2)
    cv2.line(reference, (0, 16), (31, 16), (255, 255, 255), 1)
    model = fit_daytime_detail_model([reference])
    transfer = DayNightDetailTransfer(model)
    night_frame = cv2.GaussianBlur(reference, (0, 0), 2.2)
    restored = transfer.process(night_frame)
    blurred_gradient, _ = analyze_frame(night_frame)
    restored_gradient, _ = analyze_frame(restored)
    assert restored_gradient > blurred_gradient
