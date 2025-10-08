"""Frame pre-processing operators."""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .interfaces import IFrameProcessor
from .capture_profiles import CaptureProfile
from .config import LightingCondition
from .detail_models import DaytimeDetailModel, analyze_frame


class GlareReducer(IFrameProcessor):
    """Reduce glare by compressing highlights in HSV space."""

    def __init__(
        self,
        v_thresh: float = 0.9,
        s_thresh: float = 0.4,
        knee_tau: float = 0.75,
        knee_strength: float = 3.0,
        dilate_px: int = 2,
    ) -> None:
        self.v_thresh = v_thresh
        self.s_thresh = s_thresh
        self.knee_tau = knee_tau
        self.knee_strength = knee_strength
        self.dilate_px = dilate_px

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
        h, s, v = cv2.split(hsv)
        mask = (v > self.v_thresh) & (s < self.s_thresh)
        mask = mask.astype(np.uint8) * 255
        if self.dilate_px > 0:
            kernel = np.ones((self.dilate_px, self.dilate_px), np.uint8)
            mask = cv2.dilate(mask, kernel)
        over = v > self.knee_tau
        v[over] = self.knee_tau + (v[over] - self.knee_tau) / (
            1 + self.knee_strength * (v[over] - self.knee_tau)
        )
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


class Denoiser(IFrameProcessor):
    """Remove chroma/luma noise using non-local means."""

    def __init__(
        self,
        h_luma: int = 7,
        h_chroma: int = 5,
        template: int = 7,
        search: int = 21,
    ) -> None:
        self.h_luma = h_luma
        self.h_chroma = h_chroma
        self.template = template
        self.search = search

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(
            frame_bgr,
            None,
            h=self.h_luma,
            hColor=self.h_chroma,
            templateWindowSize=self.template,
            searchWindowSize=self.search,
        )


class Deblurrer(IFrameProcessor):
    """Sharpen blurry frames by subtracting a Gaussian blur."""

    def __init__(self, alpha: float = 0.6, sigma: float = 1.2) -> None:
        self.alpha = alpha
        self.sigma = sigma

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(frame_bgr, (0, 0), self.sigma)
        return cv2.addWeighted(frame_bgr, 1 + self.alpha, blur, -self.alpha, 0)


class MotionAwareDeblurrer(IFrameProcessor):
    """Adaptive motion deblurring tuned using capture metadata."""

    def __init__(
        self,
        profile: CaptureProfile,
        daytime_model: Optional[DaytimeDetailModel] = None,
    ) -> None:
        self._profile = profile
        self._daytime_model = daytime_model if profile.has_daytime_reference else None

    @staticmethod
    def _wiener_channel(
        channel: np.ndarray, kernel: np.ndarray, balance: float
    ) -> np.ndarray:
        h, w = channel.shape
        pad = np.zeros((h, w), dtype=np.float32)
        kh, kw = kernel.shape
        pad[:kh, :kw] = kernel
        pad = np.roll(np.roll(pad, -kh // 2, axis=0), -kw // 2, axis=1)
        kernel_fft = np.fft.fft2(pad)
        channel_fft = np.fft.fft2(channel.astype(np.float32))
        denom = np.abs(kernel_fft) ** 2 + balance
        wiener = np.conj(kernel_fft) / np.maximum(denom, 1e-6)
        restored = np.fft.ifft2(channel_fft * wiener)
        return np.real(restored).astype(np.float32)

    def _wiener_deblur(self, frame_bgr: np.ndarray) -> np.ndarray:
        kernel = self._profile.motion_kernel()
        kernel /= max(float(kernel.sum()), 1e-6)
        balance = max(1e-4, self._profile.wiener_balance())
        channels = cv2.split(frame_bgr.astype(np.float32))
        restored_channels = [
            self._wiener_channel(channel, kernel, balance) for channel in channels
        ]
        merged = cv2.merge(restored_channels)
        return np.clip(merged, 0, 255).astype(np.uint8)

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        preconditioned = self._wiener_deblur(frame_bgr)

        blur_strength = self._profile.motion_blur_strength()
        gaussian_sigma = 0.45 + 0.45 * blur_strength
        alpha = 0.55 + 0.5 * blur_strength
        softened = cv2.GaussianBlur(preconditioned, (0, 0), gaussian_sigma)
        high_boost = cv2.addWeighted(preconditioned, 1 + alpha, softened, -alpha, 0)

        ycrcb = cv2.cvtColor(high_boost, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        laplacian = cv2.Laplacian(y, cv2.CV_32F, ksize=3)
        gain = self._profile.detail_gain()
        enhanced_luma = y.astype(np.float32) + gain * laplacian
        enhanced_luma = np.clip(enhanced_luma, 0, 255).astype(np.uint8)
        merged = cv2.merge([enhanced_luma, cr, cb])
        restored = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

        if self._daytime_model is not None:
            gradient, lap = analyze_frame(restored)
            day_alpha, day_sigma = self._daytime_model.estimate_parameters(gradient, lap)
            refined = cv2.GaussianBlur(restored, (0, 0), day_sigma)
            restored = cv2.addWeighted(restored, 1 + day_alpha, refined, -day_alpha, 0)

        if self._profile.lighting is LightingCondition.DAYTIME:
            restored = cv2.bilateralFilter(restored, d=5, sigmaColor=28, sigmaSpace=10)

        return restored


class DayNightDetailTransfer(IFrameProcessor):
    """Inject daytime detail characteristics into dim captures."""

    def __init__(
        self,
        model: DaytimeDetailModel,
        *,
        residual_blend: float = 0.65,
        max_detail_gain: float = 1.9,
    ) -> None:
        self._model = model
        self._residual_blend = residual_blend
        self._max_detail_gain = max_detail_gain

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        gradient, laplacian = analyze_frame(frame_bgr)
        alpha, sigma = self._model.estimate_parameters(gradient, laplacian)
        softened = cv2.GaussianBlur(frame_bgr, (0, 0), sigma)
        unsharp = cv2.addWeighted(frame_bgr, 1 + alpha, softened, -alpha, 0)

        ycrcb = cv2.cvtColor(unsharp, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        lap = cv2.Laplacian(y, cv2.CV_32F, ksize=3)
        detail_gain = np.clip(alpha / max(self._model.base_alpha, 1e-3), 0.55, self._max_detail_gain)
        enhanced_luma = y.astype(np.float32) + self._residual_blend * detail_gain * lap
        enhanced_luma = np.clip(enhanced_luma, 0, 255).astype(np.uint8)
        merged = cv2.merge([enhanced_luma, cr, cb])
        result = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

        base_gradient, _ = analyze_frame(unsharp)
        enhanced_gradient, _ = analyze_frame(result)
        if enhanced_gradient < base_gradient:
            return unsharp
        return result


class FrameProcessingPipeline(IFrameProcessor):
    """Compose several frame processors."""

    def __init__(self, processors: List[IFrameProcessor]) -> None:
        self._processors = processors

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        out = frame_bgr
        for processor in self._processors:
            out = processor.process(out)
        return out


def build_preprocessor(
    enable_glare: bool,
    enable_denoise: bool,
    enable_deblur: bool,
    enable_motion_compensation: bool,
    enable_daytime_detail_transfer: bool = False,
    *,
    glare_v_thresh: float,
    glare_s_thresh: float,
    glare_knee_tau: float,
    glare_knee_strength: float,
    glare_dilate_px: int,
    denoise_h_luma: int,
    denoise_h_chroma: int,
    denoise_template: int,
    denoise_search: int,
    deblur_alpha: float,
    deblur_sigma: float,
    motion_profile: Optional[CaptureProfile] = None,
    daytime_model: Optional[DaytimeDetailModel] = None,
) -> Optional[FrameProcessingPipeline]:
    """Create a processing pipeline based on configuration flags."""

    processors: List[IFrameProcessor] = []
    if enable_glare:
        processors.append(
            GlareReducer(
                glare_v_thresh,
                glare_s_thresh,
                glare_knee_tau,
                glare_knee_strength,
                glare_dilate_px,
            )
        )
    if enable_denoise:
        processors.append(
            Denoiser(
                denoise_h_luma,
                denoise_h_chroma,
                denoise_template,
                denoise_search,
            )
        )
    if enable_deblur:
        processors.append(Deblurrer(deblur_alpha, deblur_sigma))
    if enable_motion_compensation and motion_profile is not None:
        processors.append(MotionAwareDeblurrer(motion_profile, daytime_model))
    if enable_daytime_detail_transfer and daytime_model is not None:
        processors.append(DayNightDetailTransfer(daytime_model))
    return FrameProcessingPipeline(processors) if processors else None
