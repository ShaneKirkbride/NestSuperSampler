"""Frame pre-processing operators."""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .interfaces import IFrameProcessor


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
    return FrameProcessingPipeline(processors) if processors else None
