"""Frame sampling strategies."""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .interfaces import IFrameSampler


class NthFrameSampler(IFrameSampler):
    """Keep every N-th frame."""

    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError("n must be >= 1")
        self._n = n

    def should_keep(self, idx: int, ts_sec: float, frame_bgr: np.ndarray) -> bool:
        return (idx % self._n) == 0


class SceneChangeSampler(IFrameSampler):
    """Keep frames when the scene changes based on HSV histogram difference."""

    def __init__(self, hist_threshold: float, min_gap_frames: int) -> None:
        self._thresh = float(hist_threshold)
        self._min_gap = int(min_gap_frames)
        self._last_kept_idx: Optional[int] = None
        self._last_hist: Optional[np.ndarray] = None

    @staticmethod
    def _norm_hist(img_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def should_keep(self, idx: int, ts_sec: float, frame_bgr: np.ndarray) -> bool:
        hist = self._norm_hist(frame_bgr)
        if self._last_hist is None:
            self._last_hist = hist
            self._last_kept_idx = idx
            return True
        if self._last_kept_idx is not None and (idx - self._last_kept_idx) < self._min_gap:
            return False
        score = cv2.compareHist(self._last_hist, hist, cv2.HISTCMP_CORREL)
        change = 1.0 - float(score)
        if change >= self._thresh:
            self._last_hist = hist
            self._last_kept_idx = idx
            return True
        return False


class CompositeSampler(IFrameSampler):
    """Combine multiple samplers using logical OR."""

    def __init__(self, samplers: List[IFrameSampler]) -> None:
        if not samplers:
            raise ValueError("CompositeSampler requires at least one sampler")
        self._samplers = samplers

    def should_keep(self, idx: int, ts_sec: float, frame_bgr: np.ndarray) -> bool:
        return any(s.should_keep(idx, ts_sec, frame_bgr) for s in self._samplers)
