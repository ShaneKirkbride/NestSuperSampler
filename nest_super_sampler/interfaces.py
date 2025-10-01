"""Core protocol interfaces used across the pipeline."""

from __future__ import annotations

from typing import Dict, Iterator, Protocol, Tuple

import numpy as np


class IVideoReader(Protocol):
    """Iterates through a video stream."""

    def info(self) -> Dict[str, float]:
        """Return metadata about the video such as width, height and FPS."""

    def frames(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        """Yield tuples of (frame index, timestamp seconds, frame BGR array)."""


class IFrameSampler(Protocol):
    """Decides whether a frame should be kept for further processing."""

    def should_keep(self, idx: int, ts_sec: float, frame_bgr: np.ndarray) -> bool:
        """Return True when the frame should be retained."""


class ISuperSampler(Protocol):
    """Performs super-resolution on frames."""

    def upscale(self, img_bgr: np.ndarray) -> np.ndarray:
        """Return an upscaled BGR image."""


class IImageSink(Protocol):
    """Persists frames and metadata to disk or another destination."""

    def write(
        self,
        frame_idx: int,
        ts_sec: float,
        original_bgr: np.ndarray,
        upscaled_bgr: np.ndarray,
        metadata: Dict[str, float],
    ) -> None:
        """Persist the provided frame data."""

    def close(self) -> None:
        """Finalize the sink, flushing any buffered data."""


class IFrameProcessor(Protocol):
    """Transforms frames before super-resolution (e.g. denoise, deblur)."""

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a processed frame."""


class IPostProcessor(Protocol):
    """Transforms frames after super-resolution."""

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a processed frame."""
