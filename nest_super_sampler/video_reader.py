"""Video reader implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Tuple

import cv2
import numpy as np

from .interfaces import IVideoReader


class OpenCVVideoReader(IVideoReader):
    """Video reader based on :mod:`cv2`."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def info(self) -> Dict[str, float]:
        duration = (
            (self._frame_count / self._fps)
            if (self._fps > 0 and self._frame_count > 0)
            else 0.0
        )
        return {
            "width": float(self._width),
            "height": float(self._height),
            "fps": float(self._fps),
            "frame_count": float(self._frame_count),
            "duration_s": duration,
        }

    def frames(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            ts_sec = idx / self._fps if self._fps > 0 else 0.0
            yield idx, ts_sec, frame
            idx += 1
        self._cap.release()
