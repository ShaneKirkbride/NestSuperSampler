"""Image sink implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

from .interfaces import IImageSink


class DiskImageSink(IImageSink):
    """Write images and metadata to disk."""

    def __init__(
        self,
        out_dir: Path,
        image_format: str = "png",
        jpg_quality: int = 95,
        write_full_size_only: bool = True,
    ) -> None:
        self._out = out_dir
        self._out.mkdir(parents=True, exist_ok=True)
        self._fmt = image_format.lower()
        self._jpg_quality = jpg_quality
        self._write_full_only = write_full_size_only
        self._rows: List[Dict[str, float]] = []

    def _imwrite(self, path: Path, bgr: np.ndarray) -> None:
        if self._fmt in ("jpg", "jpeg"):
            cv2.imwrite(
                str(path),
                bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), self._jpg_quality],
            )
        else:
            cv2.imwrite(str(path), bgr)

    def write(
        self,
        frame_idx: int,
        ts_sec: float,
        original_bgr: np.ndarray,
        upscaled_bgr: np.ndarray,
        metadata: Dict[str, float],
    ) -> None:
        stem = f"frame_{frame_idx:08d}_t{ts_sec:010.3f}"
        up_path = self._out / f"{stem}_SR.{self._fmt}"
        self._imwrite(up_path, upscaled_bgr)
        if not self._write_full_only:
            orig_path = self._out / f"{stem}_ORIG.{self._fmt}"
            self._imwrite(orig_path, original_bgr)
        row: Dict[str, float] = {
            "frame_idx": float(frame_idx),
            "timestamp_s": float(ts_sec),
            "orig_w": float(original_bgr.shape[1]),
            "orig_h": float(original_bgr.shape[0]),
            "sr_w": float(upscaled_bgr.shape[1]),
            "sr_h": float(upscaled_bgr.shape[0]),
        }
        row.update(metadata)
        self._rows.append(row)

    def close(self) -> None:
        if not self._rows:
            return
        df = pd.DataFrame(self._rows)
        df.sort_values(by=["frame_idx"], inplace=True)
        df.to_csv(self._out / "frames_metadata.csv", index=False)
