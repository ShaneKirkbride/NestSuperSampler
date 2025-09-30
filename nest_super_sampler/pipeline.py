"""Pipeline orchestration."""

from __future__ import annotations

import time
from typing import Dict, Optional

from .interfaces import (
    IFrameProcessor,
    IFrameSampler,
    IImageSink,
    ISuperSampler,
    IVideoReader,
)


class SuperSamplingPipeline:
    """Coordinate reading, processing, and writing frames."""

    def __init__(
        self,
        reader: IVideoReader,
        sampler: IFrameSampler,
        supersampler: ISuperSampler,
        sink: IImageSink,
        max_frames: Optional[int] = None,
        preprocessor: Optional[IFrameProcessor] = None,
    ) -> None:
        self._reader = reader
        self._sampler = sampler
        self._super = supersampler
        self._sink = sink
        self._max_frames = max_frames
        self._pre = preprocessor

    def run(self) -> Dict[str, float]:
        info = self._reader.info()
        processed = 0
        kept = 0
        start = time.time()
        for idx, ts, frame in self._reader.frames():
            processed += 1
            if self._max_frames is not None and kept >= self._max_frames:
                break
            if self._sampler.should_keep(idx, ts, frame):
                work = self._pre.process(frame) if self._pre is not None else frame
                upscaled = self._super.upscale(work)
                self._sink.write(idx, ts, frame, upscaled, {"fps": info.get("fps", 0.0)})
                kept += 1
        self._sink.close()
        elapsed = time.time() - start
        return {
            "frames_read": float(processed),
            "frames_kept": float(kept),
            "elapsed_s": float(elapsed),
            "fps_input": float(info.get("fps", 0.0)),
        }
