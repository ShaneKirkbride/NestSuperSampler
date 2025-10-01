from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

import numpy as np

from nest_super_sampler.pipeline import SuperSamplingPipeline


@dataclass
class FakeReader:
    frames_data: List[Tuple[int, float, np.ndarray]]

    def info(self) -> Dict[str, float]:
        return {"fps": 30.0}

    def frames(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        for item in self.frames_data:
            yield item


class KeepEvenSampler:
    def should_keep(self, idx: int, ts_sec: float, frame_bgr: np.ndarray) -> bool:
        return idx % 2 == 0


class MultiplySupersampler:
    def upscale(self, img_bgr: np.ndarray) -> np.ndarray:
        return img_bgr * 2


class RecordingSink:
    def __init__(self) -> None:
        self.records: List[Tuple[int, float, np.ndarray, np.ndarray, Dict[str, float]]] = []
        self.closed = False

    def write(
        self,
        frame_idx: int,
        ts_sec: float,
        original_bgr: np.ndarray,
        upscaled_bgr: np.ndarray,
        metadata: Dict[str, float],
    ) -> None:
        self.records.append((frame_idx, ts_sec, original_bgr, upscaled_bgr, metadata))

    def close(self) -> None:
        self.closed = True


class AddOnePreprocessor:
    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        return frame_bgr + 1


def test_pipeline_processes_and_records_frames() -> None:
    frames = [
        (0, 0.0, np.zeros((2, 2, 3), dtype=np.uint8)),
        (1, 0.1, np.ones((2, 2, 3), dtype=np.uint8)),
        (2, 0.2, np.full((2, 2, 3), 2, dtype=np.uint8)),
    ]
    reader = FakeReader(frames)
    sampler = KeepEvenSampler()
    supersampler = MultiplySupersampler()
    sink = RecordingSink()
    pipeline = SuperSamplingPipeline(
        reader,
        sampler,
        supersampler,
        sink,
        max_frames=None,
        preprocessor=AddOnePreprocessor(),
    )
    metrics = pipeline.run()
    assert len(sink.records) == 2
    assert sink.closed
    first_record = sink.records[0]
    assert first_record[0] == 0
    np.testing.assert_array_equal(first_record[3], (frames[0][2] + 1) * 2)
    assert metrics["frames_read"] == len(frames)
    assert metrics["frames_kept"] == 2
    assert metrics["fps_input"] == 30.0


def test_pipeline_respects_max_frames() -> None:
    frames = [
        (0, 0.0, np.zeros((2, 2, 3), dtype=np.uint8)),
        (2, 0.2, np.full((2, 2, 3), 2, dtype=np.uint8)),
        (4, 0.4, np.full((2, 2, 3), 4, dtype=np.uint8)),
    ]
    reader = FakeReader(frames)
    sampler = KeepEvenSampler()
    supersampler = MultiplySupersampler()
    sink = RecordingSink()
    pipeline = SuperSamplingPipeline(
        reader,
        sampler,
        supersampler,
        sink,
        max_frames=1,
        preprocessor=None,
    )
    metrics = pipeline.run()
    assert len(sink.records) == 1
    assert metrics["frames_kept"] == 1
