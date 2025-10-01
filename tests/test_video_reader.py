from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from nest_super_sampler.video_reader import OpenCVVideoReader


class FakeCapture:
    def __init__(self, frames: List[np.ndarray], *, opened: bool = True) -> None:
        self.frames = frames
        self.opened = opened
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mimics OpenCV API
        return self.opened

    def get(self, prop_id: int) -> float:  # noqa: N802 - mimics OpenCV API
        if prop_id == 5:  # cv2.CAP_PROP_FPS
            return 24.0
        if prop_id == 7:  # cv2.CAP_PROP_FRAME_COUNT
            return len(self.frames)
        if prop_id == 3:  # width
            return self.frames[0].shape[1]
        if prop_id == 4:  # height
            return self.frames[0].shape[0]
        return 0.0

    def read(self) -> Tuple[bool, np.ndarray]:  # noqa: N802
        if self.index >= len(self.frames):
            return False, np.empty((0, 0, 3), dtype=np.uint8)
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


def test_video_reader_info_and_frames(tmp_path, monkeypatch) -> None:
    path = tmp_path / "video.mp4"
    path.write_bytes(b"fake")
    frames = [np.zeros((2, 3, 3), dtype=np.uint8), np.ones((2, 3, 3), dtype=np.uint8)]

    def fake_capture_factory(_):
        return FakeCapture(frames)

    monkeypatch.setattr("cv2.VideoCapture", fake_capture_factory)
    reader = OpenCVVideoReader(path)
    info = reader.info()
    assert info["fps"] == 24.0
    assert info["frame_count"] == len(frames)
    collected = list(reader.frames())
    assert len(collected) == len(frames)
    assert collected[0][0] == 0
    np.testing.assert_array_equal(collected[0][2], frames[0])


def test_video_reader_raises_when_capture_fails(tmp_path, monkeypatch) -> None:
    path = tmp_path / "video.mp4"
    path.write_bytes(b"fake")

    def fake_capture_factory(_):
        return FakeCapture([], opened=False)

    monkeypatch.setattr("cv2.VideoCapture", fake_capture_factory)
    with pytest.raises(RuntimeError):
        OpenCVVideoReader(path)


def test_video_reader_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        OpenCVVideoReader(tmp_path / "missing.mp4")
