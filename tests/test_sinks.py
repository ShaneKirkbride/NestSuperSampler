from pathlib import Path

import numpy as np
import pandas as pd

from nest_super_sampler.sinks import DiskImageSink


def test_disk_image_sink_writes_images_and_metadata(tmp_path, monkeypatch) -> None:
    writes = []

    def fake_imwrite(path: str, image: np.ndarray, params=None):  # type: ignore[unused-argument]
        writes.append(Path(path))
        return True

    monkeypatch.setattr("cv2.imwrite", fake_imwrite)
    sink = DiskImageSink(tmp_path, image_format="png", write_full_size_only=False)
    original = np.zeros((2, 2, 3), dtype=np.uint8)
    upscaled = np.ones((4, 4, 3), dtype=np.uint8)
    sink.write(1, 0.033, original, upscaled, {"fps": 30.0})
    sink.close()

    assert len(writes) == 2
    metadata = pd.read_csv(tmp_path / "frames_metadata.csv")
    assert metadata.loc[0, "frame_idx"] == 1
    assert metadata.loc[0, "sr_w"] == 4


def test_disk_image_sink_no_close_when_no_rows(tmp_path, monkeypatch) -> None:
    close_called = False

    def fake_to_csv(self, path, index=False):  # type: ignore[unused-argument]
        nonlocal close_called
        close_called = True

    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv, raising=False)
    sink = DiskImageSink(tmp_path)
    sink.close()
    assert not close_called
