from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from nest_super_sampler.detail_models import (
    DaytimeDetailModel,
    _collect_reference_frames,
    analyze_frame,
    fit_daytime_detail_model,
    fit_daytime_model_from_path,
)
from nest_super_sampler.capture_profiles import derive_capture_profile
from nest_super_sampler.config import LightingCondition, PipelineConfig
from nest_super_sampler.preprocessors import DayNightDetailTransfer


def _write_reference_images(root: Path, count: int = 3) -> None:
    for idx in range(count):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.line(image, (0, idx + 4), (31, idx + 4), (255, 255, 255), 1)
        cv2.imwrite(str(root / f"frame_{idx}.png"), image)


def _write_reference_video(path: Path, frame_count: int = 12) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 24.0, (32, 32))
    if not writer.isOpened():  # pragma: no cover - guard in case codec is unavailable
        raise RuntimeError("Failed to create reference video for testing")
    for idx in range(frame_count):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.rectangle(frame, (4 + idx % 4, 4), (28, 28), (255, 255, 255), 1)
        writer.write(frame)
    writer.release()


def test_collect_reference_frames_from_directory(tmp_path) -> None:
    _write_reference_images(tmp_path)
    frames = _collect_reference_frames(tmp_path, max_frames=2)
    assert len(frames) == 2


def test_collect_reference_frames_from_video(tmp_path) -> None:
    video_path = tmp_path / "daytime.avi"
    _write_reference_video(video_path)
    frames = _collect_reference_frames(video_path, max_frames=5)
    assert len(frames) == 5


def test_fit_daytime_model_from_path_for_directory(tmp_path) -> None:
    _write_reference_images(tmp_path)
    model = fit_daytime_model_from_path(tmp_path)
    assert isinstance(model, DaytimeDetailModel)


def test_fit_daytime_model_from_path_for_video(tmp_path) -> None:
    video_path = tmp_path / "daytime.avi"
    _write_reference_video(video_path, frame_count=6)
    model = fit_daytime_model_from_path(video_path, max_frames=4)
    assert isinstance(model, DaytimeDetailModel)


def test_daynight_detail_transfer_falls_back_when_detail_reduces() -> None:
    reference = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.rectangle(reference, (6, 6), (26, 26), (255, 255, 255), 2)
    model = fit_daytime_detail_model([reference])
    transfer = DayNightDetailTransfer(model, residual_blend=-0.5)
    night_frame = cv2.GaussianBlur(reference, (0, 0), 1.2)

    result = transfer.process(night_frame)
    gradient, laplacian = analyze_frame(night_frame)
    alpha, sigma = model.estimate_parameters(gradient, laplacian)
    softened = cv2.GaussianBlur(night_frame, (0, 0), sigma)
    unsharp = cv2.addWeighted(night_frame, 1 + alpha, softened, -alpha, 0)

    assert np.array_equal(result, unsharp)


def test_derive_capture_profile_auto_lighting(tmp_path) -> None:
    config = PipelineConfig(
        input_video=tmp_path / "dummy.mp4",
        output_dir=tmp_path,
        enable_motion_compensation=True,
        lighting=LightingCondition.AUTO,
    )
    profile = derive_capture_profile(config, {"fps": 12.0})
    assert profile.lighting is LightingCondition.NIGHT
