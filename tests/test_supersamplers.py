import sys
from types import ModuleType

import numpy as np
import pytest

from nest_super_sampler.config import PipelineConfig, SuperResAlgo
from nest_super_sampler.supersamplers import (
    BicubicSuperSampler,
    DNNSuperSampler,
    build_supersampler,
)


class DummySR:
    def __init__(self) -> None:
        self.model = None
        self.model_path = None

    def setModel(self, model_name: str, scale: int) -> None:  # noqa: N802 - OpenCV naming
        self.model = (model_name, scale)

    def readModel(self, model_path: str) -> None:  # noqa: N802 - OpenCV naming
        self.model_path = model_path

    def upsample(self, img: np.ndarray) -> np.ndarray:
        return img + 1


def install_fake_dnn_module(monkeypatch: pytest.MonkeyPatch) -> DummySR:
    fake_module = ModuleType("cv2.dnn_superres")
    instance: DummySR = DummySR()

    def factory() -> DummySR:
        return instance

    fake_module.DnnSuperResImpl_create = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cv2.dnn_superres", fake_module)
    return instance


def test_bicubic_supersampler_upscale_changes_shape() -> None:
    sampler = BicubicSuperSampler(scale=2)
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    result = sampler.upscale(image)
    assert result.shape == (8, 10, 3)


def test_bicubic_supersampler_invalid_scale() -> None:
    with pytest.raises(ValueError):
        BicubicSuperSampler(scale=0)


def test_dnnsupersampler_loads_model_and_upscales(tmp_path, monkeypatch) -> None:
    instance = install_fake_dnn_module(monkeypatch)
    model_path = tmp_path / "EDSR_x2.pb"
    model_path.write_bytes(b"fake")
    sampler = DNNSuperSampler(SuperResAlgo.EDSR, 2, tmp_path)
    image = np.ones((2, 2, 3), dtype=np.uint8)
    result = sampler.upscale(image)
    assert np.array_equal(result, image + 1)
    assert instance.model == ("edsr", 2)
    assert instance.model_path == str(model_path.resolve())


def test_dnnsupersampler_missing_model(tmp_path, monkeypatch) -> None:
    install_fake_dnn_module(monkeypatch)
    with pytest.raises(FileNotFoundError):
        DNNSuperSampler(SuperResAlgo.EDSR, 2, tmp_path)


def test_build_supersampler_returns_bicubic_when_requested(tmp_path) -> None:
    cfg = PipelineConfig(
        input_video=tmp_path / "in.mp4",
        output_dir=tmp_path / "out",
        algo=SuperResAlgo.BICUBIC,
        upscale_factor=3,
    )
    sampler = build_supersampler(cfg)
    assert isinstance(sampler, BicubicSuperSampler)


def test_build_supersampler_falls_back_when_missing_model(tmp_path) -> None:
    cfg = PipelineConfig(
        input_video=tmp_path / "in.mp4",
        output_dir=tmp_path / "out",
        algo=SuperResAlgo.EDSR,
        upscale_factor=2,
        model_dir=tmp_path,
        fail_on_missing_superres=False,
    )
    sampler = build_supersampler(cfg)
    assert isinstance(sampler, BicubicSuperSampler)
