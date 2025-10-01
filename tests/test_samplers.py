import numpy as np
import pytest

from nest_super_sampler.samplers import CompositeSampler, NthFrameSampler, SceneChangeSampler


def test_nth_frame_sampler_keeps_every_nth_frame() -> None:
    sampler = NthFrameSampler(3)
    assert sampler.should_keep(0, 0.0, np.zeros((1, 1, 3), dtype=np.uint8))
    assert not sampler.should_keep(1, 0.0, np.zeros((1, 1, 3), dtype=np.uint8))
    assert sampler.should_keep(3, 0.0, np.zeros((1, 1, 3), dtype=np.uint8))


def test_nth_frame_sampler_rejects_invalid_n() -> None:
    with pytest.raises(ValueError):
        NthFrameSampler(0)


def test_scene_change_sampler_detects_change() -> None:
    sampler = SceneChangeSampler(hist_threshold=0.05, min_gap_frames=1)
    frame1 = np.zeros((8, 8, 3), dtype=np.uint8)
    frame2 = np.zeros((8, 8, 3), dtype=np.uint8)
    frame2[:, :4] = [255, 0, 0]
    frame2[:, 4:] = [0, 255, 0]
    assert sampler.should_keep(0, 0.0, frame1)
    assert sampler.should_keep(1, 0.0, frame2)
    assert not sampler.should_keep(2, 0.0, frame2)


def test_composite_sampler_requires_samplers() -> None:
    with pytest.raises(ValueError):
        CompositeSampler([])


def test_composite_sampler_combines_results() -> None:
    sampler = CompositeSampler([NthFrameSampler(2), NthFrameSampler(3)])
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    assert sampler.should_keep(2, 0.0, frame)
    assert sampler.should_keep(3, 0.0, frame)
    assert not sampler.should_keep(5, 0.0, frame)
