"""Unit tests for the frame sampler's pure helpers (no video decode needed)."""

from __future__ import annotations

import numpy as np

from backend.features.frame_sampling import (
    FrameSample,
    SampledFrame,
    _target_dimensions,
    window_indices,
)


def test_target_dimensions_downscales_and_preserves_aspect():
    # 1080x1920 capped to a 256 longest side -> 144x256 (9:16 preserved).
    assert _target_dimensions(1080, 1920, 256) == (144, 256)


def test_target_dimensions_never_upscales():
    assert _target_dimensions(100, 80, 256) == (100, 80)


def test_window_indices_select_by_time():
    timestamps = [0.0, 0.5, 1.5, 5.0, 7.0]
    assert window_indices(timestamps, "first_1s") == [0, 1]
    assert window_indices(timestamps, "first_3s") == [0, 1, 2]
    assert window_indices(timestamps, "first_6s") == [0, 1, 2, 3]
    assert window_indices(timestamps, "full") == [0, 1, 2, 3, 4]


def test_window_indices_short_video_fallback():
    # No frame lands before 1s, but we still keep at least the first frame.
    assert window_indices([4.0, 5.0], "first_1s") == [0]


def test_window_indices_empty():
    assert window_indices([], "full") == []


def test_framesample_window_fallback_and_full():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    fs = FrameSample(
        4, 4, 3.0, 5.0, 4, 4, [SampledFrame(4.0, img), SampledFrame(5.0, img)]
    )
    assert len(fs.window("first_1s")) == 1  # fallback to first frame
    assert len(fs.window("full")) == 2
    assert fs.is_empty is False


def test_framesample_is_empty():
    assert FrameSample(0, 0, 0.0, 0.0, 0, 0).is_empty is True
