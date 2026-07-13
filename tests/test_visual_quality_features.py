"""Unit tests for visual quality math, using synthetic frames (no models)."""

from __future__ import annotations

import numpy as np

from backend.features.frame_sampling import FrameSample
from backend.features.visual_quality_features import (
    _FEATURE_KEYS,
    extract_visual_quality_features,
)

_SHAPE = (32, 24, 3)
_TIMESTAMPS = [0.0, 1.0, 2.0, 3.0]


def _solid(value: int, n: int = 4) -> list[np.ndarray]:
    return [np.full(_SHAPE, value, dtype=np.uint8) for _ in range(n)]


def test_all_feature_keys_present(make_sample):
    features = extract_visual_quality_features(make_sample(_solid(0), _TIMESTAMPS))
    assert set(_FEATURE_KEYS) <= set(features)


def test_black_frames_have_zero_metrics(make_sample):
    features = extract_visual_quality_features(make_sample(_solid(0), _TIMESTAMPS))
    assert features["brightness_mean_full"] == 0.0
    assert features["contrast_full"] == 0.0
    assert features["sharpness_full"] == 0.0
    assert features["blur_full"] == 1.0  # 1/(1+0)
    assert features["colorfulness_full"] == 0.0


def test_uniform_frame_brightness_matches_value(make_sample):
    features = extract_visual_quality_features(make_sample(_solid(200), _TIMESTAMPS))
    assert abs(features["brightness_mean_full"] - 200.0) < 1e-6
    assert features["contrast_full"] == 0.0  # flat image -> no contrast


def test_noise_is_sharper_than_flat(make_sample):
    rng = np.random.default_rng(0)
    noise = [rng.integers(0, 256, _SHAPE, dtype=np.uint8) for _ in range(4)]
    features = extract_visual_quality_features(make_sample(noise, _TIMESTAMPS))
    assert features["sharpness_full"] > 100.0
    assert features["blur_full"] < 0.05


def test_empty_sample_returns_all_none():
    features = extract_visual_quality_features(FrameSample(0, 0, 0.0, 0.0, 0, 0))
    assert all(features[key] is None for key in _FEATURE_KEYS)
