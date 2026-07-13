"""Unit tests for audio helpers and the no-audio handling (PRD 11.6)."""

from __future__ import annotations

import numpy as np

from backend.features.audio_features import (
    _FEATURE_KEYS,
    _empty_features,
    _segment_energy,
)


def test_no_audio_returns_nulls_and_failed_status():
    features = _empty_features("failed")
    assert features["audio_feature_extraction_status"] == "failed"
    assert all(features[key] is None for key in _FEATURE_KEYS)


def test_segment_energy_rms_of_ones_is_one():
    y = np.ones(100, dtype=np.float32)
    assert abs(_segment_energy(y, sr=10, seconds=None) - 1.0) < 1e-6


def test_segment_energy_respects_window_length():
    # First 1 second (10 samples) is silence, the rest is full scale.
    y = np.concatenate([np.zeros(10, dtype=np.float32), np.ones(90, dtype=np.float32)])
    assert _segment_energy(y, sr=10, seconds=1.0) == 0.0
    assert _segment_energy(y, sr=10, seconds=None) > 0.0


def test_segment_energy_empty_is_none():
    assert _segment_energy(np.array([], dtype=np.float32), sr=10, seconds=None) is None
