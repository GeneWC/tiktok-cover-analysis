"""Unit tests for audio helpers and the no-audio handling (PRD 11.6 / D-007)."""

from __future__ import annotations

import librosa
import numpy as np

from backend.features.audio_features import (
    AUDIO_PRODUCTION_FEATURE_KEYS,
    AUDIO_STRUCTURE_FEATURE_KEYS,
    _FEATURE_KEYS,
    _empty_features,
    _production_features,
    _segment_energy,
    _structure_features,
)


def test_no_audio_returns_nulls_and_failed_status():
    features = _empty_features("failed")
    assert features["audio_feature_extraction_status"] == "failed"
    assert all(features[key] is None for key in _FEATURE_KEYS)


def test_production_keys_are_in_feature_schema():
    for key in AUDIO_PRODUCTION_FEATURE_KEYS:
        assert key in _FEATURE_KEYS
    assert "speech_ratio" in _FEATURE_KEYS
    assert "music_after_speech_gap" in _FEATURE_KEYS


def test_structure_keys_are_in_feature_schema():
    for key in AUDIO_STRUCTURE_FEATURE_KEYS:
        assert key in _FEATURE_KEYS


def test_production_features_on_tone():
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    feats = _production_features(y, sr)
    for key in AUDIO_PRODUCTION_FEATURE_KEYS:
        assert key in feats
        assert feats[key] is None or np.isfinite(feats[key])
    if feats["audio_harmonic_ratio"] is not None:
        assert feats["audio_harmonic_ratio"] > feats.get("audio_percussive_ratio", 0)


def test_structure_features_on_tone():
    sr = 22050
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    y = (0.5 * (1 + np.sin(2 * np.pi * 3 * t)) * np.sin(2 * np.pi * 440 * t)).astype(
        np.float32
    )
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    feats = _structure_features(y, sr, onset_env)
    for key in AUDIO_STRUCTURE_FEATURE_KEYS:
        assert key in feats
        assert feats[key] is None or np.isfinite(feats[key])


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
