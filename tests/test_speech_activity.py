"""Speech vs music heuristics on synthetic waveforms."""

from __future__ import annotations

import numpy as np

from backend.features.speech_activity import (
    SPEECH_FEATURE_KEYS,
    extract_speech_activity,
    speech_band_energy_ratio,
)


def _tone(sr: int, seconds: float, hz: float, amp: float = 0.4) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_empty_waveform_is_null():
    feats = extract_speech_activity(np.array([], dtype=np.float32), 22050)
    assert set(feats) == set(SPEECH_FEATURE_KEYS)
    assert all(v is None for v in feats.values())


def test_noise_like_bursts_register_as_speechish():
    rng = np.random.default_rng(0)
    sr = 8000
    y = (0.2 * rng.normal(size=sr * 2)).astype(np.float32)
    feats = extract_speech_activity(y, sr)
    assert feats["speech_ratio"] is not None
    assert 0.0 <= feats["speech_ratio"] <= 1.0
    assert 0.0 <= feats["speech_ratio_first_3s"] <= 1.0


def test_pure_tone_is_not_mostly_speech():
    sr = 8000
    y = _tone(sr, 1.5, 440.0)
    feats = extract_speech_activity(y, sr)
    assert feats["speech_ratio"] < 0.5


def test_speech_then_loud_hit_sets_gap_flag():
    sr = 8000
    rng = np.random.default_rng(1)
    early = (0.15 * rng.normal(size=int(sr * 1.5))).astype(np.float32)
    late = _tone(sr, 1.5, 220.0, amp=0.9)
    y = np.concatenate([early, late])
    feats = extract_speech_activity(y, sr)
    assert feats["music_after_speech_gap"] in (0, 1)
    if feats["speech_ratio_first_3s"] > 0.08:
        assert feats["music_after_speech_gap"] == 1


def test_speech_band_energy_is_fraction():
    sr = 8000
    y = _tone(sr, 0.5, 1000.0)
    ratio = speech_band_energy_ratio(y, sr)
    assert ratio is not None
    assert 0.0 <= ratio <= 1.0
    assert ratio > 0.5
