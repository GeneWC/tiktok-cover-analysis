"""Unit tests for motion features.

Hand positions are passed in explicitly so the Hand Landmarker never runs -
these tests stay fast and need no model download.
"""

from __future__ import annotations

import numpy as np

from backend.features.motion_features import _FEATURE_KEYS, extract_motion_features

_SHAPE = (32, 24, 3)


def test_identical_frames_have_no_motion(make_sample):
    img = np.full(_SHAPE, 128, dtype=np.uint8)
    sample = make_sample([img, img], [0.0, 0.5])
    features = extract_motion_features(sample, hand_positions=[None, None])
    assert features["motion_energy_full"] == 0.0
    assert features["camera_stability_score"] == 1.0
    # No two frames had a hand -> hand motion unmeasurable.
    assert features["hand_detection_failed"] == 1
    assert features["hand_motion_energy_full"] is None


def test_changing_frames_have_motion(make_sample):
    black = np.zeros(_SHAPE, dtype=np.uint8)
    white = np.full(_SHAPE, 255, dtype=np.uint8)
    sample = make_sample([black, white, black], [0.0, 0.5, 1.0])
    features = extract_motion_features(sample, hand_positions=[None, None, None])
    assert features["motion_energy_full"] > 0.0
    assert features["camera_stability_score"] < 1.0


def test_hand_motion_displacement_measured(make_sample):
    img = np.full(_SHAPE, 100, dtype=np.uint8)
    sample = make_sample([img, img], [0.0, 0.5])
    # Hand centroid moves 0.1 in normalized x between the two frames.
    features = extract_motion_features(sample, hand_positions=[(0.5, 0.5), (0.6, 0.5)])
    assert features["hand_detection_failed"] == 0
    assert abs(features["hand_motion_energy_full"] - 0.1) < 1e-6


def test_too_few_frames_returns_none(make_sample):
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    features = extract_motion_features(make_sample([img], [0.0]), hand_positions=[None])
    assert all(features[key] is None for key in _FEATURE_KEYS)
