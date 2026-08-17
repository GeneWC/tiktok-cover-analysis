"""Unit tests for motion features.

Hand positions are passed in explicitly so the Hand Landmarker never runs -
these tests stay fast and need no model download.
"""

from __future__ import annotations

import cv2
import numpy as np

from backend.features.motion_features import _FEATURE_KEYS, extract_motion_features

_SHAPE = (32, 24, 3)


def _textured(height: int = 160, width: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[..., 0] = np.clip(50 + 90 * (xx / width) + 30 * np.sin(yy / 5.0), 0, 255).astype(
        np.uint8
    )
    img[..., 1] = np.clip(40 + 110 * (yy / height), 0, 255).astype(np.uint8)
    img[..., 2] = 70
    for _ in range(30):
        center = (int(rng.integers(4, width - 4)), int(rng.integers(4, height - 4)))
        color = tuple(int(v) for v in rng.integers(160, 255, size=3))
        cv2.circle(img, center, int(rng.integers(2, 6)), color, -1)
    return img


def _shift(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


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


def test_translated_background_is_camera_motion(make_sample):
    bg = _textured()
    sample = make_sample([bg, _shift(bg, 10, 0), _shift(bg, 20, 0)], [0.0, 0.4, 0.8])
    features = extract_motion_features(sample, hand_positions=[None, None, None])
    assert features["motion_energy_full"] > 0.0
    assert features["camera_stability_score"] < 1.0
    if features["camera_tracking_failed"] == 0:
        assert features["camera_translation_mean"] > 0.01


def test_moving_foreground_on_static_background(make_sample):
    bg = _textured()
    frames = []
    for x in (6, 22, 38):
        frame = bg.copy()
        frame[28:52, x : x + 12] = (0, 0, 255)
        frames.append(frame)
    sample = make_sample(frames, [0.0, 0.4, 0.8])
    features = extract_motion_features(sample, hand_positions=[None, None, None])
    assert features["motion_energy_full"] > 0.0
    if features["camera_tracking_failed"] == 0:
        assert features["performer_motion_energy_full"] is not None
        assert features["performer_motion_energy_full"] > features["camera_translation_mean"]
        assert features["camera_stability_score"] > 0.5


def test_increasing_blur_is_not_required_here(make_sample):
    # Sanity: static textured frames stay high-stability when nothing moves.
    bg = _textured()
    sample = make_sample([bg, bg, bg], [0.0, 0.4, 0.8])
    features = extract_motion_features(sample, hand_positions=[None, None, None])
    assert features["motion_energy_full"] == 0.0
    assert features["camera_stability_score"] >= 0.9
