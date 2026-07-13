"""Motion features (PRD 11.5).

Emphasizes *performance* motion, not editing. We deliberately do NOT compute
cuts-per-second, shot length, or scene transitions (PRD 11.5).

- Overall motion energy: mean absolute pixel difference between consecutive
  sampled grayscale frames, averaged per time window.
- motion_consistency = 1 / (1 + std(per-pair motion energy over the full video)).
- Hand motion energy: normalized displacement of the hand-landmark centroid
  between consecutive frames (MediaPipe Hand Landmarker). Missing hands ->
  null + hand_detection_failed = 1 (PRD 11.5 / 12.5).
- camera_stability_score = 1 - normalized total motion. Without background
  isolation this uses total frame motion as a documented weak proxy (PRD 11.5).

`extract_motion_features` accepts an optional precomputed `hand_positions`
sequence (one normalized (x, y) centroid or None per sampled frame) so the
aggregator can share hand detection with the framing step instead of running it
twice.
"""

from __future__ import annotations

import math

import cv2
import mediapipe as mp
import numpy as np

from backend.features.frame_sampling import FrameSample, window_indices
from backend.features.mediapipe_models import create_hand_landmarker

# Weak-proxy normalization for camera stability: maps mean abs grayscale diff
# (0-255 scale) onto ~0-1 "background motion". Documented approximation.
_CAMERA_MOTION_NORM = 25.0

_FEATURE_KEYS = (
    "motion_energy_first_1s",
    "motion_energy_first_3s",
    "motion_energy_first_6s",
    "motion_energy_full",
    "motion_energy_ratio_first_3s_to_full",
    "motion_consistency",
    "hand_motion_energy_first_3s",
    "hand_motion_energy_full",
    "hand_motion_consistency",
    "camera_stability_score",
)


def _empty_features() -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {key: None for key in _FEATURE_KEYS}
    out["hand_detection_failed"] = 0
    return out


def _hand_positions(sample: FrameSample) -> list[tuple[float, float] | None]:
    """Run the Hand Landmarker and return a normalized centroid (or None) per frame."""
    positions: list[tuple[float, float] | None] = []
    landmarker = create_hand_landmarker()
    try:
        for frame in sample.frames:
            rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                cx = float(np.mean([lm.x for lm in landmarks]))
                cy = float(np.mean([lm.y for lm in landmarks]))
                positions.append((cx, cy))
            else:
                positions.append(None)
    finally:
        landmarker.close()
    return positions


def _window_mean(values: list[float], pair_ts: list[float], window: str) -> float | None:
    """Mean of per-pair values whose (later-frame) timestamp falls in a window."""
    idx = window_indices(pair_ts, window)
    selected = [values[i] for i in idx if i < len(values)]
    return float(np.mean(selected)) if selected else None


def extract_motion_features(
    sample: FrameSample,
    hand_positions: list[tuple[float, float] | None] | None = None,
) -> dict[str, float | int | None]:
    """Compute motion features (PRD 11.5) from sampled frames."""
    if sample.is_empty or len(sample.frames) < 2:
        return _empty_features()

    grays = [cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY) for f in sample.frames]

    # Per consecutive-pair overall motion energy; tag each with the later frame's ts.
    pair_ts: list[float] = []
    pair_energy: list[float] = []
    for i in range(1, len(grays)):
        pair_energy.append(float(cv2.absdiff(grays[i], grays[i - 1]).mean()))
        pair_ts.append(sample.frames[i].timestamp)

    features: dict[str, float | int | None] = {}
    for window in ("first_1s", "first_3s", "first_6s", "full"):
        value = _window_mean(pair_energy, pair_ts, window)
        features[f"motion_energy_{window}"] = round(value, 4) if value is not None else None

    energy_full = features["motion_energy_full"]
    energy_first_3s = features["motion_energy_first_3s"]
    if energy_full and energy_full > 0 and energy_first_3s is not None:
        features["motion_energy_ratio_first_3s_to_full"] = round(energy_first_3s / energy_full, 4)
    else:
        features["motion_energy_ratio_first_3s_to_full"] = None

    features["motion_consistency"] = (
        round(1.0 / (1.0 + float(np.std(pair_energy))), 4) if pair_energy else None
    )

    # --- hand motion ---
    if hand_positions is None:
        hand_positions = _hand_positions(sample)

    hand_pair_ts: list[float] = []
    hand_pair_disp: list[float] = []
    for i in range(1, len(hand_positions)):
        prev, curr = hand_positions[i - 1], hand_positions[i]
        if prev is not None and curr is not None:
            disp = math.dist(prev, curr)  # normalized (coords are 0-1)
            hand_pair_disp.append(disp)
            hand_pair_ts.append(sample.frames[i].timestamp)

    if not hand_pair_disp:
        # No two consecutive frames both had a hand -> can't measure hand motion.
        features["hand_motion_energy_first_3s"] = None
        features["hand_motion_energy_full"] = None
        features["hand_motion_consistency"] = None
        features["hand_detection_failed"] = 1
    else:
        hm_first_3s = _window_mean(hand_pair_disp, hand_pair_ts, "first_3s")
        hm_full = _window_mean(hand_pair_disp, hand_pair_ts, "full")
        features["hand_motion_energy_first_3s"] = (
            round(hm_first_3s, 6) if hm_first_3s is not None else None
        )
        features["hand_motion_energy_full"] = round(hm_full, 6) if hm_full is not None else None
        features["hand_motion_consistency"] = round(
            1.0 / (1.0 + float(np.std(hand_pair_disp))), 4
        )
        features["hand_detection_failed"] = 0

    # --- camera stability (weak proxy) ---
    if energy_full is not None:
        normalized_motion = min(1.0, energy_full / _CAMERA_MOTION_NORM)
        features["camera_stability_score"] = round(1.0 - normalized_motion, 4)
    else:
        features["camera_stability_score"] = None

    return features
