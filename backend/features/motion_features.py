"""Motion features: performer motion vs camera motion.

Existing serving keys stay stable so committed models keep their contract.
`camera_stability_score` is now estimated from background ORB tracking when
that is reliable, and falls back to the old pixel-difference proxy otherwise.

Extra keys (not in the 70-feature serving schema) describe camera transform,
residual/performer motion, and cheap temporal structure. Missing detections
stay None rather than a fake zero when zero would mean "no motion".
"""

from __future__ import annotations

import math

import cv2
import mediapipe as mp
import numpy as np

from backend.features.camera_motion import (
    PairwiseCameraMotion,
    camera_stability_from_pairs,
    estimate_pairwise_camera_motion,
)
from backend.features.frame_sampling import FrameSample, window_indices
from backend.features.mediapipe_models import create_hand_landmarker

# Weak-proxy normalization for camera stability when ORB tracking fails.
_CAMERA_MOTION_NORM = 25.0
_CUT_ENERGY_FLOOR = 18.0

_CORE_FEATURE_KEYS = (
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

_EXTRA_FEATURE_KEYS = (
    "camera_translation_mean",
    "camera_rotation_mean",
    "camera_scale_change_mean",
    "camera_tracking_failed",
    "performer_motion_energy_first_3s",
    "performer_motion_energy_full",
    "motion_subject_fraction",
    "shot_cut_count",
    "shot_cut_frequency",
    "average_shot_duration",
)

# Backward-compatible alias used by existing tests.
_FEATURE_KEYS = _CORE_FEATURE_KEYS


def _empty_features() -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {
        key: None for key in _CORE_FEATURE_KEYS + _EXTRA_FEATURE_KEYS
    }
    out["hand_detection_failed"] = 0
    out["camera_tracking_failed"] = 1
    out["shot_cut_count"] = 0
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


def _shot_cuts(pair_energy: list[float], pair_ts: list[float], duration: float) -> dict[str, float | int | None]:
    """Count large sudden pixel jumps as cuts. None when there is no pair data."""
    if not pair_energy:
        return {
            "shot_cut_count": 0,
            "shot_cut_frequency": None,
            "average_shot_duration": None,
        }
    median = float(np.median(pair_energy))
    mad = float(np.median(np.abs(np.asarray(pair_energy) - median)))
    threshold = max(_CUT_ENERGY_FLOOR, median + 4.0 * (mad if mad > 0 else float(np.std(pair_energy))))
    cut_times = [pair_ts[i] for i, energy in enumerate(pair_energy) if energy >= threshold]
    n_cuts = len(cut_times)
    n_shots = n_cuts + 1
    duration = max(float(duration), 1e-6)
    return {
        "shot_cut_count": n_cuts,
        "shot_cut_frequency": round(n_cuts / duration, 4),
        "average_shot_duration": round(duration / n_shots, 4),
    }


def extract_motion_features(
    sample: FrameSample,
    hand_positions: list[tuple[float, float] | None] | None = None,
    person_boxes: list[tuple[float, float, float, float] | None] | None = None,
) -> dict[str, float | int | None]:
    """Compute motion features from sampled frames."""
    if sample.is_empty or len(sample.frames) < 2:
        return _empty_features()

    grays = [cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY) for f in sample.frames]

    pair_ts: list[float] = []
    pair_energy: list[float] = []
    camera_pairs: list[PairwiseCameraMotion] = []
    residual_energy: list[float] = []
    residual_ts: list[float] = []
    for i in range(1, len(grays)):
        energy = float(cv2.absdiff(grays[i], grays[i - 1]).mean())
        pair_energy.append(energy)
        pair_ts.append(sample.frames[i].timestamp)
        box = None
        if person_boxes is not None and i < len(person_boxes):
            box = person_boxes[i] or person_boxes[i - 1]
        tracked = estimate_pairwise_camera_motion(grays[i - 1], grays[i], person_box=box)
        if tracked is not None:
            camera_pairs.append(tracked)
            residual_energy.append(tracked.residual_energy)
            residual_ts.append(sample.frames[i].timestamp)

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

    if hand_positions is None:
        hand_positions = _hand_positions(sample)

    hand_pair_ts: list[float] = []
    hand_pair_disp: list[float] = []
    for i in range(1, len(hand_positions)):
        prev, curr = hand_positions[i - 1], hand_positions[i]
        if prev is not None and curr is not None:
            hand_pair_disp.append(math.dist(prev, curr))
            hand_pair_ts.append(sample.frames[i].timestamp)

    if not hand_pair_disp:
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

    if camera_pairs:
        features["camera_tracking_failed"] = 0
        features["camera_translation_mean"] = round(
            float(np.mean([p.translation for p in camera_pairs])), 6
        )
        features["camera_rotation_mean"] = round(
            float(np.mean([p.rotation_deg for p in camera_pairs])), 4
        )
        features["camera_scale_change_mean"] = round(
            float(np.mean([p.scale_change for p in camera_pairs])), 6
        )
        features["camera_stability_score"] = round(camera_stability_from_pairs(camera_pairs), 4)
        performer_full = _window_mean(residual_energy, residual_ts, "full")
        performer_first_3s = _window_mean(residual_energy, residual_ts, "first_3s")
        features["performer_motion_energy_full"] = (
            round(performer_full, 4) if performer_full is not None else None
        )
        features["performer_motion_energy_first_3s"] = (
            round(performer_first_3s, 4) if performer_first_3s is not None else None
        )
        camera_component = float(np.mean([p.translation for p in camera_pairs]))
        if performer_full is not None and (performer_full + camera_component) > 0:
            features["motion_subject_fraction"] = round(
                performer_full / (performer_full + camera_component * _CAMERA_MOTION_NORM),
                4,
            )
        else:
            features["motion_subject_fraction"] = None
    else:
        features["camera_tracking_failed"] = 1
        features["camera_translation_mean"] = None
        features["camera_rotation_mean"] = None
        features["camera_scale_change_mean"] = None
        features["performer_motion_energy_full"] = features["hand_motion_energy_full"]
        features["performer_motion_energy_first_3s"] = features["hand_motion_energy_first_3s"]
        features["motion_subject_fraction"] = None
        if energy_full is not None:
            normalized_motion = min(1.0, energy_full / _CAMERA_MOTION_NORM)
            features["camera_stability_score"] = round(1.0 - normalized_motion, 4)
        else:
            features["camera_stability_score"] = None

    features.update(_shot_cuts(pair_energy, pair_ts, sample.duration_seconds or 0.0))
    return features
