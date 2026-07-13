"""Framing & subject-visibility features (PRD 11.4).

Uses MediaPipe's off-the-shelf Tasks detectors (Pose Landmarker, Face Detector,
Hand Landmarker) to measure how visible and well-framed the performer is:
  - person / face / hand / upper-body visibility ratios (full + first_3s)
  - subject centering and size, face size

Key practices:
- MediaPipe expects RGB; our sampled frames are BGR (OpenCV), so we convert and
  wrap each frame in an `mp.Image`.
- Detectors run in IMAGE running mode (each sampled frame is independent; the
  samples are not consecutive video frames, so temporal tracking doesn't apply).
- Missing vs failed is distinct (PRD 11.4 / 12.5): if no person is detected,
  centering is null and `subject_centering_missing = 1`; if a detector errors
  technically, that's flagged separately (`hand_detection_failed`).
- Model bundles are downloaded/cached lazily via `mediapipe_models`.
"""

from __future__ import annotations

import math

import cv2
import mediapipe as mp
import numpy as np

from backend.features.frame_sampling import FrameSample, window_indices
from backend.features.mediapipe_models import (
    create_face_detector,
    create_hand_landmarker,
    create_pose_landmarker,
)

# Pose landmark indices for the upper body (shoulders, elbows, wrists).
_UPPER_BODY_LANDMARKS = (11, 12, 13, 14, 15, 16)
_VISIBILITY_THRESHOLD = 0.5
# Maximum possible distance from frame center in normalized coords: sqrt(0.5^2+0.5^2).
_MAX_CENTER_DISTANCE = math.sqrt(0.5)

_FEATURE_KEYS = (
    "person_visible_ratio",
    "face_visible_ratio",
    "hand_visible_ratio",
    "upper_body_visible_ratio",
    "person_visible_ratio_first_3s",
    "face_visible_ratio_first_3s",
    "hand_visible_ratio_first_3s",
    "subject_centering_score",
    "subject_size_ratio",
    "face_size_ratio",
)


def _empty_features() -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {key: None for key in _FEATURE_KEYS}
    out["subject_size_ratio"] = 0.0
    out["face_size_ratio"] = 0.0
    out["subject_centering_missing"] = 1
    out["hand_detection_failed"] = 0
    return out


def _person_bbox(landmarks) -> tuple[float, float, float]:
    """Return (center_x, center_y, area) of the person bbox from pose landmarks.

    Coordinates are normalized (0-1), so the bbox area is already the fraction
    of the frame the subject occupies.
    """
    xs = [min(max(lm.x, 0.0), 1.0) for lm in landmarks]
    ys = [min(max(lm.y, 0.0), 1.0) for lm in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    area = (max_x - min_x) * (max_y - min_y)
    return (min_x + max_x) / 2.0, (min_y + max_y) / 2.0, area


def _centering_from_center(cx: float, cy: float) -> float:
    """1 - normalized distance of the subject center from the frame center."""
    distance = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
    return max(0.0, min(1.0, 1.0 - distance / _MAX_CENTER_DISTANCE))


def _build_detectors():
    """Create the three Tasks detectors (downloads model bundles if needed)."""
    return create_pose_landmarker(), create_face_detector(), create_hand_landmarker()


def extract_framing_features(
    sample: FrameSample,
    return_hand_positions: bool = False,
):
    """Compute framing/subject-visibility features (PRD 11.4) from sampled frames.

    When `return_hand_positions=True`, also returns the per-frame hand centroid
    (normalized (x, y) or None) computed during hand detection, so the motion
    step can reuse it instead of running the Hand Landmarker a second time.
    Returns either `features` or `(features, hand_positions)`.
    """
    if sample.is_empty:
        empty = _empty_features()
        return (empty, []) if return_hand_positions else empty

    timestamps = [f.timestamp for f in sample.frames]

    person_present: list[bool] = []
    upper_body_present: list[bool] = []
    face_present: list[bool] = []
    hand_present: list[bool] = []
    hand_positions: list[tuple[float, float] | None] = []
    centering_scores: list[float] = []  # only for frames where a person was found
    subject_sizes: list[float] = []
    face_sizes: list[float] = []
    hand_detection_failed = 0

    pose, face, hands = _build_detectors()
    try:
        for frame in sample.frames:
            rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
            frame_h, frame_w = rgb.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # --- Pose / person ---
            pose_result = pose.detect(mp_image)
            if pose_result.pose_landmarks:
                landmarks = pose_result.pose_landmarks[0]
                person_present.append(True)
                cx, cy, area = _person_bbox(landmarks)
                centering_scores.append(_centering_from_center(cx, cy))
                subject_sizes.append(area)
                upper_visible = all(
                    landmarks[i].visibility >= _VISIBILITY_THRESHOLD
                    for i in _UPPER_BODY_LANDMARKS
                )
                upper_body_present.append(upper_visible)
            else:
                person_present.append(False)
                upper_body_present.append(False)

            # --- Face (Tasks bbox is in pixels) ---
            face_result = face.detect(mp_image)
            if face_result.detections:
                box = face_result.detections[0].bounding_box
                area_px = max(0, box.width) * max(0, box.height)
                face_present.append(True)
                face_sizes.append(area_px / float(frame_w * frame_h) if frame_w and frame_h else 0.0)
            else:
                face_present.append(False)

            # --- Hands (also capture the centroid for motion reuse) ---
            try:
                hand_result = hands.detect(mp_image)
                if hand_result.hand_landmarks:
                    hand_present.append(True)
                    landmarks = hand_result.hand_landmarks[0]
                    cx = float(np.mean([lm.x for lm in landmarks]))
                    cy = float(np.mean([lm.y for lm in landmarks]))
                    hand_positions.append((cx, cy))
                else:
                    hand_present.append(False)
                    hand_positions.append(None)
            except Exception:  # noqa: BLE001 - technical failure differs from absence
                hand_detection_failed = 1
                hand_present.append(False)
                hand_positions.append(None)
    finally:
        pose.close()
        face.close()
        hands.close()

    def ratio(values: list[bool], window: str) -> float | None:
        idx = window_indices(timestamps, window)
        return round(float(np.mean([values[i] for i in idx])), 4) if idx else None

    features: dict[str, float | int | None] = {
        "person_visible_ratio": ratio(person_present, "full"),
        "face_visible_ratio": ratio(face_present, "full"),
        "hand_visible_ratio": ratio(hand_present, "full"),
        "upper_body_visible_ratio": ratio(upper_body_present, "full"),
        "person_visible_ratio_first_3s": ratio(person_present, "first_3s"),
        "face_visible_ratio_first_3s": ratio(face_present, "first_3s"),
        "hand_visible_ratio_first_3s": ratio(hand_present, "first_3s"),
    }

    # Subject centering: averaged over frames where a person was detected.
    if centering_scores:
        features["subject_centering_score"] = round(float(np.mean(centering_scores)), 4)
        features["subject_centering_missing"] = 0
    else:
        features["subject_centering_score"] = None
        features["subject_centering_missing"] = 1

    # Sizes: 0 when never detected (PRD 11.4), else mean over detected frames.
    features["subject_size_ratio"] = round(float(np.mean(subject_sizes)), 6) if subject_sizes else 0.0
    features["face_size_ratio"] = round(float(np.mean(face_sizes)), 6) if face_sizes else 0.0

    # If hands errored technically, report null rather than a misleading 0 ratio.
    if hand_detection_failed:
        features["hand_visible_ratio"] = None
        features["hand_visible_ratio_first_3s"] = None
    features["hand_detection_failed"] = hand_detection_failed

    if return_hand_positions:
        return features, hand_positions
    return features
