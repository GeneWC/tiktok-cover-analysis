"""MediaPipe Tasks model-asset management.

MediaPipe 0.10.x exposes only the Tasks API, which loads model bundles
(.task/.tflite) from disk. This module downloads the required bundles on first
use and caches them locally, so the rest of the code can just ask for a path.
The files are large and regenerated/fetched on demand, so they are git-ignored.
"""

from __future__ import annotations

import urllib.request

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from backend.core.config import settings

# name -> (local filename, download URL). Using the lightweight float16 models.
_MODELS: dict[str, tuple[str, str]] = {
    "pose": (
        "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    ),
    "face": (
        "blaze_face_short_range.tflite",
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    ),
    "hand": (
        "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task",
    ),
}


def ensure_model(name: str) -> str:
    """Return a local path to the model bundle, downloading it if missing."""
    filename, url = _MODELS[name]
    destination = settings.mediapipe_models_dir / filename
    if not destination.exists():
        settings.mediapipe_models_dir.mkdir(parents=True, exist_ok=True)
        # Download to a temp file then rename, so an interrupted download never
        # leaves a corrupt "complete" file behind.
        tmp = destination.with_suffix(destination.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(destination)
    return str(destination)


def ensure_all_models() -> dict[str, str]:
    """Download (if needed) and return paths for all framing models."""
    return {name: ensure_model(name) for name in _MODELS}


# --- Tasks detector factories ---------------------------------------------
# Centralized here so framing and motion construct detectors with identical
# options and all MediaPipe configuration lives in one place. Each runs in
# IMAGE mode (sampled frames are independent, not consecutive video frames).


def create_pose_landmarker() -> "vision.PoseLandmarker":
    return vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=ensure_model("pose")),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
        )
    )


def create_face_detector() -> "vision.FaceDetector":
    return vision.FaceDetector.create_from_options(
        vision.FaceDetectorOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=ensure_model("face")),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
    )


def create_hand_landmarker() -> "vision.HandLandmarker":
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=ensure_model("hand")),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
        )
    )
