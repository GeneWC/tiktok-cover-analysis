"""Visual quality features (PRD 11.3).

Computes brightness, contrast, sharpness, blur, and colorfulness from the
sampled frames, aggregated over time windows. Per-frame metrics are computed
once and then averaged within each window (cheap + consistent).

Definitions (PRD 11.3):
- brightness_mean   = average grayscale pixel intensity (0-255)
- contrast          = standard deviation of grayscale pixel intensity
- sharpness         = variance of the Laplacian (classic focus measure)
- blur              = normalized inverse of sharpness, 1 / (1 + sharpness)
- colorfulness      = Hasler-Susstrunk colorfulness metric

Output uses the exact "required final feature names" from PRD 11.3.
"""

from __future__ import annotations

import cv2
import numpy as np

from backend.features.frame_sampling import FrameSample, window_indices

# Windows each metric is emitted for (per the PRD's required final names).
_BRIGHTNESS_WINDOWS = ("first_1s", "first_3s", "first_6s", "full")
_DETAIL_WINDOWS = ("first_3s", "full")

# Full set of output keys, used for the empty/failed case.
_FEATURE_KEYS = (
    [f"brightness_mean_{w}" for w in _BRIGHTNESS_WINDOWS]
    + [f"contrast_{w}" for w in _DETAIL_WINDOWS]
    + [f"sharpness_{w}" for w in _DETAIL_WINDOWS]
    + [f"blur_{w}" for w in _DETAIL_WINDOWS]
    + [f"colorfulness_{w}" for w in _DETAIL_WINDOWS]
    + ["brightness_std_full", "contrast_std_full"]
)


def _colorfulness(image_bgr: np.ndarray) -> float:
    """Hasler-Susstrunk colorfulness on a BGR image."""
    b, g, r = cv2.split(image_bgr.astype("float32"))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std_root + 0.3 * mean_root)


def _frame_metrics(image_bgr: np.ndarray) -> tuple[float, float, float, float]:
    """Return (brightness, contrast, sharpness, colorfulness) for one frame."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    colorfulness = _colorfulness(image_bgr)
    return brightness, contrast, sharpness, colorfulness


def extract_visual_quality_features(sample: FrameSample) -> dict[str, float | None]:
    """Compute visual quality features (PRD 11.3) from sampled frames."""
    if sample.is_empty:
        return {key: None for key in _FEATURE_KEYS}

    timestamps = [f.timestamp for f in sample.frames]
    brightness, contrast, sharpness, colorfulness = (
        np.array(values)
        for values in zip(*(_frame_metrics(f.image) for f in sample.frames))
    )

    def window_mean(values: np.ndarray, window: str) -> float | None:
        indices = window_indices(timestamps, window)
        return float(values[indices].mean()) if indices else None

    features: dict[str, float | None] = {}

    for window in _BRIGHTNESS_WINDOWS:
        value = window_mean(brightness, window)
        features[f"brightness_mean_{window}"] = round(value, 4) if value is not None else None

    for window in _DETAIL_WINDOWS:
        contrast_value = window_mean(contrast, window)
        sharpness_value = window_mean(sharpness, window)
        colorfulness_value = window_mean(colorfulness, window)

        features[f"contrast_{window}"] = (
            round(contrast_value, 4) if contrast_value is not None else None
        )
        features[f"sharpness_{window}"] = (
            round(sharpness_value, 4) if sharpness_value is not None else None
        )
        features[f"blur_{window}"] = (
            round(1.0 / (1.0 + sharpness_value), 6) if sharpness_value is not None else None
        )
        features[f"colorfulness_{window}"] = (
            round(colorfulness_value, 4) if colorfulness_value is not None else None
        )

    features["brightness_std_full"] = (
        round(float(brightness.std()), 4) if brightness.size else None
    )
    features["contrast_std_full"] = (
        round(float(contrast.std()), 4) if contrast.size else None
    )

    return features
