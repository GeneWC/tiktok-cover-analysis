"""Direct tests for ORB + RANSAC camera-motion estimation."""

from __future__ import annotations

import cv2
import numpy as np

from backend.features.camera_motion import estimate_pairwise_camera_motion


def _scene(height: int = 180, width: int = 140, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width), dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    img[:] = np.clip(40 + 80 * (xx / width) + 50 * np.sin(yy / 8.0), 0, 255).astype(
        np.uint8
    )
    for _ in range(40):
        cv2.circle(
            img,
            (int(rng.integers(6, width - 6)), int(rng.integers(6, height - 6))),
            int(rng.integers(2, 7)),
            int(rng.integers(180, 255)),
            -1,
        )
    return img


def test_translation_is_detected():
    prev = _scene()
    matrix = np.float32([[1, 0, 8], [0, 1, 0]])
    curr = cv2.warpAffine(prev, matrix, (prev.shape[1], prev.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    motion = estimate_pairwise_camera_motion(prev, curr)
    assert motion is not None
    assert motion.translation > 0.02
    assert motion.inliers >= 6


def test_identical_grays_have_tiny_translation():
    prev = _scene()
    motion = estimate_pairwise_camera_motion(prev, prev.copy())
    if motion is None:
        return
    assert motion.translation < 0.01
