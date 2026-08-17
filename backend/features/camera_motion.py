"""Background-feature camera motion (ORB + RANSAC affine).

Separates global camera movement from local performer motion. Designed for
short vertical creator videos at the already-downscaled sample resolution
(typically <= 256 px on the long side), so ORB is cheap.

Returns None when tracking is unreliable instead of inventing a transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_MIN_KEYPOINTS = 12
_MIN_MATCHES = 8
_MIN_INLIERS = 6
_ORB_FEATURES = 400
_RANSAC_REPROJ = 3.0


@dataclass(frozen=True)
class PairwiseCameraMotion:
    """One consecutive-frame camera estimate, normalized by frame size."""

    translation: float  # |t| / frame diagonal
    rotation_deg: float
    scale_change: float  # |scale - 1|
    residual_energy: float  # mean absdiff after warping prev -> curr
    inliers: int


def _orb_descriptors(gray: np.ndarray, mask: np.ndarray | None):
    orb = cv2.ORB_create(nfeatures=_ORB_FEATURES)
    return orb.detectAndCompute(gray, mask)


def _person_mask(shape: tuple[int, int], person_box: tuple[float, float, float, float] | None):
    """Optional inverted person box: 255 = background (keep), 0 = person (drop)."""
    if person_box is None:
        return None
    height, width = shape
    x0, y0, x1, y1 = person_box
    mask = np.full((height, width), 255, dtype=np.uint8)
    xa, xb = int(max(0, x0) * width), int(min(1.0, x1) * width)
    ya, yb = int(max(0, y0) * height), int(min(1.0, y1) * height)
    if xb > xa and yb > ya:
        mask[ya:yb, xa:xb] = 0
        # If the box ate almost the whole frame, tracking would be empty.
        if mask.mean() < 20:
            return None
    return mask


def estimate_pairwise_camera_motion(
    gray_prev: np.ndarray,
    gray_curr: np.ndarray,
    person_box: tuple[float, float, float, float] | None = None,
) -> PairwiseCameraMotion | None:
    """Estimate a partial affine (translation / rotation / scale) with RANSAC."""
    if gray_prev.shape != gray_curr.shape or gray_prev.size == 0:
        return None
    height, width = gray_prev.shape[:2]
    mask = _person_mask((height, width), person_box)
    _kp1, des1 = _orb_descriptors(gray_prev, mask)
    kp2, des2 = _orb_descriptors(gray_curr, mask)
    if des1 is None or des2 is None or len(des1) < _MIN_KEYPOINTS or len(des2) < _MIN_KEYPOINTS:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if len(matches) < _MIN_MATCHES:
        return None
    matches = sorted(matches, key=lambda item: item.distance)[:120]

    src = np.float32([_kp1[m.queryIdx].pt for m in matches])
    dst = np.float32([kp2[m.trainIdx].pt for m in matches])
    matrix, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=_RANSAC_REPROJ,
    )
    if matrix is None or inliers is None:
        return None
    n_inliers = int(inliers.sum())
    if n_inliers < _MIN_INLIERS:
        return None

    tx, ty = float(matrix[0, 2]), float(matrix[1, 2])
    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    scale = float(np.hypot(a, b))
    rotation = float(np.degrees(np.arctan2(b, a)))
    diagonal = float(np.hypot(width, height)) or 1.0

    warped = cv2.warpAffine(
        gray_prev,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    residual = float(cv2.absdiff(warped, gray_curr).mean())

    return PairwiseCameraMotion(
        translation=float(np.hypot(tx, ty) / diagonal),
        rotation_deg=abs(rotation),
        scale_change=abs(scale - 1.0),
        residual_energy=residual,
        inliers=n_inliers,
    )


def camera_stability_from_pairs(pairs: list[PairwiseCameraMotion]) -> float:
    """1 = no global motion; approaches 0 as translation/rotation/scale grow."""
    if not pairs:
        raise ValueError("no camera pairs")
    translation = float(np.mean([p.translation for p in pairs]))
    rotation = float(np.mean([p.rotation_deg for p in pairs]))
    scale = float(np.mean([p.scale_change for p in pairs]))
    # Empirically: 2% of the diagonal, 4 degrees, or 4% scale is already shaky.
    score = 1.0 / (1.0 + translation / 0.02 + rotation / 4.0 + scale / 0.04)
    return float(max(0.0, min(1.0, score)))
