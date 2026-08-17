"""OCR text-presence features (PRD 11.7).

On-screen text is optional for instrumental covers. Presence and area still
come from EAST (PRD 11.7). Box geometry is also returned so
`text_semantics` can score *what* the text is doing without a second EAST pass.
OpenCV's DNN module with the EAST text detector, which localizes text regions as
(rotated) boxes. Text area per frame is measured by rasterizing the kept boxes
onto a mask, so overlapping detections don't double-count.

Failure semantics (PRD 12.5 / 22.2):
- No text detected -> all presence flags 0, area ratios 0, first timestamp null
  (a legitimate, neutral outcome).
- Detector/model error (or no frames) -> features null and `ocr_failed = 1`,
  which is distinct from "no text present".
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from backend.core.config import settings
from backend.features.frame_sampling import FrameSample, window_indices
from backend.features.ocr_models import ensure_east_model
from backend.features.text_semantics import TextDetection

# EAST output layers: per-cell text confidence, and box geometry.
_EAST_LAYERS = ("feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3")
# ImageNet mean EAST was trained with (BGR order before swapRB).
_EAST_MEAN = (123.68, 116.78, 103.94)
# Target longest side for the EAST input (rounded to a multiple of 32).
_EAST_MAX_SIDE = 320
_NMS_THRESHOLD = 0.4

_FEATURE_KEYS = (
    "text_present_anywhere",
    "text_present_first_1s",
    "text_present_first_3s",
    "text_area_ratio_full",
    "text_area_ratio_first_3s",
    "first_text_timestamp",
    "average_text_area_ratio_when_present",
)


def _empty_features(failed: bool) -> dict[str, float | int | None]:
    """No-text record (failed=False) or technical-failure record (failed=True)."""
    if failed:
        out: dict[str, float | int | None] = {key: None for key in _FEATURE_KEYS}
        out["ocr_failed"] = 1
        return out
    return {
        "text_present_anywhere": 0,
        "text_present_first_1s": 0,
        "text_present_first_3s": 0,
        "text_area_ratio_full": 0.0,
        "text_area_ratio_first_3s": 0.0,
        "first_text_timestamp": None,
        "average_text_area_ratio_when_present": 0.0,
        "ocr_failed": 0,
    }


def _east_dimensions(width: int, height: int) -> tuple[int, int]:
    """Resize target whose longest side ~= _EAST_MAX_SIDE, each a multiple of 32."""
    longest = max(width, height)
    scale = (_EAST_MAX_SIDE / longest) if longest else 1.0
    new_w = max(32, int(round(width * scale / 32.0)) * 32)
    new_h = max(32, int(round(height * scale / 32.0)) * 32)
    return new_w, new_h


def _decode_boxes(scores: np.ndarray, geometry: np.ndarray, conf_threshold: float):
    """Decode EAST score/geometry maps into rotated rects + confidences."""
    num_rows, num_cols = scores.shape[2:4]
    rects: list = []
    confidences: list[float] = []
    for y in range(num_rows):
        scores_row = scores[0, 0, y]
        x0, x1, x2, x3 = (geometry[0, i, y] for i in range(4))
        angles = geometry[0, 4, y]
        for x in range(num_cols):
            score = float(scores_row[x])
            if score < conf_threshold:
                continue
            offset_x, offset_y = x * 4.0, y * 4.0
            angle = float(angles[x])
            cos, sin = math.cos(angle), math.sin(angle)
            box_h = float(x0[x] + x2[x])
            box_w = float(x1[x] + x3[x])
            end_x = offset_x + cos * x1[x] + sin * x2[x]
            end_y = offset_y - sin * x1[x] + cos * x2[x]
            cx = end_x - 0.5 * box_w
            cy = end_y - 0.5 * box_h
            rects.append(((cx, cy), (box_w, box_h), -angle * 180.0 / math.pi))
            confidences.append(score)
    return rects, confidences


def _frame_text_analysis(
    net, image_bgr: np.ndarray, conf_threshold: float
) -> tuple[float, list[TextDetection]]:
    """Union text-area ratio plus normalized boxes for the semantics pass."""
    height, width = image_bgr.shape[:2]
    new_w, new_h = _east_dimensions(width, height)
    blob = cv2.dnn.blobFromImage(
        image_bgr, 1.0, (new_w, new_h), _EAST_MEAN, swapRB=True, crop=False
    )
    net.setInput(blob)
    scores, geometry = net.forward(_EAST_LAYERS)

    rects, confidences = _decode_boxes(scores, geometry, conf_threshold)
    if not rects:
        return 0.0, []

    keep = cv2.dnn.NMSBoxesRotated(rects, confidences, conf_threshold, _NMS_THRESHOLD)
    if keep is None or len(keep) == 0:
        return 0.0, []

    mask = np.zeros((new_h, new_w), dtype=np.uint8)
    detections: list[TextDetection] = []
    for idx in np.array(keep).flatten():
        points = cv2.boxPoints(rects[int(idx)])
        cv2.fillConvexPoly(mask, points.astype(np.int32), 1)
        xs, ys = points[:, 0], points[:, 1]
        bw = float(xs.max() - xs.min()) / float(new_w)
        bh = float(ys.max() - ys.min()) / float(new_h)
        cx = float((xs.min() + xs.max()) / 2.0) / float(new_w)
        cy = float((ys.min() + ys.max()) / 2.0) / float(new_h)
        detections.append(
            TextDetection(
                cx=float(np.clip(cx, 0.0, 1.0)),
                cy=float(np.clip(cy, 0.0, 1.0)),
                width=float(np.clip(bw, 0.0, 1.0)),
                height=float(np.clip(bh, 0.0, 1.0)),
                conf=float(confidences[int(idx)]),
            )
        )
    area = float(cv2.countNonZero(mask)) / float(new_w * new_h)
    return area, detections


def _frame_text_area_ratio(net, image_bgr: np.ndarray, conf_threshold: float) -> float:
    """Fraction of the frame covered by detected text boxes (0 if none)."""
    area, _ = _frame_text_analysis(net, image_bgr, conf_threshold)
    return area


def detect_text_on_sample(
    sample: FrameSample,
) -> tuple[dict[str, float | int | None], list[list[TextDetection]]]:
    """Presence features plus per-frame EAST boxes (empty lists on failure)."""
    if sample.is_empty:
        return _empty_features(failed=True), []

    try:
        net = cv2.dnn.readNet(ensure_east_model())
        conf = settings.ocr_confidence_threshold
        timestamps = [f.timestamp for f in sample.frames]
        area_ratios: list[float] = []
        detections_per_frame: list[list[TextDetection]] = []
        for frame in sample.frames:
            area, boxes = _frame_text_analysis(net, frame.image, conf)
            area_ratios.append(area)
            detections_per_frame.append(boxes)
    except Exception:  # noqa: BLE001 - any technical failure -> ocr_failed (PRD 12.5)
        empty_boxes = [[] for _ in sample.frames]
        return _empty_features(failed=True), empty_boxes

    present = [ratio > 0.0 for ratio in area_ratios]

    def present_in(window: str) -> int:
        idx = window_indices(timestamps, window)
        return int(any(present[i] for i in idx))

    def mean_in(window: str) -> float:
        idx = window_indices(timestamps, window)
        return float(np.mean([area_ratios[i] for i in idx])) if idx else 0.0

    detected = [area_ratios[i] for i, ok in enumerate(present) if ok]
    first_ts = next((timestamps[i] for i, ok in enumerate(present) if ok), None)

    features = {
        "text_present_anywhere": int(any(present)),
        "text_present_first_1s": present_in("first_1s"),
        "text_present_first_3s": present_in("first_3s"),
        "text_area_ratio_full": round(mean_in("full"), 6),
        "text_area_ratio_first_3s": round(mean_in("first_3s"), 6),
        "first_text_timestamp": first_ts,
        "average_text_area_ratio_when_present": (
            round(float(np.mean(detected)), 6) if detected else 0.0
        ),
        "ocr_failed": 0,
    }
    return features, detections_per_frame


def extract_ocr_features(sample: FrameSample) -> dict[str, float | int | None]:
    """Compute OCR text-presence features (PRD 11.7) from sampled frames."""
    features, _ = detect_text_on_sample(sample)
    return features
