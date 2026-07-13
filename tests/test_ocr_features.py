"""Unit tests for OCR presence handling (no-text vs failed), no model needed.

An empty sample returns early before the EAST model is loaded, so these tests
never trigger the ~96MB model download.
"""

from __future__ import annotations

from backend.features.ocr_features import (
    _FEATURE_KEYS,
    _empty_features,
    extract_ocr_features,
)


def test_empty_sample_marks_ocr_failed(empty_sample):
    features = extract_ocr_features(empty_sample)
    assert features["ocr_failed"] == 1
    # Failure is distinct from "no text": presence is null, not 0.
    assert features["text_present_first_3s"] is None


def test_no_text_defaults_are_zero_not_null():
    features = _empty_features(failed=False)
    assert features["ocr_failed"] == 0
    assert features["text_present_anywhere"] == 0
    assert features["text_present_first_3s"] == 0
    assert features["text_area_ratio_full"] == 0.0
    assert features["first_text_timestamp"] is None
    assert features["average_text_area_ratio_when_present"] == 0.0


def test_failed_record_is_all_null():
    features = _empty_features(failed=True)
    assert features["ocr_failed"] == 1
    assert all(features[key] is None for key in _FEATURE_KEYS)
