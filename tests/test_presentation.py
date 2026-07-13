"""Tests for presentation subscores (PRD 15)."""

from __future__ import annotations

from backend.inference.presentation import (
    compute_presentation_scores,
    _normalize,
)

# Percentiles for every feature the subscores reference (p5=0, p95=10 for easy math).
_PCTL = {
    feature: {"p5": 0.0, "p50": 5.0, "p95": 10.0}
    for feature in (
        "sharpness_full", "contrast_full", "colorfulness_full",
        "audio_dynamic_range", "audio_clipping_ratio", "audio_silence_ratio",
        "motion_consistency", "camera_stability_score",
        "person_visible_ratio", "face_visible_ratio",
        "subject_centering_score", "subject_size_ratio",
    )
}
_CAL = {"feature_percentiles": _PCTL}


def test_normalize_edges_midpoint_and_clip():
    assert _normalize(0.0, 0.0, 10.0, "up") == 0.0
    assert _normalize(10.0, 0.0, 10.0, "up") == 100.0
    assert _normalize(5.0, 0.0, 10.0, "up") == 50.0
    assert _normalize(20.0, 0.0, 10.0, "up") == 100.0   # clipped
    assert _normalize(-5.0, 0.0, 10.0, "up") == 0.0     # clipped


def test_normalize_down_direction_inverts():
    assert _normalize(0.0, 0.0, 10.0, "down") == 100.0
    assert _normalize(10.0, 0.0, 10.0, "down") == 0.0


def test_normalize_no_spread_returns_neutral():
    assert _normalize(3.0, 5.0, 5.0, "up") == 50.0


def test_visual_subscore_averages_directional_features():
    features = {"sharpness_full": 10.0, "contrast_full": 5.0, "colorfulness_full": 0.0}
    scores = compute_presentation_scores(features, _CAL)
    assert scores.visual_quality_score == 50.0  # (100 + 50 + 0) / 3


def test_audio_down_features_reward_low_values():
    # low clipping + low silence + high dynamic range -> high audio score
    features = {
        "audio_dynamic_range": 10.0, "audio_clipping_ratio": 0.0, "audio_silence_ratio": 0.0,
    }
    scores = compute_presentation_scores(features, _CAL)
    assert scores.audio_quality_score == 100.0


def test_missing_audio_yields_none_and_overall_ignores_it():
    features = {  # no audio_* features present at all
        "sharpness_full": 10.0, "contrast_full": 10.0, "colorfulness_full": 10.0,
        "motion_consistency": 10.0, "camera_stability_score": 10.0,
        "person_visible_ratio": 10.0, "face_visible_ratio": 10.0,
        "subject_centering_score": 10.0, "subject_size_ratio": 10.0,
    }
    scores = compute_presentation_scores(features, _CAL)
    assert scores.audio_quality_score is None
    assert scores.visual_quality_score == 100.0
    assert scores.overall_presentation_score == 100.0  # averages only computed dims


def test_all_missing_yields_all_none():
    scores = compute_presentation_scores({}, _CAL)
    assert scores.visual_quality_score is None
    assert scores.overall_presentation_score is None


def test_partial_features_skipped_not_penalized():
    # only one of three visual features present -> subscore uses just that one
    features = {"sharpness_full": 10.0}
    scores = compute_presentation_scores(features, _CAL)
    assert scores.visual_quality_score == 100.0
