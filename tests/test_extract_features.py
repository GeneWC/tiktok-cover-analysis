"""Orchestrator tests using a non-existent file.

A missing path makes every group fail, which exercises the failure-isolation
paths without decoding video or loading any ML model: metadata/audio raise on
open, sampling yields an empty FrameSample, and the visual/OCR extractors return
their null templates. The merged vector must still have the full, stable schema.
"""

from __future__ import annotations

from backend.features.extract_features import extract_all_features

_BOGUS = "this_video_does_not_exist_42.mp4"
_EXPECTED_FEATURE_COUNT = 112  # 89 + 14 text semantics + 5 MIR + 3 speech + 1 opening
_PIPELINE_GROUPS = {
    "metadata",
    "frame_sampling",
    "visual_quality",
    "framing",
    "motion",
    "audio",
    "ocr",
}


def test_missing_file_fails_every_step_without_raising():
    result = extract_all_features(_BOGUS)
    assert result.ok is False
    assert set(result.steps) == _PIPELINE_GROUPS
    assert all(status == "failed" for status in result.steps.values())


def test_feature_vector_schema_is_stable_and_complete():
    first = extract_all_features(_BOGUS).features
    second = extract_all_features(_BOGUS).features
    # Same keys in the same order on every run (stable training/inference schema).
    assert list(first.keys()) == list(second.keys())
    assert len(first) == _EXPECTED_FEATURE_COUNT


def test_hand_detection_key_collision_is_resolved():
    features = extract_all_features(_BOGUS).features
    assert "hand_detection_failed" in features  # from framing
    assert "hand_motion_detection_failed" in features  # renamed from motion


def test_required_feature_names_present_even_on_failure():
    features = extract_all_features(_BOGUS).features
    required = {
        "duration_seconds",
        "brightness_mean_full",
        "person_visible_ratio",
        "subject_centering_missing",
        "motion_energy_full",
        "audio_rms_mean",
        "audio_feature_extraction_status",
        "text_present_anywhere",
        "ocr_failed",
        "text_titlecard_ratio",
        "speech_ratio",
        "opening_text_plus_face",
        "audio_harmonic_ratio",
    }
    assert required <= set(features)
