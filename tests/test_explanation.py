"""Tests for the explanation/recommendation builder (PRD 16.5-16.9)."""

from __future__ import annotations

from backend.inference.explanation import (
    MAX_SIGNALS,
    SIGNAL_CATALOG,
    build_explanation,
)

# Uniform percentiles for every catalog feature: p25=1, p75=3.
_PCTL = {
    sig.feature: {"p5": 0.0, "p25": 1.0, "p50": 2.0, "p75": 3.0, "p95": 4.0}
    for sig in SIGNAL_CATALOG
}
_CAL = {"feature_percentiles": _PCTL}


def test_up_feature_high_is_strong_low_is_weak():
    strong = build_explanation({"sharpness_full": 3.5}, _CAL)
    assert any("Footage sharpness" in s for s in strong.strong_signals)

    weak = build_explanation({"sharpness_full": 0.5}, _CAL)
    assert any("Footage sharpness" in s for s in weak.weak_signals)
    assert weak.recommendations  # a weak signal produces a recommendation


def test_down_feature_direction_inverts():
    # audio_clipping_ratio is "down": low value -> strong, high -> weak
    strong = build_explanation({"audio_clipping_ratio": 0.5}, _CAL, has_audio=True)
    assert any("Audio clipping" in s for s in strong.strong_signals)

    weak = build_explanation({"audio_clipping_ratio": 3.5}, _CAL, has_audio=True)
    assert any("Audio clipping" in s for s in weak.weak_signals)


def test_typical_value_is_neither_strong_nor_weak():
    result = build_explanation({"sharpness_full": 2.0}, _CAL)
    assert not result.strong_signals
    assert not result.weak_signals


def test_no_audio_adds_neutral_note_and_skips_audio_signals():
    features = {"audio_clipping_ratio": 3.5, "sharpness_full": 3.5}
    result = build_explanation(features, _CAL, has_audio=False)
    assert any("No audio track" in n for n in result.neutral_or_missing_signals)
    assert not any("Audio clipping" in w for w in result.weak_signals)


def test_text_absence_is_neutral_not_negative():
    result = build_explanation({"text_present_anywhere": 0}, _CAL)
    assert any("No on-screen text" in n for n in result.neutral_or_missing_signals)
    assert not result.weak_signals


def test_ocr_failure_note():
    result = build_explanation({"ocr_failed": 1}, _CAL)
    assert any("text detection was unavailable" in n for n in result.neutral_or_missing_signals)


def test_signal_lists_are_capped():
    # every up-feature high + every down-feature low -> all strong, but capped
    features = {}
    for sig in SIGNAL_CATALOG:
        features[sig.feature] = 3.5 if sig.direction == "up" else 0.5
    result = build_explanation(features, _CAL, has_audio=True)
    assert len(result.strong_signals) == MAX_SIGNALS


def test_importances_prioritize_which_signals_surface():
    # many weak up-features, but only 5 slots; the highest-importance one must appear
    features = {sig.feature: 0.5 for sig in SIGNAL_CATALOG if sig.direction == "up"}
    importances = {"top_quartile": {"subject_size_ratio": 0.99}}
    result = build_explanation(features, _CAL, importances=importances)
    assert any("Subject size in frame" in w for w in result.weak_signals)
