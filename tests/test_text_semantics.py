"""Text semantics: roles, redaction, lexicons. No EAST/EasyOCR required."""

from __future__ import annotations

import numpy as np

from backend.features.text_semantics import (
    TEXT_SEMANTIC_KEYS,
    TextDetection,
    extract_text_semantics_features,
    redact_ocr_text,
    role_for_box,
    tokenize,
)


def test_redact_strips_handles_urls_and_creator():
    raw = "Follow @onesemble https://tiktok.com/@x piano cover"
    cleaned = redact_ocr_text(raw, creator_username="onesemble")
    assert "@" not in cleaned
    assert "http" not in cleaned.lower()
    assert "onesemble" not in cleaned.lower()
    assert "piano" in cleaned.lower()
    assert "cover" in cleaned.lower()


def test_tokenize_letter_runs_only():
    assert tokenize("Part 2: Piano Cover!") == ["Part", "Piano", "Cover"]


def test_role_geometry():
    assert role_for_box(TextDetection(0.5, 0.45, 0.4, 0.2, 0.9)) == "titlecard"
    assert role_for_box(TextDetection(0.5, 0.88, 0.5, 0.1, 0.9)) == "caption"
    assert role_for_box(TextDetection(0.08, 0.08, 0.12, 0.08, 0.9)) == "corner"


def test_titlecard_and_cta_with_injected_reader(make_sample):
    image = np.zeros((120, 80, 3), dtype=np.uint8)
    sample = make_sample([image, image], [0.2, 1.0])
    boxes = [
        [TextDetection(0.5, 0.45, 0.5, 0.25, 0.99)],
        [TextDetection(0.5, 0.88, 0.4, 0.1, 0.99)],
    ]

    def read(_crops):
        return ["PIANO COVER", "Follow me"]

    feats = extract_text_semantics_features(sample, boxes, read_crops=read)
    assert feats["text_titlecard_ratio"] > 0
    assert feats["text_caption_ratio"] > 0
    assert feats["text_has_song_or_piece_cue"] == 1
    assert feats["text_has_cta"] == 1
    assert feats["text_read_failed"] == 0
    assert feats["text_char_count"] > 0
    assert feats["text_all_caps_ratio"] > 0


def test_missing_reader_is_null_not_zero(make_sample):
    image = np.zeros((60, 40, 3), dtype=np.uint8)
    sample = make_sample([image], [0.0])
    boxes = [[TextDetection(0.5, 0.5, 0.4, 0.2, 0.9)]]
    feats = extract_text_semantics_features(
        sample, boxes, read_crops=lambda _c: None
    )
    assert feats["text_titlecard_ratio"] > 0
    assert feats["text_char_count"] is None
    assert feats["text_has_cta"] is None
    assert feats["text_read_failed"] == 1


def test_no_boxes_are_zeros(make_sample):
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    sample = make_sample([image], [0.0])
    feats = extract_text_semantics_features(sample, [[]])
    assert feats["text_char_count"] == 0
    assert feats["text_has_cta"] == 0
    assert feats["text_read_failed"] == 0
    assert set(TEXT_SEMANTIC_KEYS) <= set(feats)


def test_empty_sample_failed(empty_sample):
    feats = extract_text_semantics_features(empty_sample, [])
    assert feats["text_read_failed"] == 1
    assert feats["text_titlecard_ratio"] is None
