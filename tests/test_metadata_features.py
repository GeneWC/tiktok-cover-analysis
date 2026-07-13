"""Unit tests for the metadata feature record (no-video / empty case)."""

from __future__ import annotations

from backend.features.metadata_features import _empty_metadata

_EXPECTED_KEYS = {
    "duration_seconds",
    "fps",
    "width",
    "height",
    "aspect_ratio",
    "resolution_area",
    "bitrate",
    "has_audio",
    "is_vertical_video",
    "is_square_video",
}


def test_empty_metadata_has_all_keys():
    assert set(_empty_metadata()) == _EXPECTED_KEYS


def test_empty_metadata_sensible_defaults():
    meta = _empty_metadata()
    assert meta["has_audio"] is False
    assert meta["resolution_area"] == 0
    assert meta["duration_seconds"] is None
    assert meta["bitrate"] is None
