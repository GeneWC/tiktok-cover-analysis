"""Tests for the training feature-extraction driver (PRD 9.2 / 8.5).

The real `extract_all_features` is stubbed so these run instantly and never touch
video files or ML models. We verify the CSV contract, the missing-file failed
status, and resumability.
"""

from __future__ import annotations

import csv

import pytest

import backend.training.extract_training_features as etf
from backend.collectors.base_collector import RAW_CSV_FIELDS
from backend.features.extract_features import FeatureExtractionResult

_FEATURES = {"feat_a": 1.0, "feat_b": 2.0}


def _fake_extract(path: str) -> FeatureExtractionResult:
    # Schema probe (non-existent path) and "bad" videos -> no frames.
    if "schema_probe" in path or "bad" in path:
        return FeatureExtractionResult(
            features={k: None for k in _FEATURES}, steps={}, frames_sampled=0
        )
    return FeatureExtractionResult(
        features=dict(_FEATURES), steps={}, frames_sampled=10, duration_seconds=5.0
    )


@pytest.fixture(autouse=True)
def stub_extractor(monkeypatch):
    monkeypatch.setattr(etf, "extract_all_features", _fake_extract)


def _write_raw(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_extracts_features_and_flags_missing_files(tmp_path):
    good = tmp_path / "good.mp4"
    good.write_bytes(b"\x00")
    raw = tmp_path / "raw.csv"
    _write_raw(
        raw,
        [
            {"video_id": "1", "video_file": str(good)},
            {"video_id": "2", "video_file": ""},  # no file -> failed (PRD 8.5)
        ],
    )
    out = tmp_path / "features.csv"

    summary = etf.build_video_features_csv(raw, out, progress=False)
    assert summary["processed"] == 2
    assert summary["complete"] == 1
    assert summary["failed"] == 1

    rows = {r["video_id"]: r for r in csv.DictReader(out.open(encoding="utf-8"))}
    assert rows["1"]["video_feature_extraction_status"] == "complete"
    assert rows["1"]["feat_a"] == "1.0"
    assert rows["2"]["video_feature_extraction_status"] == "failed"
    assert rows["2"]["feat_a"] == ""  # blank features for a missing file


def test_resumes_and_skips_already_processed(tmp_path):
    good = tmp_path / "good.mp4"
    good.write_bytes(b"\x00")
    raw = tmp_path / "raw.csv"
    _write_raw(
        raw,
        [
            {"video_id": "1", "video_file": str(good)},
            {"video_id": "2", "video_file": str(good)},
        ],
    )
    out = tmp_path / "features.csv"

    first = etf.build_video_features_csv(raw, out, limit=1, progress=False)
    assert first["processed"] == 1

    second = etf.build_video_features_csv(raw, out, progress=False)
    assert second["skipped_existing"] == 1
    assert second["processed"] == 1  # only the remaining video

    ids = [r["video_id"] for r in csv.DictReader(out.open(encoding="utf-8"))]
    assert ids == ["1", "2"]  # appended, no duplicates


def test_feature_names_match_extractor():
    assert etf.feature_names() == list(_FEATURES.keys())
