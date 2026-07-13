"""Tests for the training dataset builder (PRD 9.3 / 8.5)."""

from __future__ import annotations

import csv

from backend.collectors.base_collector import RAW_CSV_FIELDS
from backend.training.build_training_dataset import build_training_dataset


def _write_raw(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_features(path, rows):
    cols = ["video_id", "video_feature_extraction_status", "feat_a", "feat_b"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make(tmp_path):
    _write_raw(
        tmp_path / "raw.csv",
        [
            {"video_id": "1", "creator_username": "amy", "views": 1000, "likes": 100, "comments": 10, "shares": 5},
            {"video_id": "2", "creator_username": "amy", "views": 2000, "likes": 200, "comments": 20, "shares": 9},
            {"video_id": "3", "creator_username": "amy", "views": 500, "likes": 50, "comments": 5, "shares": 1},
        ],
    )
    _write_features(
        tmp_path / "features.csv",
        [
            {"video_id": "1", "video_feature_extraction_status": "complete", "feat_a": 0.5, "feat_b": 9},
            {"video_id": "2", "video_feature_extraction_status": "failed", "feat_a": "", "feat_b": ""},
            # video 3 has no feature row at all
        ],
    )


def test_join_excludes_failed_and_missing(tmp_path):
    _make(tmp_path)
    out = tmp_path / "dataset.csv"
    summary = build_training_dataset(
        tmp_path / "raw.csv", tmp_path / "features.csv", out
    )
    assert summary["labeled_videos"] == 3
    assert summary["written"] == 1            # only video 1 is complete
    assert summary["skipped_failed"] == 1     # video 2
    assert summary["skipped_no_features"] == 1  # video 3

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [r["video_id"] for r in rows] == ["1"]


def test_row_carries_labels_and_features(tmp_path):
    _make(tmp_path)
    out = tmp_path / "dataset.csv"
    build_training_dataset(tmp_path / "raw.csv", tmp_path / "features.csv", out)

    (row,) = list(csv.DictReader(out.open(encoding="utf-8")))
    # label/identifier columns present
    assert row["creator_username"] == "amy"
    assert row["like_rate"] == "0.1"
    assert row["top_quartile_for_creator"] in {"True", "False"}
    # feature columns joined in
    assert row["feat_a"] == "0.5"
    assert row["feat_b"] == "9"
    assert row["video_feature_extraction_status"] == "complete"


def test_header_has_labels_then_features(tmp_path):
    _make(tmp_path)
    out = tmp_path / "dataset.csv"
    build_training_dataset(tmp_path / "raw.csv", tmp_path / "features.csv", out)

    header = next(csv.reader(out.open(encoding="utf-8")))
    assert header[0] == "video_id"
    assert "creator_username" in header
    # feature columns appear after the label block; status is last.
    assert header.index("feat_a") > header.index("top_quartile_for_creator")
    assert header[-1] == "video_feature_extraction_status"
