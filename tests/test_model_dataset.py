"""Tests for leakage-safe model dataset loading (PRD 12.1 / 12.4 / 8.5)."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from backend.training.compute_labels import LABEL_FIELDS
from backend.training.model_dataset import (
    NON_FEATURE_COLUMNS,
    PRIMARY_TARGET,
    REGRESSION_TARGETS,
    ModelDataset,
    _coerce_numeric,
    assert_no_leakage,
    load_model_dataset,
    select_feature_columns,
)

import pandas as pd

# A couple of real feature columns + the status columns the loader must drop.
_FEATURE_COLS = ["duration_seconds", "has_audio", "person_visible_ratio"]
_ALL_COLS = (
    list(LABEL_FIELDS)
    + _FEATURE_COLS
    + ["audio_feature_extraction_status", "video_feature_extraction_status"]
)


def _write_dataset(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_ALL_COLS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in _ALL_COLS})


def _base_row(**overrides):
    row = {c: "" for c in _ALL_COLS}
    row.update(
        {
            "video_id": "v1",
            "creator_username": "amy",
            "duration_seconds": "18.4",
            "has_audio": "True",
            "person_visible_ratio": "0.9",
            "top_quartile_for_creator": "True",
            "engagement_rate": "0.1",
            "audio_feature_extraction_status": "ok",
            "video_feature_extraction_status": "complete",
        }
    )
    row.update(overrides)
    return row


def test_select_feature_columns_drops_labels_and_status():
    selected = select_feature_columns(_ALL_COLS)
    assert selected == _FEATURE_COLS  # only the real features, original order
    # nothing banned slipped through
    assert not (set(selected) & NON_FEATURE_COLUMNS)


def test_assert_no_leakage_flags_banned_columns():
    with pytest.raises(ValueError, match="Leakage"):
        assert_no_leakage(["duration_seconds", "views"])
    assert_no_leakage(["duration_seconds", "has_audio"])  # clean -> no raise


def test_coerce_numeric_handles_bools_blanks_numbers():
    out = _coerce_numeric(pd.Series(["True", "False", "", "0.5", "x"]))
    assert out.tolist()[:2] == [1.0, 0.0]
    assert out.tolist()[3] == 0.5
    assert np.isnan(out.tolist()[2]) and np.isnan(out.tolist()[4])


def test_load_model_dataset_is_leakage_safe(tmp_path):
    out = tmp_path / "training_dataset.csv"
    _write_dataset(out, [_base_row(), _base_row(video_id="v2", creator_username="bea")])
    ds = load_model_dataset(out)

    assert isinstance(ds, ModelDataset)
    assert ds.feature_names == _FEATURE_COLS
    # X is purely numeric (bool string coerced to float)
    assert ds.X["has_audio"].tolist() == [1.0, 1.0]
    assert ds.X["duration_seconds"].tolist() == [18.4, 18.4]
    # groups expose the creator key for GroupKFold
    assert list(ds.groups) == ["amy", "bea"]


def test_xy_drops_rows_missing_the_target(tmp_path):
    out = tmp_path / "training_dataset.csv"
    _write_dataset(
        out,
        [
            _base_row(video_id="v1", engagement_rate="0.1"),
            _base_row(video_id="v2", creator_username="bea", engagement_rate=""),
        ],
    )
    ds = load_model_dataset(out)

    # classifier target present on both rows
    X, y, groups = ds.xy(PRIMARY_TARGET)
    assert len(X) == len(y) == len(groups) == 2

    # engagement_rate missing on v2 -> that row excluded for this target only
    Xe, ye, ge = ds.xy("engagement_rate")
    assert len(Xe) == len(ye) == 1
    assert list(ge) == ["amy"]
    assert "engagement_rate" in REGRESSION_TARGETS
