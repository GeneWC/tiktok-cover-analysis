"""Tests for the preprocessing pipeline + artifacts (PRD 12.5 / 13)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backend.training.preprocessing import (
    classify_features,
    fit_preprocessor,
    load_preprocessor,
    save_preprocessing_artifacts,
)


def _frame():
    # `num` + `cont` are continuous; `flag` is boolean (values in {0,1}).
    # Row index 2 has a missing value in each of `num` and `flag`.
    return pd.DataFrame(
        {
            "num": [1.0, 2.0, np.nan, 5.0],
            "flag": [1.0, 0.0, np.nan, 1.0],
            "cont": [0.1, 0.2, 0.3, 0.4],
        }
    )


def test_classify_features_splits_boolean_from_numeric():
    numeric, boolean = classify_features(_frame())
    assert boolean == ["flag"]
    assert numeric == ["num", "cont"]


def test_transform_imputes_scales_and_zero_fills():
    fitted = fit_preprocessor(_frame())
    out = fitted.transform(_frame())

    # 4 rows x 3 features, fully finite (no NaN survives preprocessing)
    assert out.shape == (4, 3)
    assert np.isfinite(out).all()

    # transformed order is numeric block then boolean -> flag is the last column
    assert fitted.output_features == ["num", "cont", "flag"]
    # boolean NaN (row 2) was zero-filled, not scaled
    assert out[2, 2] == 0.0
    # numeric column was standard-scaled -> ~zero mean across rows
    assert abs(out[:, 0].mean()) < 1e-9


def test_missing_numeric_imputed_with_training_median():
    fitted = fit_preprocessor(_frame())
    scaler = fitted.preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    # median of [1, 2, 5] = 2.0; after scaling it maps to (2 - mean)/std.
    # The imputed row (index 2) must equal the scaled median exactly.
    out = fitted.transform(_frame())
    expected = (2.0 - scaler.mean_[0]) / scaler.scale_[0]
    assert out[2, 0] == expected


def test_save_and_load_roundtrip(tmp_path):
    fitted = fit_preprocessor(_frame())
    paths = save_preprocessing_artifacts(fitted, models_dir=tmp_path)

    for key in ("preprocessor", "imputer", "scaler", "feature_schema"):
        assert paths[key].exists()

    schema = json.loads((tmp_path / "feature_schema.json").read_text())
    assert schema["features"] == ["num", "flag", "cont"]
    assert schema["boolean_features"] == ["flag"]
    assert len(schema["features"]) == len(schema["numeric_features"]) + len(
        schema["boolean_features"]
    )

    reloaded = load_preprocessor(models_dir=tmp_path)
    np.testing.assert_array_equal(reloaded.transform(_frame()), fitted.transform(_frame()))
