"""Tests that inference rejects a changed feature contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.inference.feature_assembly import (
    FeatureSchemaError,
    assert_feature_schema_compatible,
    to_feature_frame,
)
from backend.inference.prediction import predict
from backend.training.export_artifacts import SCHEMA_VERSION, export_models, feature_fingerprint
from backend.inference.model_registry import load_registry
from backend.training.model_dataset import ModelDataset


_FEATURES = [
    "person_visible_ratio",
    "face_visible_ratio",
    "brightness_mean_full",
    "sharpness_full",
    "audio_rms_mean",
    "audio_silence_ratio",
    "motion_energy_full",
]


def _dataset(n=60, n_creators=6, seed=4):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f: rng.random(n) for f in _FEATURES})
    frame = X.astype(str).copy()
    frame["creator_username"] = [f"c{i % n_creators}" for i in range(n)]
    frame["top_quartile_for_creator"] = np.where(
        X["person_visible_ratio"] > 0.5, "True", "False"
    )
    frame["engagement_rate"] = [f"{v:.4f}" for v in rng.random(n)]
    frame["creator_relative_log_views"] = [f"{v:.4f}" for v in rng.normal(size=n)]
    frame["share_rate"] = [f"{v:.4f}" for v in rng.random(n)]
    return ModelDataset(frame=frame, X=X, feature_names=list(_FEATURES))


def test_committed_serving_schema_is_still_seventy_columns():
    import json
    from pathlib import Path

    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "backend" / "models" / "feature_schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert schema["schema_version"] == 1
    assert len(schema["all_features"]) == 70
    assert "text_has_cta" not in schema["all_features"]
    assert "speech_ratio" not in schema["all_features"]
    X = to_feature_frame({"a": 1.0, "b": 2.0}, ["a", "b"])
    with pytest.raises(FeatureSchemaError, match="mismatch"):
        assert_feature_schema_compatible(X, ["b", "a"])


def test_assert_rejects_fingerprint_drift():
    X = to_feature_frame({"a": 1.0, "b": 2.0}, ["a", "b"])
    with pytest.raises(FeatureSchemaError, match="fingerprint"):
        assert_feature_schema_compatible(
            X, ["a", "b"], expected_fingerprint="not-the-real-hash"
        )


def test_assert_accepts_matching_contract():
    names = ["a", "b"]
    X = to_feature_frame({"a": 1.0, "b": 2.0}, names)
    assert_feature_schema_compatible(
        X,
        names,
        schema_version=SCHEMA_VERSION,
        expected_fingerprint=feature_fingerprint(names),
    )


def test_predict_rejects_column_drift(tmp_path):
    ds = _dataset()
    export_models(ds, out_dir=tmp_path)
    registry = load_registry(tmp_path)
    drifted = ds.X.copy()
    drifted.columns = [f"{c}_v2" for c in drifted.columns]
    with pytest.raises(FeatureSchemaError, match="columns"):
        predict(drifted.iloc[[0]], registry=registry)
