"""Tests for model artifact export (PRD 13)."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from backend.training.evaluate import evaluate_all
from backend.training.export_artifacts import (
    CALIBRATION_FILE,
    FEATURE_SCHEMA_FILE,
    IMPORTANCES_FILE,
    METADATA_FILE,
    export_models,
    load_calibration,
    load_model_schema,
)
from backend.training.model_dataset import ModelDataset

_FEATURES = [
    "person_visible_ratio", "face_visible_ratio",
    "brightness_mean_full", "sharpness_full",
    "audio_rms_mean", "audio_silence_ratio",
    "motion_energy_full",
]


def _dataset(n=90, n_creators=6, seed=1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f: rng.random(n) for f in _FEATURES})
    frame = X.astype(str).copy()
    frame["creator_username"] = [f"c{i % n_creators}" for i in range(n)]
    frame["top_quartile_for_creator"] = np.where(X["person_visible_ratio"] > 0.5, "True", "False")
    frame["engagement_rate"] = [f"{v:.4f}" for v in rng.random(n)]
    frame["creator_relative_log_views"] = [f"{v:.4f}" for v in rng.normal(size=n)]
    frame["share_rate"] = [f"{v:.4f}" for v in rng.random(n)]
    return ModelDataset(frame=frame, X=X, feature_names=list(_FEATURES))


def test_export_writes_all_artifacts(tmp_path):
    written = export_models(_dataset(), out_dir=tmp_path)
    for artifact in (
        "top_quartile_classifier.pkl", "engagement_model.pkl",
        "creator_relative_regressor.pkl", "shareability_model.pkl",
        FEATURE_SCHEMA_FILE, IMPORTANCES_FILE, METADATA_FILE, CALIBRATION_FILE,
    ):
        assert (tmp_path / artifact).exists(), artifact
    assert written["top_quartile"].name == "top_quartile_classifier.pkl"


def test_calibration_artifact_is_loadable(tmp_path):
    ds = _dataset()
    export_models(ds, out_dir=tmp_path)
    calibration = load_calibration(tmp_path)
    assert set(calibration["regressor_tiers"]) == {
        "engagement", "creator_relative", "shareability"
    }
    assert set(calibration["feature_percentiles"]) == set(_FEATURES)


def test_saved_pipeline_predicts_on_full_feature_frame(tmp_path):
    ds = _dataset()
    export_models(ds, out_dir=tmp_path)
    pipeline = joblib.load(tmp_path / "top_quartile_classifier.pkl")
    schema = load_model_schema(tmp_path)
    features = schema["models"]["top_quartile"]["features"]
    proba = pipeline.predict_proba(ds.X[features])
    assert proba.shape == (len(ds.X), 2)


def test_schema_records_feature_contract(tmp_path):
    ds = _dataset()
    export_models(ds, out_dir=tmp_path)
    schema = load_model_schema(tmp_path)
    assert schema["all_features"] == _FEATURES
    tq = schema["models"]["top_quartile"]
    # primary classifier consumes only framing + visual
    assert tq["features"] == [
        "person_visible_ratio", "face_visible_ratio",
        "brightness_mean_full", "sharpness_full",
    ]
    assert schema["models"]["creator_relative"]["low_confidence"] is True


def test_importances_cover_each_models_features(tmp_path):
    ds = _dataset()
    export_models(ds, out_dir=tmp_path)
    importances = json.loads((tmp_path / IMPORTANCES_FILE).read_text())
    tq = importances["top_quartile"]
    assert set(tq) == {
        "person_visible_ratio", "face_visible_ratio",
        "brightness_mean_full", "sharpness_full",
    }
    assert abs(sum(tq.values()) - 1.0) < 1e-6  # RF importances sum to 1


def test_metadata_includes_versions_and_eval_metrics(tmp_path):
    ds = _dataset()
    evaluations = evaluate_all(ds)
    export_models(ds, out_dir=tmp_path, evaluations=evaluations)
    metadata = json.loads((tmp_path / METADATA_FILE).read_text())
    assert "sklearn_version" in metadata
    assert metadata["validation"]["scheme"] == "GroupKFold by creator"
    assert metadata["models"]["top_quartile"]["metrics"] is not None
    assert "roc_auc" in metadata["models"]["top_quartile"]["metrics"]
