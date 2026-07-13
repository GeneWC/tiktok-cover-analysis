"""Tests for the serving model registry (PRD 13 / 18.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.training.export_artifacts import export_models
from backend.inference.model_registry import get_registry, load_registry
from backend.training.model_dataset import ModelDataset

_FEATURES = [
    "person_visible_ratio", "face_visible_ratio",
    "brightness_mean_full", "sharpness_full",
    "audio_rms_mean", "audio_silence_ratio",
    "motion_energy_full",
]


def _dataset(n=90, n_creators=6, seed=3):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f: rng.random(n) for f in _FEATURES})
    frame = X.astype(str).copy()
    frame["creator_username"] = [f"c{i % n_creators}" for i in range(n)]
    frame["top_quartile_for_creator"] = np.where(X["person_visible_ratio"] > 0.5, "True", "False")
    frame["engagement_rate"] = [f"{v:.4f}" for v in rng.random(n)]
    frame["creator_relative_log_views"] = [f"{v:.4f}" for v in rng.normal(size=n)]
    frame["share_rate"] = [f"{v:.4f}" for v in rng.random(n)]
    return ModelDataset(frame=frame, X=X, feature_names=list(_FEATURES))


def test_registry_loads_all_artifacts(tmp_path):
    ds = _dataset()
    export_models(ds, out_dir=tmp_path)
    registry = load_registry(tmp_path)

    assert set(registry.models) == {
        "top_quartile", "engagement", "creator_relative", "shareability"
    }
    assert registry.all_features == _FEATURES
    assert registry.calibration["tier_order"]
    assert "top_quartile" in registry.importances


def test_registry_classifier_and_regressor_views(tmp_path):
    ds = _dataset()
    export_models(ds, out_dir=tmp_path)
    registry = load_registry(tmp_path)

    assert registry.classifier.name == "top_quartile"
    assert registry.classifier.task == "classification"
    assert {m.name for m in registry.regressors} == {
        "engagement", "creator_relative", "shareability"
    }


def test_loaded_pipeline_predicts_on_its_subset(tmp_path):
    ds = _dataset()
    export_models(ds, out_dir=tmp_path)
    model = load_registry(tmp_path).classifier
    proba = model.pipeline.predict_proba(ds.X[model.features])
    assert proba.shape == (len(ds.X), 2)


def test_missing_artifacts_fail_fast(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_models"):
        load_registry(tmp_path)  # empty dir -> no schema


def test_get_registry_is_cached():
    get_registry.cache_clear()
    first = get_registry()  # loads real artifacts from backend/models
    second = get_registry()
    assert first is second
    get_registry.cache_clear()
