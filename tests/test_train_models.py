"""Tests for spec-driven model training (PRD 12.1 / 12.2 / 8.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from backend.training.model_dataset import ModelDataset
from backend.training.model_specs import SPECS_BY_NAME
from backend.training.train_models import (
    build_pipeline_for_spec,
    fit_model_for_spec,
    train_all_models,
)

# Feature columns with real names so feature_groups can classify them.
_FEATURES = [
    "person_visible_ratio",   # framing
    "face_visible_ratio",     # framing
    "brightness_mean_full",   # visual
    "sharpness_full",         # visual
    "audio_rms_mean",         # audio
    "audio_silence_ratio",    # audio
    "motion_energy_full",     # motion
]


def _dataset(n=60, n_missing_engagement=12, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f: rng.random(n) for f in _FEATURES})

    engagement = [f"{v:.4f}" for v in rng.random(n)]
    for i in range(n_missing_engagement):  # blanks -> excluded for engagement
        engagement[i] = ""

    frame = X.astype(str).copy()
    frame["creator_username"] = np.where(np.arange(n) % 2 == 0, "amy", "bea")
    frame["top_quartile_for_creator"] = np.where(X["person_visible_ratio"] > 0.5, "True", "False")
    frame["engagement_rate"] = engagement
    frame["creator_relative_log_views"] = [f"{v:.4f}" for v in rng.normal(size=n)]
    frame["share_rate"] = [f"{v:.4f}" for v in rng.random(n)]
    return ModelDataset(frame=frame, X=X, feature_names=list(_FEATURES))


def test_build_pipeline_matches_task():
    ds = _dataset()
    clf = build_pipeline_for_spec(SPECS_BY_NAME["top_quartile"], ds.X)
    reg = build_pipeline_for_spec(SPECS_BY_NAME["engagement"], ds.X)
    assert isinstance(clf.named_steps["model"], RandomForestClassifier)
    assert isinstance(reg.named_steps["model"], RandomForestRegressor)


def test_primary_uses_framing_visual_subset():
    ds = _dataset(n=60)
    fitted = fit_model_for_spec(SPECS_BY_NAME["top_quartile"], ds)
    assert fitted.features == [
        "person_visible_ratio",
        "face_visible_ratio",
        "brightness_mean_full",
        "sharpness_full",
    ]
    assert fitted.n_samples == 60  # target present for all rows
    proba = fitted.pipeline.predict_proba(ds.X[fitted.features])
    assert proba.shape == (60, 2)


def test_engagement_uses_subset_and_drops_missing_rows():
    ds = _dataset(n=60, n_missing_engagement=12)
    fitted = fit_model_for_spec(SPECS_BY_NAME["engagement"], ds)
    # framing + visual + audio, no motion/metadata
    assert set(fitted.features) == {
        "person_visible_ratio", "face_visible_ratio",
        "brightness_mean_full", "sharpness_full",
        "audio_rms_mean", "audio_silence_ratio",
    }
    assert fitted.n_samples == 48  # 60 - 12 blank engagement rows (PRD 8.5)


def test_low_confidence_regressors_use_all_features():
    ds = _dataset()
    fitted = fit_model_for_spec(SPECS_BY_NAME["creator_relative"], ds)
    assert fitted.features == _FEATURES  # empty feature_groups -> all features
    assert fitted.spec.low_confidence


def test_train_all_models_returns_four_fitted_models():
    ds = _dataset()
    models = train_all_models(ds)
    assert set(models) == {"top_quartile", "engagement", "creator_relative", "shareability"}
    # each pipeline is fitted and predicts one value per input row on its subset
    for fitted in models.values():
        preds = fitted.pipeline.predict(ds.X[fitted.features])
        assert len(preds) == len(ds.X)
        assert np.isfinite(preds).all()
    assert models["engagement"].n_samples == 48  # blank-engagement rows excluded
