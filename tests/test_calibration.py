"""Tests for serving calibration (PRD 12.2 / 15)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.training.calibration import (
    TIER_ORDER,
    compute_calibration,
    tier_for,
)
from backend.training.model_dataset import ModelDataset
from backend.training.train_models import train_all_models

_FEATURES = [
    "person_visible_ratio", "face_visible_ratio",
    "brightness_mean_full", "sharpness_full",
    "audio_rms_mean", "audio_silence_ratio",
    "motion_energy_full",
]


def _dataset(n=90, n_creators=6, seed=2):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f: rng.random(n) for f in _FEATURES})
    frame = X.astype(str).copy()
    frame["creator_username"] = [f"c{i % n_creators}" for i in range(n)]
    frame["top_quartile_for_creator"] = np.where(X["person_visible_ratio"] > 0.5, "True", "False")
    frame["engagement_rate"] = [f"{v:.4f}" for v in rng.random(n)]
    frame["creator_relative_log_views"] = [f"{v:.4f}" for v in rng.normal(size=n)]
    frame["share_rate"] = [f"{v:.4f}" for v in rng.random(n)]
    return ModelDataset(frame=frame, X=X, feature_names=list(_FEATURES))


def test_tier_for_maps_value_to_expected_tier():
    thresholds = {"q25": 1.0, "q50": 2.0, "q75": 3.0}
    assert tier_for(0.5, thresholds) == "low"
    assert tier_for(1.5, thresholds) == "medium"
    assert tier_for(2.5, thresholds) == "medium_high"
    assert tier_for(3.5, thresholds) == "high"
    # boundaries are inclusive at the lower edge
    assert tier_for(1.0, thresholds) == "medium"
    assert tier_for(3.0, thresholds) == "high"


def test_calibration_has_thresholds_for_each_regressor_only():
    ds = _dataset()
    models = train_all_models(ds)
    calibration = compute_calibration(ds, models)

    assert calibration["tier_order"] == list(TIER_ORDER)
    # regressors only, classifier excluded
    assert set(calibration["regressor_tiers"]) == {
        "engagement", "creator_relative", "shareability"
    }
    for entry in calibration["regressor_tiers"].values():
        t = entry["thresholds"]
        assert t["q25"] <= t["q50"] <= t["q75"]
        assert entry["tier_name"] is not None


def test_feature_percentiles_cover_every_feature_and_are_ordered():
    ds = _dataset()
    calibration = compute_calibration(ds, train_all_models(ds))
    percentiles = calibration["feature_percentiles"]
    assert set(percentiles) == set(_FEATURES)
    for stats in percentiles.values():
        assert stats["p5"] <= stats["p50"] <= stats["p95"]
