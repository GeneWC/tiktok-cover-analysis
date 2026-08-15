"""Tests for channel-mode scoring helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.training.channel_mode import score_channel_features
from backend.training.creator_residuals import MIN_CREATOR_VIDEOS_FOR_RESIDUALS
from backend.training.baselines import build_rf_control_pipeline
from backend.training.creator_residuals import impute_median, within_creator_loo_zscore


def test_score_channel_rejects_small_batch():
    feats = ["brightness_mean_full", "contrast_full"]
    X = pd.DataFrame({f: np.random.RandomState(0).rand(3) for f in feats})
    # Tiny dummy pipeline trained on residuals
    groups = np.array(["c"] * 8)
    train = pd.DataFrame({f: np.random.RandomState(1).rand(8) for f in feats})
    imp, med = impute_median(train)
    loo = within_creator_loo_zscore(imp, groups)
    pipe = build_rf_control_pipeline(loo)
    pipe.fit(loo, np.array([0, 1, 0, 1, 0, 1, 0, 1]))

    result = score_channel_features(X, pipe, feats, med)
    assert result.message is not None
    assert "Need at least" in result.message


def test_score_channel_returns_probabilities():
    feats = ["brightness_mean_full", "contrast_full", "sharpness_full", "face_visible_ratio", "subject_centering_score"]
    rng = np.random.RandomState(0)
    n_train = 40
    train = pd.DataFrame({f: rng.rand(n_train) for f in feats})
    groups = np.repeat([f"c{i}" for i in range(8)], 5)
    y = (train["brightness_mean_full"] > 0.5).astype(int).to_numpy()
    imp, med = impute_median(train)
    loo = within_creator_loo_zscore(imp, groups)
    pipe = build_rf_control_pipeline(loo)
    pipe.fit(loo, y)

    batch = pd.DataFrame({f: rng.rand(MIN_CREATOR_VIDEOS_FOR_RESIDUALS) for f in feats})
    result = score_channel_features(batch, pipe, feats, med, video_ids=[f"v{i}" for i in range(5)])
    assert result.message is None
    assert len(result.scores) == 5
    assert all(0.0 <= s <= 1.0 for s in result.scores)
