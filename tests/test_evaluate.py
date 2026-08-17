"""Tests for cross-creator OOF evaluation (PRD 12.6 / 12.7)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.training.evaluate import (
    evaluate_all,
    evaluate_spec,
    precision_at_k,
)
from backend.training.model_dataset import ModelDataset
from backend.training.model_specs import SPECS_BY_NAME
from backend.training.validation import INSUFFICIENT_CREATORS_MESSAGE

_FEATURES = [
    "person_visible_ratio", "face_visible_ratio",   # framing
    "brightness_mean_full", "sharpness_full",        # visual
    "audio_rms_mean", "audio_silence_ratio",         # audio
    "motion_energy_full",                            # motion
]


def _dataset(n=90, n_creators=6, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f: rng.random(n) for f in _FEATURES})
    frame = X.astype(str).copy()
    frame["creator_username"] = [f"c{i % n_creators}" for i in range(n)]
    # top_quartile tracks person_visible_ratio (learnable signal)
    frame["top_quartile_for_creator"] = np.where(X["person_visible_ratio"] > 0.5, "True", "False")
    frame["engagement_rate"] = [f"{v:.4f}" for v in X["audio_rms_mean"] + rng.normal(0, 0.05, n)]
    frame["creator_relative_log_views"] = [f"{v:.4f}" for v in rng.normal(size=n)]
    frame["share_rate"] = [f"{v:.4f}" for v in rng.random(n)]
    return ModelDataset(frame=frame, X=X, feature_names=list(_FEATURES))


def test_precision_at_k_picks_top_scored():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    assert precision_at_k(y, scores, k=2) == 1.0
    assert precision_at_k(y, scores, k=4) == 0.5


def test_classification_evaluation_reports_ranking_metrics():
    result = evaluate_spec(SPECS_BY_NAME["top_quartile"], _dataset())
    assert result.task == "classification"
    assert result.n_folds > 0
    assert set(result.metrics) >= {"roc_auc", "precision_at_k", "positive_rate", "n_positive"}
    assert 0.0 <= result.metrics["roc_auc"] <= 1.0
    # signal is learnable, so ranking should beat chance
    assert result.metrics["roc_auc"] > 0.5


def test_regression_evaluation_reports_error_and_rank_metrics():
    result = evaluate_spec(SPECS_BY_NAME["engagement"], _dataset())
    assert result.task == "regression"
    assert set(result.metrics) >= {"r2", "mae", "rmse", "spearman"}
    assert result.metrics["mae"] >= 0.0


def test_insufficient_creators_yields_message_not_crash():
    ds = _dataset(n=30, n_creators=1)
    result = evaluate_spec(SPECS_BY_NAME["top_quartile"], ds)
    assert result.metrics == {}
    assert result.message == INSUFFICIENT_CREATORS_MESSAGE
    assert result.n_folds == 0


def test_evaluate_all_covers_every_spec():
    results = evaluate_all(_dataset())
    assert set(results) == {"top_quartile", "engagement", "creator_relative", "shareability"}
