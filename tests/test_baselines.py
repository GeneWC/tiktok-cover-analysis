"""Tests for registered baselines + held-out eval harness."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.training.baselines import (
    ConstantProbaClassifier,
    RandomProbaClassifier,
    SIMPLE_LOGISTIC_FEATURES,
    classification_metrics,
)
from backend.training.creator_splits import make_creator_splits
from backend.training.heldout_eval import evaluate_estimator_heldout, run_baseline_suite
from backend.training.model_dataset import ModelDataset


def _toy_dataset(n_creators: int = 10, n_per: int = 12) -> ModelDataset:
    rows = []
    rng = np.random.RandomState(0)
    for c in range(n_creators):
        for i in range(n_per):
            bright = rng.random()
            rows.append(
                {
                    "creator_username": f"creator_{c}",
                    "video_id": f"{c}_{i}",
                    "top_quartile_for_creator": "true" if bright > 0.7 else "false",
                    "creator_relative_log_views": str(bright - 0.5),
                    "engagement_rate": str(0.05 + 0.01 * bright),
                    "share_rate": str(0.001 * bright),
                    "brightness_mean_full": str(bright),
                    "contrast_full": str(rng.random()),
                    "sharpness_full": str(rng.random()),
                    "face_visible_ratio": str(rng.random()),
                    "subject_centering_score": str(rng.random()),
                    # extra framing/visual so PRIMARY_SPEC groups resolve
                    "brightness_mean_first_3s": str(bright),
                    "contrast_first_3s": str(rng.random()),
                    "sharpness_first_3s": str(rng.random()),
                    "person_visible_ratio": str(rng.random()),
                    "face_visible_ratio_first_3s": str(rng.random()),
                    "hand_visible_ratio": str(rng.random()),
                    "subject_size_ratio": str(rng.random()),
                    "face_size_ratio": str(rng.random()),
                }
            )
    frame = pd.DataFrame(rows)
    # Build X like load_model_dataset would for the feature columns we care about
    feature_names = [
        c
        for c in frame.columns
        if c
        not in {
            "creator_username",
            "video_id",
            "top_quartile_for_creator",
            "creator_relative_log_views",
            "engagement_rate",
            "share_rate",
        }
    ]
    X = frame[feature_names].apply(pd.to_numeric, errors="coerce")
    return ModelDataset(frame=frame, X=X, feature_names=feature_names)


def test_constant_proba_uses_train_rate():
    clf = ConstantProbaClassifier()
    X = np.zeros((10, 2))
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    clf.fit(X, y)
    proba = clf.predict_proba(X[:3])[:, 1]
    assert np.allclose(proba, 0.2)


def test_random_proba_deterministic():
    a = RandomProbaClassifier(seed=1)
    b = RandomProbaClassifier(seed=1)
    X = np.zeros((5, 1))
    a.fit(X, np.array([0, 1, 0, 1, 0]))
    b.fit(X, np.array([0, 1, 0, 1, 0]))
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


def test_classification_metrics_keys():
    y = np.array([0, 1, 0, 1, 0, 0])
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.4])
    m = classification_metrics(y, scores)
    assert "roc_auc" in m and "precision_at_k" in m and "brier" in m
    assert m["roc_auc"] > 0.9


def test_baseline_suite_runs_on_toy():
    dataset = _toy_dataset()
    membership = make_creator_splits(dataset.groups, seed=42)
    # Ensure simple features exist
    for f in SIMPLE_LOGISTIC_FEATURES:
        assert f in dataset.feature_names
    results = run_baseline_suite(dataset, membership, eval_split="val")
    names = {r.name for r in results}
    assert names >= {"majority_positive_rate", "random", "simple_logistic", "rf_control"}
    for r in results:
        assert r.n_eval > 0
        assert "roc_auc" in r.metrics


def test_heldout_no_creator_leakage():
    dataset = _toy_dataset()
    membership = make_creator_splits(dataset.groups, seed=0)
    result = evaluate_estimator_heldout(
        "majority_positive_rate",
        ConstantProbaClassifier(),
        dataset,
        membership,
        eval_split="val",
    )
    assert result.n_train + result.n_eval <= len(dataset.frame)
    train_creators = set(membership.train_creators)
    val_creators = set(membership.val_creators)
    assert train_creators.isdisjoint(val_creators)
