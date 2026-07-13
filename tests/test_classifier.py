"""Tests for the primary classifier (PRD 12.1 / 12.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from backend.training.classifier import (
    CLASSIFIER_PARAMS,
    build_classifier,
    build_classifier_pipeline,
)


def _separable_frame(n=60, seed=0):
    """Feature `signal` decides the label; `noise` and a `flag` are distractors."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(size=n),
            "flag": rng.integers(0, 2, size=n).astype(float),
        }
    )
    y = (signal > 0).astype(int)
    return X, y


def test_build_classifier_uses_configured_params():
    clf = build_classifier()
    assert isinstance(clf, RandomForestClassifier)
    assert clf.class_weight == "balanced"
    assert clf.random_state == CLASSIFIER_PARAMS["random_state"]
    # oob flag is opt-in
    assert build_classifier(oob_score=True).oob_score is True


def test_pipeline_has_preprocess_and_model_steps():
    X, _ = _separable_frame()
    pipe = build_classifier_pipeline(X)
    assert list(dict(pipe.named_steps)) == ["preprocess", "model"]
    assert isinstance(pipe.named_steps["model"], RandomForestClassifier)


def test_pipeline_handles_missing_values_via_imputation():
    X, y = _separable_frame()
    X.loc[0, "signal"] = np.nan  # NaN must be imputed inside the pipeline
    pipe = build_classifier_pipeline(X)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.isfinite(proba).all()


def test_pipeline_is_deterministic():
    X, y = _separable_frame()
    p1 = build_classifier_pipeline(X).fit(X, y).predict_proba(X)
    p2 = build_classifier_pipeline(X).fit(X, y).predict_proba(X)
    # Same seed -> identical forest; allow only float rounding from parallel sums.
    np.testing.assert_allclose(p1, p2, atol=1e-12)


def test_classifier_learns_the_signal():
    X, y = _separable_frame(n=120)
    pipe = build_classifier_pipeline(X).fit(X, y)
    # On clearly separable data the model should fit the training labels well.
    assert (pipe.predict(X) == y).mean() > 0.9
