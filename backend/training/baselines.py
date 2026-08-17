"""Registered baselines for the primary classifier (docs/DECISIONS.md D-003).

Baselines are fit on train creators only and scored on val/test with the same
metrics as the RandomForest control (ROC AUC, precision@k).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from backend.training.classifier import build_classifier_pipeline
from backend.training.metrics import classification_metrics
from backend.training.preprocessing import build_preprocessor_for
from backend.training.reproducibility import SKLEARN_RANDOM_STATE

# Hand-picked "obvious" presentation features (exist in current schema).
SIMPLE_LOGISTIC_FEATURES: tuple[str, ...] = (
    "brightness_mean_full",
    "contrast_full",
    "sharpness_full",
    "face_visible_ratio",
    "subject_centering_score",
)


@dataclass
class BaselineResult:
    name: str
    split: str
    metrics: dict[str, float]
    n_train: int
    n_eval: int


class ConstantProbaClassifier(BaseEstimator, ClassifierMixin):
    """Predicts a constant positive probability (majority / positive-rate baseline)."""

    def __init__(self, proba: float = 0.25):
        self.proba = float(proba)

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        self.proba_ = float(y.mean()) if len(y) else self.proba
        return self

    def predict_proba(self, X):
        n = len(X)
        p = np.full(n, self.proba_)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class RandomProbaClassifier(BaseEstimator, ClassifierMixin):
    """Uniform random scores in [0, 1] (seeded). Not fit-dependent beyond length."""

    def __init__(self, seed: int = SKLEARN_RANDOM_STATE):
        self.seed = seed

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = np.asarray(X).shape[1] if np.ndim(X) == 2 else 1
        return self

    def predict_proba(self, X):
        n = len(X)
        rng = np.random.RandomState(self.seed)
        p = rng.random_sample(n)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_simple_logistic_pipeline(feature_frame: pd.DataFrame) -> Pipeline:
    """Logistic regression on the fixed simple-feature subset (impute+scale)."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor_for(feature_frame)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=SKLEARN_RANDOM_STATE,
                ),
            ),
        ]
    )


def build_rf_control_pipeline(feature_frame: pd.DataFrame) -> Pipeline:
    """Current RandomForest control (same hyperparams as production classifier)."""
    return build_classifier_pipeline(feature_frame)


def build_hist_gradient_pipeline(feature_frame: pd.DataFrame) -> Pipeline:
    """Histogram gradient boosting baseline (impute+scale, class-balanced)."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor_for(feature_frame)),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_depth=6,
                    learning_rate=0.05,
                    max_iter=200,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    class_weight="balanced",
                    random_state=SKLEARN_RANDOM_STATE,
                ),
            ),
        ]
    )


def available_simple_features(columns: list[str]) -> list[str]:
    colset = set(columns)
    missing = [c for c in SIMPLE_LOGISTIC_FEATURES if c not in colset]
    if missing:
        raise ValueError(f"Simple logistic features missing from dataset: {missing}")
    return list(SIMPLE_LOGISTIC_FEATURES)


def fit_predict_scores(estimator, X_train, y_train, X_eval) -> np.ndarray:
    est = clone(estimator) if hasattr(estimator, "get_params") else estimator
    est.fit(X_train, y_train)
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X_eval)[:, 1]
    return np.asarray(est.predict(X_eval), dtype=float)
