"""Registered baselines for the primary classifier (docs/DECISIONS.md D-003).

Baselines are fit on train creators only and scored on val/test with the same
metrics as the RandomForest control (ROC AUC, precision@k).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from backend.training.classifier import build_classifier_pipeline
from backend.training.evaluate import precision_at_k
from backend.training.preprocessing import build_preprocessor_for

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

    def __init__(self, seed: int = 42):
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


def classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())
    metrics: dict[str, float] = {
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "n_positive": float(n_pos),
        "precision_at_k": precision_at_k(y_true, scores, k=max(n_pos, 1)),
    }
    if len(np.unique(y_true)) < 2:
        metrics["roc_auc"] = float("nan")
        metrics["brier"] = float("nan")
        return metrics
    metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    metrics["brier"] = float(brier_score_loss(y_true, np.clip(scores, 0.0, 1.0)))
    return metrics


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
                    random_state=42,
                ),
            ),
        ]
    )


def build_rf_control_pipeline(feature_frame: pd.DataFrame) -> Pipeline:
    """Current RandomForest control (same hyperparams as production classifier)."""
    return build_classifier_pipeline(feature_frame)


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
