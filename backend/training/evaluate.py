"""Model evaluation via cross-creator out-of-fold predictions (PRD 12.6 / 12.7).

The honest generalization estimate for every model. For each spec we build its
`Pipeline(preprocess-on-subset, estimator)`, run GroupKFold **by creator** (a
creator's videos are entirely in train or entirely in test), and collect
out-of-fold (OOF) predictions - each row scored only by folds that never saw its
creator. Metrics are computed on those OOF predictions, so they reflect how the
model does on *unseen creators*, not memorized ones.

Metrics (PRD 12.6):
- classification -> ROC-AUC and Precision@K (K = #positives), the ranking quality
  that matters for "which uploads look top-quartile".
- regression -> R^2, MAE, RMSE, and Spearman rank correlation (tiers are
  rank-based, so rank agreement is the meaningful signal).

If there are too few creators for grouped CV, the result carries a message and no
metrics instead of crashing (PRD 12.7 guard).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from backend.training.feature_groups import select_group_features
from backend.training.model_dataset import ModelDataset
from backend.training.model_specs import MODEL_SPECS, ModelSpec
from backend.training.train_models import build_pipeline_for_spec
from backend.training.validation import DEFAULT_N_SPLITS, group_cv_splits, plan_group_cv


@dataclass
class EvaluationResult:
    """OOF metrics for one model (or a message if grouped CV was unavailable)."""

    name: str
    task: str
    n_samples: int
    n_folds: int
    metrics: dict[str, float] = field(default_factory=dict)
    message: str | None = None


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of the top-k highest-scored rows that are truly positive."""
    k = max(1, min(k, len(scores)))
    top = np.argsort(scores)[::-1][:k]
    return float(np.mean(y_true[top]))


def _select(spec: ModelSpec, dataset: ModelDataset):
    """Return (X_subset, y, groups) for a spec, dropping rows missing the target."""
    features = select_group_features(dataset.feature_names, spec.feature_groups)
    y = dataset.target(spec.target)
    mask = y.notna().to_numpy()
    X = dataset.X.loc[mask, features].reset_index(drop=True)
    y = y[mask].to_numpy()
    groups = dataset.groups[mask]
    if spec.task == "classification":
        y = y.astype(int)
    return X, y, groups


def _oof_predictions(pipeline, X, y, splits) -> np.ndarray:
    """Out-of-fold scores: each row predicted by a fold blind to its creator."""
    is_clf = hasattr(pipeline.named_steps["model"], "predict_proba")
    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in splits:
        est = clone(pipeline)
        est.fit(X.iloc[train_idx], y[train_idx])
        if is_clf:
            oof[test_idx] = est.predict_proba(X.iloc[test_idx])[:, 1]
        else:
            oof[test_idx] = est.predict(X.iloc[test_idx])
    return oof


def _classification_metrics(y: np.ndarray, oof: np.ndarray) -> dict[str, float]:
    n_pos = int(y.sum())
    return {
        "roc_auc": float(roc_auc_score(y, oof)),
        "precision_at_k": precision_at_k(y, oof, k=n_pos),
        "positive_rate": float(y.mean()),
        "n_positive": n_pos,
    }


def _regression_metrics(y: np.ndarray, oof: np.ndarray) -> dict[str, float]:
    spearman = pd.Series(y).corr(pd.Series(oof), method="spearman")
    return {
        "r2": float(r2_score(y, oof)),
        "mae": float(mean_absolute_error(y, oof)),
        "rmse": float(np.sqrt(mean_squared_error(y, oof))),
        "spearman": float(spearman) if pd.notna(spearman) else 0.0,
    }


def evaluate_spec(
    spec: ModelSpec, dataset: ModelDataset, n_splits: int = DEFAULT_N_SPLITS
) -> EvaluationResult:
    """Cross-creator OOF metrics for a single model spec."""
    X, y, groups = _select(spec, dataset)
    plan = plan_group_cv(groups, n_splits)
    if not plan.available:
        return EvaluationResult(
            name=spec.name, task=spec.task, n_samples=len(y),
            n_folds=0, message=plan.message,
        )

    splits = group_cv_splits(groups, n_splits)
    pipeline = build_pipeline_for_spec(spec, X)
    oof = _oof_predictions(pipeline, X, y, splits)

    metrics = (
        _classification_metrics(y, oof)
        if spec.task == "classification"
        else _regression_metrics(y, oof)
    )
    return EvaluationResult(
        name=spec.name, task=spec.task, n_samples=len(y),
        n_folds=plan.n_splits, metrics=metrics,
    )


def evaluate_all(
    dataset: ModelDataset,
    specs: tuple[ModelSpec, ...] = MODEL_SPECS,
    n_splits: int = DEFAULT_N_SPLITS,
) -> dict[str, EvaluationResult]:
    """Evaluate every spec; returns {spec.name: EvaluationResult}."""
    return {spec.name: evaluate_spec(spec, dataset, n_splits) for spec in specs}
