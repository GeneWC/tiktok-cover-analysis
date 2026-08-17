"""Shared classification, regression, calibration, and ranking metrics.

Kept independent of evaluate/baselines so both can import without a cycle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from backend.training.ranking_metrics import within_creator_metric

_EPS = 1e-15
_WITHIN_CREATOR_MIN_N = 5


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of the top-k highest-scored rows that are truly positive."""
    k = max(1, min(k, len(scores)))
    top = np.argsort(scores)[::-1][:k]
    return float(np.mean(y_true[top]))


def expected_calibration_error(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 10,
) -> float:
    """L1 expected calibration error over equal-width probability bins."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        if not np.any(mask):
            continue
        acc = float(y_true[mask].mean())
        conf = float(scores[mask].mean())
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Ranking, threshold, and calibration metrics for a binary score."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())
    metrics: dict[str, float] = {
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "n_positive": float(n_pos),
        "precision_at_k": precision_at_k(y_true, scores, k=max(n_pos, 1)),
    }
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        for key in (
            "roc_auc",
            "pr_auc",
            "f1",
            "balanced_accuracy",
            "log_loss",
            "brier",
            "ece",
        ):
            metrics[key] = float("nan")
        return metrics

    clipped = np.clip(scores, _EPS, 1.0 - _EPS)
    pred = (scores >= 0.5).astype(int)
    metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    metrics["pr_auc"] = float(average_precision_score(y_true, scores))
    metrics["f1"] = float(f1_score(y_true, pred, zero_division=0))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, pred))
    metrics["log_loss"] = float(log_loss(y_true, clipped))
    metrics["brier"] = float(brier_score_loss(y_true, np.clip(scores, 0.0, 1.0)))
    metrics["ece"] = expected_calibration_error(y_true, scores)
    return metrics


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    spearman = pd.Series(y_true).corr(pd.Series(pred), method="spearman")
    return {
        "r2": float(r2_score(y_true, pred)),
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "spearman": float(spearman) if pd.notna(spearman) else 0.0,
    }


def attach_within_creator_metrics(
    metrics: dict[str, float],
    y_true: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray | None,
    *,
    min_n: int = _WITHIN_CREATOR_MIN_N,
) -> dict[str, float]:
    """Add mean within-creator Spearman and pairwise rank accuracy when possible."""
    if groups is None or len(groups) != len(y_true):
        return metrics
    for metric_name, key in (
        ("spearman", "within_creator_spearman"),
        ("pairwise", "within_creator_pairwise"),
    ):
        result = within_creator_metric(
            y_true, scores, groups, metric=metric_name, min_n=min_n
        )
        if result["mean"] is not None:
            metrics[key] = float(result["mean"])
            metrics[f"{key}_n_creators"] = float(result["n_creators"])
    return metrics
