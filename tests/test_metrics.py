"""Tests for shared evaluation metrics and calibration error."""

from __future__ import annotations

import numpy as np
import pytest

from backend.training.metrics import (
    attach_within_creator_metrics,
    classification_metrics,
    expected_calibration_error,
    precision_at_k,
    regression_metrics,
)


def test_classification_metrics_include_ranking_and_calibration():
    y = np.array([0, 1, 0, 1, 0, 0, 1, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.25, 0.3, 0.7, 0.85])
    metrics = classification_metrics(y, scores)
    assert set(metrics) >= {
        "roc_auc",
        "pr_auc",
        "f1",
        "balanced_accuracy",
        "log_loss",
        "brier",
        "ece",
        "precision_at_k",
    }
    assert metrics["roc_auc"] > 0.9
    assert 0.0 <= metrics["ece"] < 0.5


def test_perfect_scores_have_zero_ece():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, scores, n_bins=4) == 0.0


def test_precision_at_k_reexport_behavior():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    assert precision_at_k(y, scores, k=2) == 1.0


def test_regression_metrics_keys():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.1, 1.9, 3.2, 3.8])
    metrics = regression_metrics(y, pred)
    assert set(metrics) == {"r2", "mae", "rmse", "spearman"}
    assert metrics["mae"] > 0.0


def test_within_creator_metrics_attach_when_enough_videos():
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    scores = y + 0.1
    groups = np.array(["a"] * 5 + ["b"] * 5)
    metrics = attach_within_creator_metrics({}, y, scores, groups, min_n=5)
    assert metrics["within_creator_spearman"] == pytest.approx(1.0)
    assert metrics["within_creator_pairwise"] == pytest.approx(1.0)
    assert metrics["within_creator_spearman_n_creators"] == 2.0
