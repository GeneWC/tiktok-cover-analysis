"""Unit tests for within-creator ranking helpers (Exp C5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.training.ranking_metrics import (
    pairwise_rank_accuracy,
    spearman_corr,
    within_creator_metric,
    within_creator_zscore_features,
)


def test_spearman_perfect_and_inverse():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman_corr(y, y) == 1.0
    assert spearman_corr(y, -y) == -1.0


def test_spearman_undefined_on_constant():
    y = np.array([1.0, 2.0, 3.0])
    assert spearman_corr(y, np.ones(3)) is None


def test_pairwise_rank_accuracy_perfect():
    y = np.array([0.1, 0.5, 0.9])
    assert pairwise_rank_accuracy(y, y) == 1.0
    assert pairwise_rank_accuracy(y, -y) == 0.0


def test_within_creator_metric_means_per_creator():
    # Two creators: perfect rank on c0, inverse on c1
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    groups = np.array(["c0"] * 5 + ["c1"] * 5)
    out = within_creator_metric(y, pred, groups, metric="spearman", min_n=5)
    assert out["n_creators"] == 2
    assert out["per_creator"]["c0"] == pytest.approx(1.0)
    assert out["per_creator"]["c1"] == pytest.approx(-1.0)
    assert out["mean"] == pytest.approx(0.0)


def test_within_creator_zscore_zero_mean_unit_std():
    X = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "b": [5.0, 5.0, 5.0, 1.0, 2.0, 3.0],
        }
    )
    groups = np.array(["x", "x", "x", "y", "y", "y"])
    Z = within_creator_zscore_features(X, groups)
    for creator in ("x", "y"):
        block = Z.loc[groups == creator, "a"]
        assert abs(block.mean()) < 1e-9
        assert abs(block.std(ddof=0) - 1.0) < 1e-9
    # constant within creator -> zeros
    assert np.allclose(Z.loc[groups == "x", "b"].to_numpy(), 0.0)
