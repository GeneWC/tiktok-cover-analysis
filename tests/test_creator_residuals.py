"""Tests for within-creator residual feature transforms (D-015)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.training.creator_residuals import (
    impute_median,
    within_creator_batch_zscore,
    within_creator_loo_zscore,
)


def test_impute_median_uses_provided_medians():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, np.nan, 2.0]})
    filled, medians = impute_median(X, medians=pd.Series({"a": 10.0, "b": 20.0}))
    assert filled.loc[1, "a"] == 10.0
    assert filled.loc[0, "b"] == 20.0
    assert medians["a"] == 10.0


def test_loo_zscore_known_values():
    # Creator A: values 0, 2, 4  -> LOO means for each: (2+4)/2=3, (0+4)/2=2, (0+2)/2=1
    # std of others: for n=3, std of {2,4}=1, {0,4}=2, {0,2}=1
    X = pd.DataFrame({"f": [0.0, 2.0, 4.0, 10.0, 12.0]})
    groups = np.array(["a", "a", "a", "b", "b"])
    Z = within_creator_loo_zscore(X, groups)
    # First row: (0-3)/1 = -3
    assert Z.loc[0, "f"] == pytest.approx(-3.0)
    # Second: (2-2)/2 = 0
    assert Z.loc[1, "f"] == pytest.approx(0.0)
    # Third: (4-1)/1 = 3
    assert Z.loc[2, "f"] == pytest.approx(3.0)
    # Creator b n=2: (10-12)/1 = -2, (12-10)/1 = 2
    assert Z.loc[3, "f"] == pytest.approx(-2.0)
    assert Z.loc[4, "f"] == pytest.approx(2.0)


def test_batch_zscore_zero_mean_per_creator():
    X = pd.DataFrame({"f": [1.0, 3.0, 5.0, 7.0]})
    groups = np.array(["a", "a", "b", "b"])
    Z = within_creator_batch_zscore(X, groups)
    assert Z.loc[[0, 1], "f"].mean() == pytest.approx(0.0)
    assert Z.loc[[2, 3], "f"].mean() == pytest.approx(0.0)


def test_singleton_creator_is_zero():
    X = pd.DataFrame({"f": [5.0, 1.0, 2.0]})
    groups = np.array(["solo", "a", "a"])
    Z = within_creator_loo_zscore(X, groups)
    assert Z.loc[0, "f"] == 0.0
