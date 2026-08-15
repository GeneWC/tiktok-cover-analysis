"""Within-creator residual (z-scored) features for channel-mode modeling (D-015).

Absolute presentation levels are creator-style; residuals ask whether a video is
unusual *for that creator*. Leave-one-out (LOO) z-scores avoid using a video's
own value in its baseline.

These transforms use features only (never labels). Creators are independent, so
computing residuals on the full matrix is equivalent to per-split computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_CREATOR_VIDEOS_FOR_RESIDUALS = 5  # channel-mode product floor
_EPS = 1e-8


def impute_median(X: pd.DataFrame, medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Fill NaNs with column medians. If `medians` is None, fit from `X`."""
    if medians is None:
        medians = X.median(numeric_only=True)
    filled = X.fillna(medians)
    # Columns that were entirely NaN stay NaN after fillna(median=NaN); zero-fill.
    filled = filled.fillna(0.0)
    return filled, medians


def within_creator_loo_zscore(
    X: pd.DataFrame,
    groups: np.ndarray | list[str],
) -> pd.DataFrame:
    """Leave-one-out z-score each column within each creator.

    Creators with n < 2 get zeros (undefined LOO baseline).
    Creators with n == 2 use the other point as mean and std=1.
    Near-zero LOO std is replaced with 1 so residuals stay finite.
    """
    groups = np.asarray(groups)
    if len(groups) != len(X):
        raise ValueError("groups length must match X rows")

    values = X.to_numpy(dtype=float, copy=True)
    out = np.zeros_like(values)
    feature_names = list(X.columns)

    for creator in np.unique(groups):
        idx = np.flatnonzero(groups == creator)
        n = int(idx.size)
        block = values[idx]
        if n < 2:
            out[idx] = 0.0
            continue

        total = np.nansum(block, axis=0)
        sumsq = np.nansum(block * block, axis=0)
        # Counts of non-nan per column (usually n after imputation).
        counts = np.sum(~np.isnan(block), axis=0).astype(float)
        counts = np.maximum(counts, 2.0)

        # LOO mean of the other n-1 points (per row).
        mean_loo = (total - np.nan_to_num(block, nan=0.0)) / (counts - 1.0)

        if n == 2:
            std_loo = np.ones_like(mean_loo)
        else:
            # Population variance of the other points:
            # E[x^2] - E[x]^2 on the leave-one-out set.
            mean_sq_loo = (sumsq - np.nan_to_num(block, nan=0.0) ** 2) / (counts - 1.0)
            var_loo = np.maximum(mean_sq_loo - mean_loo**2, 0.0)
            std_loo = np.sqrt(var_loo)
            std_loo = np.where(std_loo < _EPS, 1.0, std_loo)

        out[idx] = (np.nan_to_num(block, nan=0.0) - mean_loo) / std_loo

    return pd.DataFrame(out, columns=feature_names, index=X.index)


def within_creator_batch_zscore(
    X: pd.DataFrame,
    groups: np.ndarray | list[str],
) -> pd.DataFrame:
    """Z-score using the full creator batch mean/std (includes self).

    Slightly optimistic vs LOO for tiny n; useful as a channel-mode approximation
    when scoring a whole upload batch at once.
    """
    groups = np.asarray(groups)
    values = X.to_numpy(dtype=float, copy=True)
    out = np.zeros_like(values)

    for creator in np.unique(groups):
        idx = np.flatnonzero(groups == creator)
        block = values[idx]
        if idx.size < 2:
            out[idx] = 0.0
            continue
        mean = np.nanmean(block, axis=0)
        std = np.nanstd(block, axis=0)
        std = np.where(std < _EPS, 1.0, std)
        out[idx] = (np.nan_to_num(block, nan=0.0) - mean) / std

    return pd.DataFrame(out, columns=list(X.columns), index=X.index)


def residual_feature_names(feature_names: list[str], suffix: str = "_resid") -> list[str]:
    """Rename columns to make residual matrices obvious in logs."""
    return [f"{name}{suffix}" for name in feature_names]
