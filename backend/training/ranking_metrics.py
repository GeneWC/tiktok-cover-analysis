"""Within-creator ranking metrics for continuous relative targets.

Used by Exp C5 (within-creator ranking / continuous targets). Metrics are
computed *inside* each creator so absolute level differences across creators
do not dominate Spearman / pairwise scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_MIN_CREATOR_N = 5


def spearman_corr(y_true, y_score) -> float | None:
    """Spearman rank correlation; None if undefined (constant / too short)."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    if len(y_true) < 2:
        return None
    if np.unique(y_true).size < 2 or np.unique(y_score).size < 2:
        return None
    corr = pd.Series(y_true).corr(pd.Series(y_score), method="spearman")
    if corr is None or not np.isfinite(corr):
        return None
    return float(corr)


def pairwise_rank_accuracy(y_true, y_score) -> float | None:
    """Fraction of comparable pairs where score order matches label order.

    Ties in either y_true or y_score are skipped. Returns None if no pairs.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    n = len(y_true)
    if n < 2:
        return None
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            dy = y_true[i] - y_true[j]
            ds = y_score[i] - y_score[j]
            if dy == 0.0 or ds == 0.0:
                continue
            total += 1
            if (dy > 0) == (ds > 0):
                correct += 1
    if total == 0:
        return None
    return float(correct / total)


def within_creator_metric(
    y_true,
    y_score,
    groups,
    *,
    metric: str = "spearman",
    min_n: int = DEFAULT_MIN_CREATOR_N,
) -> dict:
    """Per-creator ranking metric + mean across creators with enough videos.

    ``metric`` is ``\"spearman\"`` or ``\"pairwise\"``.
    """
    if metric == "spearman":
        fn = spearman_corr
    elif metric == "pairwise":
        fn = pairwise_rank_accuracy
    else:
        raise ValueError(f"Unknown metric '{metric}' (expected spearman|pairwise)")

    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    groups = np.asarray(groups).astype(str)

    per_creator: dict[str, float] = {}
    for creator in sorted(set(groups)):
        idx = np.flatnonzero(groups == creator)
        if len(idx) < min_n:
            continue
        val = fn(y_true[idx], y_score[idx])
        if val is not None:
            per_creator[creator] = val

    values = list(per_creator.values())
    mean = float(np.mean(values)) if values else None
    return {
        "metric": metric,
        "min_n": min_n,
        "n_creators": len(per_creator),
        "mean": mean,
        "per_creator": per_creator,
    }


def within_creator_zscore_features(
    X: pd.DataFrame,
    groups: np.ndarray | list[str],
) -> pd.DataFrame:
    """Z-score each feature within creator (ddof=0); constant cols -> 0."""
    groups = np.asarray(groups).astype(str)
    if len(groups) != len(X):
        raise ValueError("groups length must match X rows")
    out = X.copy()
    for creator in np.unique(groups):
        idx = np.flatnonzero(groups == creator)
        block = out.iloc[idx]
        mean = block.mean(axis=0)
        std = block.std(axis=0, ddof=0)
        std = std.replace(0.0, np.nan)
        z = (block - mean) / std
        out.iloc[idx] = z.fillna(0.0).to_numpy()
    return out
