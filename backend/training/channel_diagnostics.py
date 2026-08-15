"""Channel diagnostics: within-creator hit/miss analysis without cross-creator claims.

For a batch of videos from one creator (optionally with labels), report:
- class balance / top-quartile membership if labels exist
- feature mean deltas (hits − misses) on leak-safe features
- within-creator LOO residual magnitudes for framing+visual features
- simple presentation-feature ranks

This is decision support for "what differs in MY better uploads," not a
virality forecast. See D-016.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from backend.training.creator_residuals import (
    MIN_CREATOR_VIDEOS_FOR_RESIDUALS,
    impute_median,
    within_creator_loo_zscore,
)
from backend.training.feature_groups import select_group_features
from backend.training.model_specs import PRIMARY_SPEC


@dataclass
class FeatureDelta:
    feature: str
    hit_mean: float
    miss_mean: float
    delta: float  # hit - miss


@dataclass
class ChannelDiagnostics:
    creator: str | None
    n_videos: int
    n_hits: int | None
    positive_rate: float | None
    top_feature_deltas: list[FeatureDelta] = field(default_factory=list)
    video_ranks: list[dict] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    message: str | None = None
    mode: str = "diagnostics"

    def to_dict(self) -> dict:
        payload = asdict(self)
        return payload


def recommendations_from_deltas(
    deltas: list[FeatureDelta], limit: int = 5
) -> list[str]:
    """Turn hit/miss deltas into short, actionable (non-forecast) tips."""
    tips: list[str] = []
    for d in deltas[:limit]:
        direction = "higher" if d.delta > 0 else "lower"
        tips.append(
            f"In your stronger uploads, {d.feature.replace('_', ' ')} tends to be "
            f"{direction} than in the rest of this batch "
            f"(delta {d.delta:+.3f}). Treat as a pattern to notice, not a rule."
        )
    return tips


def _coerce_labels(labels: pd.Series | np.ndarray | None) -> np.ndarray | None:
    if labels is None:
        return None
    if isinstance(labels, pd.Series):
        if labels.dtype == object or labels.dtype == "string":
            return labels.astype(str).str.lower().eq("true").to_numpy().astype(int)
        return pd.to_numeric(labels, errors="coerce").fillna(0).astype(int).to_numpy()
    arr = np.asarray(labels)
    if arr.dtype.kind in {"U", "O"}:
        return np.array([str(x).lower() == "true" for x in arr], dtype=int)
    return arr.astype(int)


def diagnose_channel(
    features: pd.DataFrame,
    *,
    labels: pd.Series | np.ndarray | None = None,
    video_ids: list[str] | None = None,
    creator: str | None = None,
    feature_names: list[str] | None = None,
    top_k_deltas: int = 10,
) -> ChannelDiagnostics:
    """Analyze one creator's video batch.

    `features` = absolute leak-safe feature matrix (one row per video).
    `labels` optional binary top-quartile (or any hit/miss) aligned to rows.
    """
    n = len(features)
    ids = video_ids or [str(i) for i in range(n)]
    if n < MIN_CREATOR_VIDEOS_FOR_RESIDUALS:
        return ChannelDiagnostics(
            creator=creator,
            n_videos=n,
            n_hits=None,
            positive_rate=None,
            message=(
                f"Need at least {MIN_CREATOR_VIDEOS_FOR_RESIDUALS} videos "
                f"for channel diagnostics (got {n})."
            ),
        )

    cols = feature_names or select_group_features(
        list(features.columns), PRIMARY_SPEC.feature_groups
    )
    cols = [c for c in cols if c in features.columns]
    X = features[cols].copy()
    X_imp, _ = impute_median(X)
    groups = np.array(["channel"] * n)
    X_loo = within_creator_loo_zscore(X_imp, groups)
    residual_norm = np.linalg.norm(X_loo.to_numpy(dtype=float), axis=1)

    # Presentation proxy: mean of available brightness/sharpness/face columns.
    present_cols = [
        c
        for c in (
            "brightness_mean_full",
            "sharpness_full",
            "contrast_full",
            "face_visible_ratio",
            "subject_centering_score",
        )
        if c in X_imp.columns
    ]
    if present_cols:
        present_score = X_imp[present_cols].mean(axis=1).to_numpy()
    else:
        present_score = X_imp.mean(axis=1).to_numpy()

    y = _coerce_labels(labels)
    deltas: list[FeatureDelta] = []
    n_hits = None
    pos_rate = None
    if y is not None and len(y) == n and y.sum() > 0 and y.sum() < n:
        n_hits = int(y.sum())
        pos_rate = float(y.mean())
        hits = X_imp.to_numpy()[y == 1]
        misses = X_imp.to_numpy()[y == 0]
        hit_mean = hits.mean(axis=0)
        miss_mean = misses.mean(axis=0)
        delta = hit_mean - miss_mean
        order = np.argsort(np.abs(delta))[::-1][:top_k_deltas]
        for i in order:
            deltas.append(
                FeatureDelta(
                    feature=cols[int(i)],
                    hit_mean=float(hit_mean[i]),
                    miss_mean=float(miss_mean[i]),
                    delta=float(delta[i]),
                )
            )

    ranks = []
    for i in range(n):
        ranks.append(
            {
                "video_id": ids[i],
                "presentation_score": float(present_score[i]),
                "residual_l2": float(residual_norm[i]),
                "label": int(y[i]) if y is not None and len(y) == n else None,
            }
        )
    ranks.sort(key=lambda r: r["presentation_score"], reverse=True)

    return ChannelDiagnostics(
        creator=creator,
        n_videos=n,
        n_hits=n_hits,
        positive_rate=pos_rate,
        top_feature_deltas=deltas,
        video_ranks=ranks,
        recommendations=recommendations_from_deltas(deltas),
        message=None,
    )
