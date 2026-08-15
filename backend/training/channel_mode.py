"""Channel-mode scoring: rank videos relative to a creator's own baseline (D-015).

Given ≥ MIN_CREATOR_VIDEOS_FOR_RESIDUALS feature rows from one creator, compute
leave-one-out residual features and score with a residual-trained classifier.

This is the product path that can use within-creator signal; single-upload
absolute models remain separate and exploratory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from backend.training.baselines import build_rf_control_pipeline, classification_metrics
from backend.training.creator_residuals import (
    MIN_CREATOR_VIDEOS_FOR_RESIDUALS,
    impute_median,
    within_creator_loo_zscore,
)
from backend.training.creator_splits import indices_for_split, load_creator_splits
from backend.training.feature_groups import select_group_features
from backend.training.model_dataset import PRIMARY_TARGET, ModelDataset, load_model_dataset
from backend.training.model_specs import PRIMARY_SPEC


@dataclass
class ChannelScoreResult:
    video_ids: list[str]
    scores: list[float]
    n_videos: int
    feature_names: list[str]
    message: str | None = None


def fit_residual_classifier(
    dataset: ModelDataset | None = None,
    feature_names: list[str] | None = None,
) -> tuple[Pipeline, list[str], pd.Series]:
    """Fit RF on train-creator LOO residuals (framing+visual by default).

    Returns (pipeline, feature_names, train_medians).
    """
    dataset = dataset or load_model_dataset()
    membership = load_creator_splits()
    feature_names = feature_names or select_group_features(
        dataset.feature_names, PRIMARY_SPEC.feature_groups
    )
    y = dataset.target(PRIMARY_TARGET)
    mask = y.notna().to_numpy()
    X = dataset.X.loc[mask, feature_names].reset_index(drop=True)
    y_arr = y[mask].to_numpy().astype(int)
    groups = dataset.groups[mask]

    train_idx = indices_for_split(groups, membership, "train")
    _, medians = impute_median(X.iloc[train_idx])
    X_imp, _ = impute_median(X, medians=medians)
    X_loo = within_creator_loo_zscore(X_imp, groups)

    pipe = build_rf_control_pipeline(X_loo.iloc[train_idx])
    pipe.fit(X_loo.iloc[train_idx], y_arr[train_idx])
    return pipe, feature_names, medians


def score_channel_features(
    feature_frame: pd.DataFrame,
    pipeline: Pipeline,
    feature_names: list[str],
    medians: pd.Series,
    video_ids: list[str] | None = None,
) -> ChannelScoreResult:
    """Score a batch of videos assumed to be from one creator.

    `feature_frame` must contain `feature_names` columns (absolute features).
    """
    n = len(feature_frame)
    ids = video_ids or [str(i) for i in range(n)]
    if n < MIN_CREATOR_VIDEOS_FOR_RESIDUALS:
        return ChannelScoreResult(
            video_ids=ids,
            scores=[float("nan")] * n,
            n_videos=n,
            feature_names=feature_names,
            message=(
                f"Need at least {MIN_CREATOR_VIDEOS_FOR_RESIDUALS} videos "
                f"for channel mode (got {n})."
            ),
        )

    X = feature_frame[feature_names].reset_index(drop=True)
    X_imp, _ = impute_median(X, medians=medians)
    # Single synthetic creator id for LOO within the batch.
    groups = np.array(["channel"] * n)
    X_loo = within_creator_loo_zscore(X_imp, groups)
    scores = pipeline.predict_proba(X_loo)[:, 1]
    return ChannelScoreResult(
        video_ids=ids,
        scores=[float(s) for s in scores],
        n_videos=n,
        feature_names=feature_names,
    )


def evaluate_channel_mode_on_split(
    eval_split: str = "val",
    dataset: ModelDataset | None = None,
) -> dict:
    """Fit on train residuals; score each eval creator as its own channel batch."""
    dataset = dataset or load_model_dataset()
    membership = load_creator_splits()
    pipe, feats, medians = fit_residual_classifier(dataset)

    y = dataset.target(PRIMARY_TARGET)
    mask = y.notna().to_numpy()
    frame = dataset.frame.loc[mask].reset_index(drop=True)
    X = dataset.X.loc[mask, feats].reset_index(drop=True)
    y_arr = y[mask].to_numpy().astype(int)
    groups = dataset.groups[mask]

    from sklearn.metrics import roc_auc_score

    eval_idx = indices_for_split(groups, membership, eval_split)
    creators = sorted(set(groups[eval_idx].astype(str)))

    all_scores = np.full(len(eval_idx), np.nan)
    all_y = y_arr[eval_idx]
    global_to_local = {int(g): i for i, g in enumerate(eval_idx)}

    per_creator = []
    for creator in creators:
        c_idx = np.flatnonzero(groups == creator)
        result = score_channel_features(
            X.iloc[c_idx],
            pipe,
            feats,
            medians,
            video_ids=(
                frame.iloc[c_idx]["video_id"].astype(str).tolist()
                if "video_id" in frame.columns
                else None
            ),
        )
        if result.message:
            per_creator.append(
                {
                    "creator": creator,
                    "n": int(c_idx.size),
                    "message": result.message,
                }
            )
            continue
        local_y = y_arr[c_idx]
        local_s = np.asarray(result.scores, dtype=float)
        for gi, score in zip(c_idx, local_s):
            all_scores[global_to_local[int(gi)]] = score
        auc = (
            float(roc_auc_score(local_y, local_s))
            if len(np.unique(local_y)) > 1
            else float("nan")
        )
        per_creator.append(
            {
                "creator": creator,
                "n": int(c_idx.size),
                "roc_auc": auc,
                "positive_rate": float(local_y.mean()),
            }
        )

    overall = classification_metrics(all_y, all_scores)
    return {
        "split": eval_split,
        "overall": overall,
        "per_creator": per_creator,
        "n_creators": len(creators),
    }
