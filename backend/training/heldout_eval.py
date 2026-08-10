"""Train-on-train / score-on-val-or-test evaluation for the primary classifier.

Uses the frozen creator split (D-001). Does not touch test unless `split="test"`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from backend.training.baselines import (
    ConstantProbaClassifier,
    RandomProbaClassifier,
    available_simple_features,
    build_rf_control_pipeline,
    build_simple_logistic_pipeline,
    classification_metrics,
)
from backend.training.creator_splits import (
    CreatorSplitMembership,
    indices_for_split,
    load_creator_splits,
)
from backend.training.feature_groups import select_group_features
from backend.training.model_dataset import PRIMARY_TARGET, ModelDataset
from backend.training.model_specs import PRIMARY_SPEC
from backend.training.validation import group_cv_splits, plan_group_cv


@dataclass
class HeldoutResult:
    name: str
    split: str
    metrics: dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    n_eval: int = 0
    message: str | None = None


def _xy_for_primary(
    dataset: ModelDataset,
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    features = feature_names or select_group_features(
        dataset.feature_names, PRIMARY_SPEC.feature_groups
    )
    y = dataset.target(PRIMARY_TARGET)
    mask = y.notna().to_numpy()
    X = dataset.X.loc[mask, features].reset_index(drop=True)
    y_arr = y[mask].to_numpy().astype(int)
    groups = dataset.groups[mask]
    return X, y_arr, groups


def _split_xy(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    membership: CreatorSplitMembership,
    eval_split: str,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    train_idx = indices_for_split(groups, membership, "train")
    eval_idx = indices_for_split(groups, membership, eval_split)
    return (
        X.iloc[train_idx].reset_index(drop=True),
        y[train_idx],
        X.iloc[eval_idx].reset_index(drop=True),
        y[eval_idx],
        groups[train_idx],
    )


def evaluate_estimator_heldout(
    name: str,
    estimator: Pipeline | object,
    dataset: ModelDataset,
    membership: CreatorSplitMembership,
    eval_split: str = "val",
    feature_names: list[str] | None = None,
) -> HeldoutResult:
    """Fit on train creators, score on val or test."""
    if eval_split not in {"val", "test"}:
        raise ValueError("eval_split must be 'val' or 'test'")

    X, y, groups = _xy_for_primary(dataset, feature_names)
    X_train, y_train, X_eval, y_eval, _ = _split_xy(
        X, y, groups, membership, eval_split
    )
    if len(X_eval) == 0 or len(X_train) == 0:
        return HeldoutResult(
            name=name,
            split=eval_split,
            message="Empty train or eval split",
            n_train=len(X_train),
            n_eval=len(X_eval),
        )

    est = clone(estimator) if hasattr(estimator, "get_params") else estimator
    est.fit(X_train, y_train)
    if hasattr(est, "predict_proba"):
        scores = est.predict_proba(X_eval)[:, 1]
    else:
        scores = np.asarray(est.predict(X_eval), dtype=float)

    return HeldoutResult(
        name=name,
        split=eval_split,
        metrics=classification_metrics(y_eval, scores),
        n_train=len(X_train),
        n_eval=len(X_eval),
    )


def evaluate_train_group_cv(
    name: str,
    estimator: Pipeline | object,
    dataset: ModelDataset,
    membership: CreatorSplitMembership,
    feature_names: list[str] | None = None,
    n_splits: int = 5,
) -> HeldoutResult:
    """GroupKFold OOF metrics restricted to train creators (model selection)."""
    X, y, groups = _xy_for_primary(dataset, feature_names)
    train_idx = indices_for_split(groups, membership, "train")
    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y[train_idx]
    g_train = groups[train_idx]

    plan = plan_group_cv(g_train, n_splits)
    if not plan.available:
        return HeldoutResult(
            name=name, split="train_cv", message=plan.message,
            n_train=len(X_train), n_eval=len(X_train),
        )

    splits = group_cv_splits(g_train, n_splits)
    oof = np.full(len(y_train), np.nan)
    for tr, te in splits:
        est = clone(estimator)
        est.fit(X_train.iloc[tr], y_train[tr])
        oof[te] = est.predict_proba(X_train.iloc[te])[:, 1]

    return HeldoutResult(
        name=name,
        split="train_cv",
        metrics=classification_metrics(y_train, oof),
        n_train=len(X_train),
        n_eval=len(X_train),
    )


def run_baseline_suite(
    dataset: ModelDataset,
    membership: CreatorSplitMembership | None = None,
    eval_split: str = "val",
    include_train_cv: bool = False,
) -> list[HeldoutResult]:
    """Majority, random, simple logistic, RF control on the frozen split."""
    membership = membership or load_creator_splits()
    results: list[HeldoutResult] = []

    # Majority / positive-rate (constant proba fitted on train labels)
    results.append(
        evaluate_estimator_heldout(
            "majority_positive_rate",
            ConstantProbaClassifier(),
            dataset,
            membership,
            eval_split=eval_split,
        )
    )
    results.append(
        evaluate_estimator_heldout(
            "random",
            RandomProbaClassifier(seed=42),
            dataset,
            membership,
            eval_split=eval_split,
        )
    )

    simple_feats = available_simple_features(dataset.feature_names)
    X_simple = dataset.X[simple_feats]
    # Build pipelines against the column subset frame shape
    simple_pipe = build_simple_logistic_pipeline(X_simple)
    results.append(
        evaluate_estimator_heldout(
            "simple_logistic",
            simple_pipe,
            dataset,
            membership,
            eval_split=eval_split,
            feature_names=simple_feats,
        )
    )

    primary_feats = select_group_features(
        dataset.feature_names, PRIMARY_SPEC.feature_groups
    )
    rf_pipe = build_rf_control_pipeline(dataset.X[primary_feats])
    results.append(
        evaluate_estimator_heldout(
            "rf_control",
            rf_pipe,
            dataset,
            membership,
            eval_split=eval_split,
            feature_names=primary_feats,
        )
    )

    if include_train_cv:
        results.append(
            evaluate_train_group_cv(
                "rf_control",
                build_rf_control_pipeline(dataset.X[primary_feats]),
                dataset,
                membership,
                feature_names=primary_feats,
            )
        )
        results.append(
            evaluate_train_group_cv(
                "simple_logistic",
                build_simple_logistic_pipeline(X_simple),
                dataset,
                membership,
                feature_names=simple_feats,
            )
        )

    return results
