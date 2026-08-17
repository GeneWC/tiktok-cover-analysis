"""Feature-group ablation for the primary classifier.

Runs on frozen creator splits (train fit, val score). Never touches the test
set. Use this to see whether a group is useful, redundant, or harmful — one
run is not enough to delete a feature.
"""

from __future__ import annotations

from backend.training.baselines import build_rf_control_pipeline
from backend.training.creator_splits import CreatorSplitMembership, load_creator_splits
from backend.training.feature_groups import GROUP_NAMES, assign_group, select_group_features
from backend.training.heldout_eval import HeldoutResult, evaluate_estimator_heldout
from backend.training.model_dataset import ModelDataset


def ablation_feature_sets(all_features: list[str]) -> dict[str, list[str]]:
    """Named feature subsets: all, leave-one-group-out, and each group alone."""
    all_feats = select_group_features(all_features, ())
    sets: dict[str, list[str]] = {"all": all_feats}
    for group in GROUP_NAMES:
        only = select_group_features(all_features, (group,))
        if only:
            sets[f"only_{group}"] = only
            sets[f"all_minus_{group}"] = [
                f for f in all_feats if assign_group(f) != group
            ]
    return {name: cols for name, cols in sets.items() if cols}


def run_group_ablation(
    dataset: ModelDataset,
    membership: CreatorSplitMembership | None = None,
    eval_split: str = "val",
) -> list[HeldoutResult]:
    """Fit the RF control on each ablation subset; score the held-out split."""
    if eval_split == "test":
        raise ValueError("Ablation must not use the frozen test set")
    membership = membership or load_creator_splits()
    results: list[HeldoutResult] = []
    for name, features in ablation_feature_sets(dataset.feature_names).items():
        present = [f for f in features if f in dataset.feature_names]
        if not present:
            continue
        pipe = build_rf_control_pipeline(dataset.X[present])
        results.append(
            evaluate_estimator_heldout(
                name,
                pipe,
                dataset,
                membership,
                eval_split=eval_split,
                feature_names=present,
            )
        )
    return results


def ablation_table(results: list[HeldoutResult]) -> list[dict]:
    """Compact rows for reports: name, n, roc_auc, pr_auc, brier, within-creator."""
    rows = []
    for result in results:
        row = {
            "name": result.name,
            "split": result.split,
            "n_train": result.n_train,
            "n_eval": result.n_eval,
            "message": result.message,
        }
        for key in (
            "roc_auc",
            "pr_auc",
            "brier",
            "ece",
            "within_creator_spearman",
            "within_creator_pairwise",
        ):
            if key in result.metrics:
                row[key] = result.metrics[key]
        rows.append(row)
    return rows
