"""Exp C6: within-creator residual features for top_quartile (channel-mode proxy).

Protocol (D-015):
- Impute medians on train creators only, apply to all.
- LOO / batch z-score within each creator (features only).
- Train RF on train-creator residuals; score val (selection) + train GroupKFold.
- Auto-touch test only if a LOO residual candidate beats absolute control by
  >= 0.02 on BOTH val and train_cv. Or pass --run-test to force control vs
  framing_visual LOO on test.

    python scripts/experiments/run_creator_residual_eval.py
    python scripts/experiments/run_creator_residual_eval.py --run-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.features.audio_features import AUDIO_STRUCTURE_FEATURE_KEYS  # noqa: E402
from backend.training.baselines import (  # noqa: E402
    build_rf_control_pipeline,
    classification_metrics,
)
from backend.training.creator_residuals import (  # noqa: E402
    impute_median,
    within_creator_batch_zscore,
    within_creator_loo_zscore,
)
from backend.training.creator_splits import (  # noqa: E402
    indices_for_split,
    load_creator_splits,
)
from backend.training.feature_groups import select_group_features  # noqa: E402
from backend.training.model_dataset import PRIMARY_TARGET, load_model_dataset  # noqa: E402
from backend.training.model_specs import PRIMARY_SPEC  # noqa: E402
from backend.training.validation import group_cv_splits, plan_group_cv  # noqa: E402

OUT = PROJECT_ROOT / "data" / "reports" / "creator_residual_eval.json"
PROMISING_MARGIN = 0.02


def _primary_xy(dataset, feature_names):
    y = dataset.target(PRIMARY_TARGET)
    mask = y.notna().to_numpy()
    X = dataset.X.loc[mask, feature_names].reset_index(drop=True)
    y_arr = y[mask].to_numpy().astype(int)
    groups = dataset.groups[mask]
    return X, y_arr, groups


def _train_cv_metrics(X, y, groups):
    plan = plan_group_cv(groups, 5)
    if not plan.available:
        return {"message": plan.message}
    oof = np.full(len(y), np.nan)
    pipe0 = build_rf_control_pipeline(X)
    for tr, te in group_cv_splits(groups, 5):
        est = clone(pipe0)
        est.fit(X.iloc[tr], y[tr])
        oof[te] = est.predict_proba(X.iloc[te])[:, 1]
    return classification_metrics(y, oof)


def _heldout_metrics(X, y, groups, membership, eval_split: str):
    tr = indices_for_split(groups, membership, "train")
    ev = indices_for_split(groups, membership, eval_split)
    est = build_rf_control_pipeline(X.iloc[tr])
    est.fit(X.iloc[tr], y[tr])
    scores = est.predict_proba(X.iloc[ev])[:, 1]
    return classification_metrics(y[ev], scores), len(tr), len(ev)


def _prepare_matrices(dataset, feature_names, membership):
    X_raw, y, groups = _primary_xy(dataset, feature_names)
    train_idx = indices_for_split(groups, membership, "train")
    _, medians = impute_median(X_raw.iloc[train_idx])
    X_imp, _ = impute_median(X_raw, medians=medians)
    return {
        "absolute": X_imp,
        "loo_residual": within_creator_loo_zscore(X_imp, groups),
        "batch_residual": within_creator_batch_zscore(X_imp, groups),
        "y": y,
        "groups": groups,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="Force test eval for framing_visual absolute vs loo_residual.",
    )
    args = parser.parse_args(argv)

    dataset = load_model_dataset()
    membership = load_creator_splits()

    framing_visual = select_group_features(
        dataset.feature_names, PRIMARY_SPEC.feature_groups
    )
    structure = [f for f in AUDIO_STRUCTURE_FEATURE_KEYS if f in dataset.feature_names]
    feature_sets = {
        "framing_visual": framing_visual,
        "framing_visual_structure": framing_visual + structure,
        "audio": select_group_features(dataset.feature_names, ("audio",)),
        "all": list(dataset.feature_names),
    }

    results = []
    control_val_auc = None
    control_cv_auc = None

    for set_name, feats in feature_sets.items():
        mats = _prepare_matrices(dataset, feats, membership)
        y, groups = mats["y"], mats["groups"]
        tr = indices_for_split(groups, membership, "train")
        for feat_kind in ("absolute", "loo_residual", "batch_residual"):
            X = mats[feat_kind]
            name = f"rf_{set_name}_{feat_kind}"
            cv_metrics = _train_cv_metrics(
                X.iloc[tr].reset_index(drop=True), y[tr], groups[tr]
            )
            val_metrics, n_train, n_val = _heldout_metrics(
                X, y, groups, membership, "val"
            )
            row = {
                "name": name,
                "feature_set": set_name,
                "feat_kind": feat_kind,
                "n_features": len(feats),
                "n_train": n_train,
                "n_val": n_val,
                "train_cv": cv_metrics,
                "val": val_metrics,
                "promising": False,
            }
            print(
                f"{name:48s} cv={cv_metrics.get('roc_auc', float('nan')):.4f} "
                f"val={val_metrics.get('roc_auc', float('nan')):.4f}"
            )
            if set_name == "framing_visual" and feat_kind == "absolute":
                control_val_auc = val_metrics.get("roc_auc")
                control_cv_auc = cv_metrics.get("roc_auc")
            results.append(row)

    promising: list[str] = []
    for row in results:
        if row["feat_kind"] != "loo_residual":
            continue
        val_auc = float(row["val"].get("roc_auc") or 0.0)
        cv_auc = float(row["train_cv"].get("roc_auc") or 0.0)
        if (
            control_val_auc is not None
            and control_cv_auc is not None
            and val_auc >= control_val_auc + PROMISING_MARGIN
            and cv_auc >= control_cv_auc + PROMISING_MARGIN
        ):
            row["promising"] = True
            promising.append(row["name"])

    test_rows = []
    names_to_test: list[str] = list(promising)
    if args.run_test:
        for required in (
            "rf_framing_visual_absolute",
            "rf_framing_visual_loo_residual",
        ):
            if required not in names_to_test:
                names_to_test.append(required)

    for row in results:
        if row["name"] not in names_to_test:
            continue
        feats = feature_sets[row["feature_set"]]
        mats = _prepare_matrices(dataset, feats, membership)
        X, y, groups = mats[row["feat_kind"]], mats["y"], mats["groups"]
        test_metrics, n_train, n_test = _heldout_metrics(
            X, y, groups, membership, "test"
        )
        test_rows.append(
            {
                "name": row["name"],
                "test": test_metrics,
                "n_train": n_train,
                "n_test": n_test,
            }
        )
        print(
            f"TEST {row['name']:43s} "
            f"auc={test_metrics.get('roc_auc', float('nan')):.4f} "
            f"p@k={test_metrics.get('precision_at_k', float('nan')):.4f}"
        )

    payload = {
        "control_val_auc": control_val_auc,
        "control_cv_auc": control_cv_auc,
        "promising": promising,
        "results": results,
        "test": test_rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"PROMISING: {promising or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
