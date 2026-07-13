"""Phase 4 focused stability check (exploratory).

The broad sweep suggested (a) feature selection helps a lot on `top_quartile`
(framing-only ~0.58 vs all-features ~0.49) and (b) `engagement_rate` is the
target with real cross-creator signal. With only 8 creators a single 5-fold
split is noisy, so this script re-checks the promising configs under
**leave-one-creator-out** (GroupKFold with n_splits = #creators), averaged over
several RandomForest seeds to gauge stability (mean +/- std of OOF AUC / R^2).

Run: python experiments/phase4_focused.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.model_dataset import load_model_dataset  # noqa: E402
from backend.training.preprocessing import build_preprocessor_for  # noqa: E402
from backend.training.validation import group_cv_splits  # noqa: E402
from experiments.phase4_signal_search import (  # noqa: E402
    FEATURE_GROUPS,
    group_features,
    precision_at_k,
    xy_for,
)

warnings.filterwarnings("ignore")
SEEDS = (42, 1, 7)

# Curated feature configurations to compare.
FEATURE_SETS = {
    "all": None,  # all features
    "framing": ("framing",),
    "framing+visual": ("framing", "visual"),
    "framing+visual+audio": ("framing", "visual", "audio"),
    "drop_motion_metadata": ("visual", "framing", "audio", "text"),
}


def _select(all_features, group_names):
    if group_names is None:
        return list(all_features)
    keywords = tuple(k for g in group_names for k in FEATURE_GROUPS[g])
    return group_features(all_features, keywords)


def _rf_clf(seed):
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, max_features="sqrt",
        class_weight="balanced", random_state=seed, n_jobs=-1,
    )


def _rf_reg(seed):
    return RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, max_features="sqrt",
        random_state=seed, n_jobs=-1,
    )


def _oof(X, y, splits, model, proba):
    out = np.full(len(y), np.nan)
    for tr, te in splits:
        pipe = Pipeline([("pre", build_preprocessor_for(X)), ("model", model)])
        pipe.fit(X.iloc[tr], y[tr])
        out[te] = pipe.predict_proba(X.iloc[te])[:, 1] if proba else pipe.predict(X.iloc[te])
    return out


def classification(ds, target):
    X_all, y, groups = xy_for(ds, target)
    y = y.astype(int)
    n_creators = len(set(groups))
    splits = group_cv_splits(groups, n_splits=n_creators)  # leave-one-creator-out
    print(f"\n[{target}]  n={len(y)}  pos_rate={y.mean():.3f}  "
          f"leave-one-creator-out ({n_creators} folds)")
    print(f"  {'feature_set':22s}  {'#f':>3s}  {'AUC(mean+/-std)':>18s}  {'P@20':>10s}")
    for name, groupset in FEATURE_SETS.items():
        feats = _select(list(X_all.columns), groupset)
        X = X_all[feats]
        aucs, precs = [], []
        for seed in SEEDS:
            oof = _oof(X, y, splits, _rf_clf(seed), proba=True)
            aucs.append(roc_auc_score(y, oof))
            precs.append(precision_at_k(y, oof, 20))
        print(f"  {name:22s}  {len(feats):3d}  "
              f"{np.mean(aucs):6.3f} +/- {np.std(aucs):.3f}     "
              f"{np.mean(precs):5.3f} +/- {np.std(precs):.3f}")


def regression(ds, target):
    X_all, y, groups = xy_for(ds, target)
    n_creators = len(set(groups))
    splits = group_cv_splits(groups, n_splits=n_creators)
    print(f"\n[{target}]  n={len(y)}  leave-one-creator-out ({n_creators} folds)")
    print(f"  {'feature_set':22s}  {'#f':>3s}  {'R2(mean+/-std)':>18s}")
    for name, groupset in FEATURE_SETS.items():
        feats = _select(list(X_all.columns), groupset)
        X = X_all[feats]
        r2s = [r2_score(y, _oof(X, y, splits, _rf_reg(seed), proba=False)) for seed in SEEDS]
        print(f"  {name:22s}  {len(feats):3d}  {np.mean(r2s):6.3f} +/- {np.std(r2s):.3f}")


def main():
    ds = load_model_dataset()
    print(f"Loaded {len(ds.frame)} rows, {len(ds.feature_names)} features, "
          f"{len(set(ds.groups))} creators.  (RF, {len(SEEDS)} seeds averaged)")
    print("\n" + "=" * 78 + "\nCLASSIFICATION (leave-one-creator-out)\n" + "=" * 78)
    classification(ds, "top_quartile_for_creator")
    classification(ds, "high_engagement_rate")
    print("\n" + "=" * 78 + "\nREGRESSION (leave-one-creator-out)\n" + "=" * 78)
    regression(ds, "engagement_rate")
    regression(ds, "creator_relative_log_views")


if __name__ == "__main__":
    main()
