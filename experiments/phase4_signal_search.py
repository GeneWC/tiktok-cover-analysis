"""Phase 4 signal-search experiment (exploratory, NOT production training).

The initial primary classifier showed cross-creator ROC-AUC ~= 0.49 (chance).
Before committing to models, this harness systematically asks "is there any
generalizable signal?" by sweeping:

  - targets      (top_quartile, outperformed_median, high_engagement, high_share;
                  regression: creator_relative_log_views, engagement_rate, share_rate)
  - models       (dummy baseline, logistic regression, random forest, grad boosting)
  - feature sets (all features + per-group ablations)
  - protocols    (honest GroupKFold-by-creator vs optimistic random StratifiedKFold)

For classification it reports out-of-fold ROC-AUC and Precision@K (the PRD's
headline metric, 12.6). For regression it reports out-of-fold R^2 and MAE.

Run: python experiments/phase4_signal_search.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.model_dataset import load_model_dataset  # noqa: E402
from backend.training.preprocessing import build_preprocessor_for  # noqa: E402
from backend.training.validation import group_cv_splits  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42

# --- feature groups (by name prefix / keyword) -----------------------------
FEATURE_GROUPS = {
    "metadata": ("duration_seconds", "fps", "width", "height", "aspect_ratio",
                 "resolution_area", "bitrate", "has_audio", "is_vertical", "is_square"),
    "visual": ("brightness", "contrast", "sharpness", "blur", "colorfulness"),
    "framing": ("person_visible", "face_visible", "hand_visible", "upper_body",
                "subject_centering", "subject_size", "face_size"),
    "motion": ("motion_energy", "motion_consistency", "hand_motion", "camera_stability"),
    "audio": ("audio_",),
    "text": ("text_", "first_text", "average_text", "ocr_failed"),
}


def group_features(all_features, keywords):
    return [f for f in all_features if any(k in f for k in keywords)]


# --- model factories --------------------------------------------------------
def clf_models():
    return {
        "dummy": lambda: DummyClassifier(strategy="stratified", random_state=SEED),
        "logreg": lambda: LogisticRegression(max_iter=3000, class_weight="balanced"),
        "rf": lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced", random_state=SEED, n_jobs=-1,
        ),
        "gb": lambda: GradientBoostingClassifier(random_state=SEED),
    }


def reg_models():
    return {
        "dummy": lambda: DummyRegressor(strategy="mean"),
        "ridge": lambda: Ridge(alpha=1.0),
        "rf": lambda: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, max_features="sqrt",
            random_state=SEED, n_jobs=-1,
        ),
        "gb": lambda: GradientBoostingRegressor(random_state=SEED),
    }


def make_pipe(X, model):
    return Pipeline([("pre", build_preprocessor_for(X)), ("model", model)])


# --- out-of-fold predictions ------------------------------------------------
def oof_predict(X, y, splits, model_factory, proba: bool):
    oof = np.full(len(y), np.nan)
    for tr, te in splits:
        pipe = make_pipe(X, model_factory())
        pipe.fit(X.iloc[tr], y[tr])
        if proba:
            oof[te] = pipe.predict_proba(X.iloc[te])[:, 1]
        else:
            oof[te] = pipe.predict(X.iloc[te])
    return oof


def precision_at_k(y_true, scores, k):
    order = np.argsort(scores)[::-1][:k]
    return float(y_true[order].mean())


def stratified_splits(y, n=5):
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))


def xy_for(ds, target):
    y = ds.target(target)
    mask = y.notna().to_numpy()
    return ds.X.loc[mask], y[mask].to_numpy(), ds.groups[mask]


# --- experiments ------------------------------------------------------------
def run_classification(ds):
    targets = [
        "top_quartile_for_creator",
        "outperformed_creator_median",
        "high_engagement_rate",
        "high_share_rate",
    ]
    print("\n" + "=" * 78)
    print("CLASSIFICATION  |  out-of-fold ROC-AUC (chance=0.5)  +  Precision@20")
    print("=" * 78)
    for target in targets:
        X, y, groups = xy_for(ds, target)
        y = y.astype(int)
        base = y.mean()
        g_splits = group_cv_splits(groups, n_splits=5)
        r_splits = stratified_splits(y, 5)
        print(f"\n[{target}]  n={len(y)}  pos_rate={base:.3f}")
        print(f"  {'model':8s}  {'group_AUC':>9s}  {'rand_AUC':>9s}  "
              f"{'grp_P@20':>9s}  {'rnd_P@20':>9s}")
        for name, factory in clf_models().items():
            g_oof = oof_predict(X, y, g_splits, factory, proba=True)
            r_oof = oof_predict(X, y, r_splits, factory, proba=True)
            g_auc = roc_auc_score(y, g_oof)
            r_auc = roc_auc_score(y, r_oof)
            g_p = precision_at_k(y, g_oof, 20)
            r_p = precision_at_k(y, r_oof, 20)
            print(f"  {name:8s}  {g_auc:9.3f}  {r_auc:9.3f}  {g_p:9.3f}  {r_p:9.3f}")


def run_feature_ablation(ds, target="top_quartile_for_creator"):
    X_all, y, groups = xy_for(ds, target)
    y = y.astype(int)
    g_splits = group_cv_splits(groups, n_splits=5)
    print("\n" + "=" * 78)
    print(f"FEATURE-GROUP ABLATION (RF, GroupKFold)  target={target}")
    print("=" * 78)
    print(f"  {'group':10s}  {'#feats':>6s}  {'group_AUC':>9s}")
    configs = {"ALL": list(X_all.columns)}
    for gname, kws in FEATURE_GROUPS.items():
        configs[gname] = group_features(list(X_all.columns), kws)
    for gname, feats in configs.items():
        if not feats:
            continue
        Xg = X_all[feats]
        oof = oof_predict(Xg, y, g_splits, clf_models()["rf"], proba=True)
        print(f"  {gname:10s}  {len(feats):6d}  {roc_auc_score(y, oof):9.3f}")


def run_regression(ds):
    targets = ["creator_relative_log_views", "engagement_rate", "share_rate"]
    print("\n" + "=" * 78)
    print("REGRESSION  |  out-of-fold R^2 (GroupKFold) and MAE")
    print("=" * 78)
    for target in targets:
        X, y, groups = xy_for(ds, target)
        g_splits = group_cv_splits(groups, n_splits=5)
        print(f"\n[{target}]  n={len(y)}")
        print(f"  {'model':8s}  {'group_R2':>9s}  {'group_MAE':>10s}")
        for name, factory in reg_models().items():
            oof = oof_predict(X, y, g_splits, factory, proba=False)
            print(f"  {name:8s}  {r2_score(y, oof):9.3f}  "
                  f"{mean_absolute_error(y, oof):10.4f}")


def main():
    ds = load_model_dataset()
    print(f"Loaded {len(ds.frame)} rows, {len(ds.feature_names)} features, "
          f"{len(set(ds.groups))} creators.")
    run_classification(ds)
    run_feature_ablation(ds)
    run_regression(ds)


if __name__ == "__main__":
    main()
