"""Phase B — diagnose why top_quartile_for_creator signal is weak (~0.53 AUC).

Leak-safe only: uses backend.training.model_dataset.select_feature_columns /
NON_FEATURE_COLUMNS. Engagement metrics, views, creator identity, and labels
are never used as model inputs.

Run (Windows):
  .venv\\Scripts\\python.exe scripts\\experiments\\phase_b_signal_diagnosis.py

Writes CSVs under data/reports/ and data/reports/phase_b_diagnosis_summary.json.
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.feature_groups import (  # noqa: E402
    GROUP_NAMES,
    assign_group,
    select_group_features,
)
from backend.training.model_dataset import (  # noqa: E402
    PRIMARY_TARGET,
    load_model_dataset,
)
from backend.training.validation import group_cv_splits  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

REPORTS = PROJECT_ROOT / "data" / "reports"
DATASET = PROJECT_ROOT / "data" / "training_dataset.csv"
TARGET = PRIMARY_TARGET
MIN_CREATOR_N = 20
NEAR_ZERO_VAR = 1e-8
SEED = 42

# Production-quality features for correlation analysis (presentation-ish).
PROD_QUALITY = [
    "brightness_mean_full",
    "brightness_mean_first_3s",
    "sharpness_full",
    "sharpness_first_3s",
    "contrast_full",
    "contrast_first_3s",
    "audio_rms_mean",
    "audio_dynamic_range",
    "audio_silence_ratio",
    "audio_clipping_ratio",
    "audio_onset_strength_mean",
    "audio_energy_first_3s",
    "subject_centering_score",
    "subject_size_ratio",
    "face_size_ratio",
    "person_visible_ratio",
    "face_visible_ratio",
]

# Hook-like early-window features (first 1s/3s).
HOOK_PREFIXES = (
    "brightness_mean_first_",
    "contrast_first_",
    "sharpness_first_",
    "blur_first_",
    "colorfulness_first_",
    "person_visible_ratio_first_",
    "face_visible_ratio_first_",
    "hand_visible_ratio_first_",
    "motion_energy_first_",
    "hand_motion_energy_first_",
    "audio_energy_first_",
    "text_present_first_",
    "text_area_ratio_first_",
    "first_text_timestamp",
)


def _safe_auc(y: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores) & np.isfinite(y.astype(float))
    if mask.sum() < 5:
        return None
    y_m, s_m = y[mask], scores[mask]
    if len(np.unique(y_m)) < 2:
        return None
    # Flip if needed is handled by roc_auc itself for ranking; use raw scores.
    try:
        return float(roc_auc_score(y_m, s_m))
    except ValueError:
        return None


def _univariate_auc(x: pd.Series, y: pd.Series) -> float | None:
    """ROC AUC treating the feature as a score; take max(auc, 1-auc)."""
    auc = _safe_auc(y.to_numpy(), x.to_numpy(dtype=float))
    if auc is None:
        return None
    return float(max(auc, 1.0 - auc))


def analysis_1_label_noise(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for creator, g in frame.groupby("creator_username", sort=True):
        y = g[TARGET].astype(float)
        n = int(len(g))
        n_pos = int(y.sum())
        pos_rate = float(y.mean()) if n else None
        # Expected ~0.25 by construction; deviation grows when n is small.
        expected = 0.25
        noise_flag = bool(n < MIN_CREATOR_N or abs(pos_rate - expected) > 0.08)
        rows.append(
            {
                "creator_username": creator,
                "n_videos": n,
                "n_pos": n_pos,
                "pos_rate": round(pos_rate, 4) if pos_rate is not None else None,
                "abs_dev_from_025": round(abs(pos_rate - expected), 4),
                "small_or_noisy": noise_flag,
            }
        )
    out = pd.DataFrame(rows).sort_values(["n_videos", "creator_username"])
    out.to_csv(REPORTS / "phase_b_label_balance_by_creator.csv", index=False)
    return out


def analysis_2_missing_variance(X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in X.columns:
        s = X[col]
        n = len(s)
        miss = float(s.isna().mean())
        s_ok = s.dropna()
        var = float(s_ok.var()) if len(s_ok) > 1 else 0.0
        nunique = int(s_ok.nunique())
        rows.append(
            {
                "feature": col,
                "group": assign_group(col) or "unmapped",
                "missing_rate": round(miss, 4),
                "n_non_null": int(s.notna().sum()),
                "variance": var,
                "nunique": nunique,
                "near_zero_variance": bool(var < NEAR_ZERO_VAR or nunique <= 1),
                "high_missing": bool(miss >= 0.2),
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["high_missing", "near_zero_variance", "missing_rate"],
        ascending=[False, False, False],
    )
    out.to_csv(REPORTS / "phase_b_feature_missingness_variance.csv", index=False)
    return out


def analysis_3_univariate(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    # Impute median for MI; AUC uses available values only.
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)
    y_arr = y.astype(int).to_numpy()
    mi = mutual_info_classif(X_imp, y_arr, discrete_features=False, random_state=SEED)

    rows = []
    for i, col in enumerate(X.columns):
        auc = _univariate_auc(X[col], y)
        rows.append(
            {
                "feature": col,
                "group": assign_group(col) or "unmapped",
                "roc_auc_abs": None if auc is None else round(auc, 4),
                "mutual_info": round(float(mi[i]), 6),
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["roc_auc_abs", "mutual_info"], ascending=[False, False], na_position="last"
    )
    out.to_csv(REPORTS / "phase_b_univariate_auc_mi.csv", index=False)
    return out


def analysis_4_prod_corr(frame: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    rel = pd.to_numeric(frame["creator_relative_log_views"], errors="coerce")
    y = frame[TARGET].astype(float)
    rows = []
    for col in PROD_QUALITY:
        if col not in X.columns:
            continue
        x = X[col]
        # Spearman via rank corr (robust to scale); pearson as secondary.
        df = pd.DataFrame({"x": x, "rel": rel, "y": y}).dropna()
        if len(df) < 10:
            continue
        rows.append(
            {
                "feature": col,
                "group": assign_group(col) or "unmapped",
                "n": int(len(df)),
                "spearman_vs_creator_relative_log_views": round(
                    float(df["x"].corr(df["rel"], method="spearman")), 4
                ),
                "pearson_vs_creator_relative_log_views": round(
                    float(df["x"].corr(df["rel"], method="pearson")), 4
                ),
                "pointbiserial_vs_top_quartile": round(
                    float(df["x"].corr(df["y"], method="pearson")), 4
                ),
                "univariate_auc_vs_top_quartile": _univariate_auc(df["x"], df["y"]),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["univariate_auc_vs_top_quartile"] = out[
            "univariate_auc_vs_top_quartile"
        ].apply(lambda v: None if v is None else round(float(v), 4))
        out = out.sort_values(
            "univariate_auc_vs_top_quartile", ascending=False, na_position="last"
        )
    out.to_csv(REPORTS / "phase_b_prod_quality_correlations.csv", index=False)
    return out


def _within_creator_auc(X: pd.DataFrame, y: pd.Series, creators: pd.Series) -> pd.DataFrame:
    """Per-creator univariate mean AUC across features + cheap RF AUC."""
    rows = []
    for creator, idx in creators.groupby(creators).groups.items():
        idx = list(idx)
        if len(idx) < MIN_CREATOR_N:
            continue
        y_c = y.loc[idx]
        if y_c.nunique() < 2:
            continue
        X_c = X.loc[idx]
        # Mean of top-10 univariate AUCs as a cheap within-creator signal proxy.
        u_aucs = []
        for col in X_c.columns:
            a = _univariate_auc(X_c[col], y_c)
            if a is not None:
                u_aucs.append(a)
        u_aucs = sorted(u_aucs, reverse=True)
        mean_top10 = float(np.mean(u_aucs[:10])) if u_aucs else None

        # Cheap RF on all leak-safe features (within creator — not for gen. claim).
        pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(
                    n_estimators=100,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=-1,
                )),
            ]
        )
        # Simple 3-fold random CV within creator (stratified if possible).
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        oof = np.full(len(y_c), np.nan)
        y_arr = y_c.to_numpy().astype(int)
        X_arr = X_c.reset_index(drop=True)
        try:
            for tr, te in skf.split(X_arr, y_arr):
                pipe.fit(X_arr.iloc[tr], y_arr[tr])
                oof[te] = pipe.predict_proba(X_arr.iloc[te])[:, 1]
            rf_auc = _safe_auc(y_arr, oof)
        except ValueError:
            rf_auc = None

        rows.append(
            {
                "creator_username": creator,
                "n_videos": int(len(idx)),
                "pos_rate": round(float(y_c.mean()), 4),
                "mean_top10_univariate_auc": None
                if mean_top10 is None
                else round(mean_top10, 4),
                "within_creator_rf_auc": None if rf_auc is None else round(rf_auc, 4),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("within_creator_rf_auc", ascending=False, na_position="last")
    out.to_csv(REPORTS / "phase_b_within_creator_signal.csv", index=False)
    return out


def _leave_one_creator_subset_auc(
    X: pd.DataFrame, y: pd.Series, groups: np.ndarray, feature_subset: list[str]
) -> dict:
    """OOF GroupKFold AUC on a feature subset (5-fold by creator)."""
    cols = [c for c in feature_subset if c in X.columns]
    if not cols:
        return {"n_features": 0, "oof_auc": None, "n_splits": 0}
    Xs = X[cols]
    y_arr = y.astype(int).to_numpy()
    splits = group_cv_splits(groups, n_splits=5)
    oof = np.full(len(y_arr), np.nan)
    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    for tr, te in splits:
        pipe.fit(Xs.iloc[tr], y_arr[tr])
        oof[te] = pipe.predict_proba(Xs.iloc[te])[:, 1]
    return {
        "n_features": len(cols),
        "oof_auc": _safe_auc(y_arr, oof),
        "n_splits": len(splits),
    }


def analysis_5_heterogeneity(
    X: pd.DataFrame, y: pd.Series, groups: np.ndarray, creators: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    within = _within_creator_auc(X, y, creators)

    # Per-creator held-out: train on all others (LOCO), score that creator.
    # Cheap: framing+visual only (production primary), RF.
    fv = select_group_features(list(X.columns), ("framing", "visual"))
    loco_rows = []
    uniq = sorted(set(groups.tolist()))
    y_arr = y.astype(int).to_numpy()
    X_fv = X[fv].reset_index(drop=True)
    for held in uniq:
        te_mask = groups == held
        tr_mask = ~te_mask
        if int(te_mask.sum()) < 8 or int(tr_mask.sum()) < 40:
            continue
        y_te = y_arr[te_mask]
        if len(np.unique(y_te)) < 2:
            continue
        pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=150,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipe.fit(X_fv.iloc[tr_mask], y_arr[tr_mask])
        proba = pipe.predict_proba(X_fv.iloc[te_mask])[:, 1]
        auc = _safe_auc(y_te, proba)
        loco_rows.append(
            {
                "held_out_creator": held,
                "n_test": int(te_mask.sum()),
                "pos_rate_test": round(float(y_te.mean()), 4),
                "loco_framing_visual_auc": None if auc is None else round(auc, 4),
            }
        )
    loco = pd.DataFrame(loco_rows)
    if not loco.empty:
        loco = loco.sort_values(
            "loco_framing_visual_auc", ascending=False, na_position="last"
        )
    loco.to_csv(REPORTS / "phase_b_loco_by_creator.csv", index=False)
    return within, loco


def analysis_6_group_signal(
    X: pd.DataFrame, y: pd.Series, groups: np.ndarray, uni: pd.DataFrame
) -> pd.DataFrame:
    feature_sets: dict[str, list[str]] = {
        "all": list(X.columns),
        "framing+visual": select_group_features(list(X.columns), ("framing", "visual")),
        "audio": select_group_features(list(X.columns), ("audio",)),
        "motion": select_group_features(list(X.columns), ("motion",)),
        "framing": select_group_features(list(X.columns), ("framing",)),
        "visual": select_group_features(list(X.columns), ("visual",)),
        "text": select_group_features(list(X.columns), ("text",)),
        "metadata": select_group_features(list(X.columns), ("metadata",)),
        "hook_early": [
            c for c in X.columns if c.startswith(HOOK_PREFIXES) or c == "first_text_timestamp"
        ],
    }

    rows = []
    for name, cols in feature_sets.items():
        sub = uni[uni["feature"].isin(cols)].copy()
        mean_auc = float(sub["roc_auc_abs"].dropna().mean()) if len(sub) else None
        max_auc = float(sub["roc_auc_abs"].dropna().max()) if len(sub) else None
        mean_mi = float(sub["mutual_info"].dropna().mean()) if len(sub) else None
        cv = _leave_one_creator_subset_auc(X, y, groups, cols)
        rows.append(
            {
                "feature_set": name,
                "n_features": len(cols),
                "mean_univariate_auc": None if mean_auc is None else round(mean_auc, 4),
                "max_univariate_auc": None if max_auc is None else round(max_auc, 4),
                "mean_mutual_info": None if mean_mi is None else round(mean_mi, 6),
                "groupkfold_oof_auc": None
                if cv["oof_auc"] is None
                else round(float(cv["oof_auc"]), 4),
            }
        )
    out = pd.DataFrame(rows).sort_values(
        "groupkfold_oof_auc", ascending=False, na_position="last"
    )
    out.to_csv(REPORTS / "phase_b_feature_set_comparison.csv", index=False)
    return out


def build_summary(
    frame: pd.DataFrame,
    X: pd.DataFrame,
    label_bal: pd.DataFrame,
    miss: pd.DataFrame,
    uni: pd.DataFrame,
    prod: pd.DataFrame,
    within: pd.DataFrame,
    loco: pd.DataFrame,
    groups_cmp: pd.DataFrame,
) -> dict:
    y = frame[TARGET].astype(float)
    top = uni.head(10)[["feature", "group", "roc_auc_abs", "mutual_info"]].to_dict(
        "records"
    )
    bottom = (
        uni.dropna(subset=["roc_auc_abs"])
        .tail(10)[["feature", "group", "roc_auc_abs", "mutual_info"]]
        .to_dict("records")
    )
    nz = miss[miss["near_zero_variance"]]["feature"].tolist()
    hi_miss = miss[miss["high_missing"]][["feature", "missing_rate"]].to_dict("records")

    fv_row = groups_cmp[groups_cmp["feature_set"] == "framing+visual"]
    all_row = groups_cmp[groups_cmp["feature_set"] == "all"]
    audio_row = groups_cmp[groups_cmp["feature_set"] == "audio"]
    hook_row = groups_cmp[groups_cmp["feature_set"] == "hook_early"]

    within_aucs = within["within_creator_rf_auc"].dropna()
    loco_aucs = loco["loco_framing_visual_auc"].dropna()

    summary = {
        "date": str(date.today()),
        "dataset": str(DATASET.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "n_rows": int(len(frame)),
        "n_creators": int(frame["creator_username"].nunique()),
        "n_leak_safe_features": int(X.shape[1]),
        "target": TARGET,
        "global_pos_rate": round(float(y.mean()), 4),
        "label_balance": {
            "creators_lt_min_n": int((label_bal["n_videos"] < MIN_CREATOR_N).sum()),
            "min_creator_n": int(label_bal["n_videos"].min()),
            "max_creator_n": int(label_bal["n_videos"].max()),
            "pos_rate_min": float(label_bal["pos_rate"].min()),
            "pos_rate_max": float(label_bal["pos_rate"].max()),
            "pos_rate_std": round(float(label_bal["pos_rate"].std()), 4),
            "small_or_noisy_creators": int(label_bal["small_or_noisy"].sum()),
        },
        "missingness": {
            "n_near_zero_variance": len(nz),
            "near_zero_variance_features": nz,
            "n_high_missing_ge_20pct": len(hi_miss),
            "high_missing_features": hi_miss,
        },
        "univariate_top10": top,
        "univariate_bottom10": bottom,
        "univariate_median_auc": round(float(uni["roc_auc_abs"].median()), 4),
        "univariate_max_auc": round(float(uni["roc_auc_abs"].max()), 4),
        "prod_quality_top_by_auc": prod.head(5).to_dict("records") if len(prod) else [],
        "within_creator": {
            "n_creators_analyzed": int(len(within)),
            "median_within_rf_auc": None
            if within_aucs.empty
            else round(float(within_aucs.median()), 4),
            "mean_within_rf_auc": None
            if within_aucs.empty
            else round(float(within_aucs.mean()), 4),
            "n_creators_auc_ge_060": int((within_aucs >= 0.60).sum())
            if not within_aucs.empty
            else 0,
        },
        "loco_framing_visual": {
            "n_creators": int(len(loco)),
            "median_auc": None
            if loco_aucs.empty
            else round(float(loco_aucs.median()), 4),
            "mean_auc": None if loco_aucs.empty else round(float(loco_aucs.mean()), 4),
            "frac_auc_le_055": None
            if loco_aucs.empty
            else round(float((loco_aucs <= 0.55).mean()), 4),
        },
        "feature_set_groupkfold_oof_auc": {
            r["feature_set"]: r["groupkfold_oof_auc"]
            for _, r in groups_cmp.iterrows()
        },
        "primary_framing_visual_oof_auc": None
        if fv_row.empty
        else fv_row.iloc[0]["groupkfold_oof_auc"],
        "all_features_oof_auc": None
        if all_row.empty
        else all_row.iloc[0]["groupkfold_oof_auc"],
        "audio_oof_auc": None
        if audio_row.empty
        else audio_row.iloc[0]["groupkfold_oof_auc"],
        "hook_early_oof_auc": None
        if hook_row.empty
        else hook_row.iloc[0]["groupkfold_oof_auc"],
        "feature_groups_present": list(GROUP_NAMES),
    }
    path = REPORTS / "phase_b_diagnosis_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ds = load_model_dataset(DATASET)
    frame = ds.frame.copy()
    # Coerce label / relative views for analysis (targets only).
    frame[TARGET] = ds.target(TARGET)
    frame["creator_relative_log_views"] = pd.to_numeric(
        frame["creator_relative_log_views"], errors="coerce"
    )
    X = ds.X
    y = frame[TARGET]
    groups = ds.groups
    creators = frame["creator_username"]

    print(f"Loaded {len(frame)} rows, {X.shape[1]} leak-safe features, "
          f"{creators.nunique()} creators")

    print("1) Label balance / noise by creator...")
    label_bal = analysis_1_label_noise(frame)

    print("2) Missingness & near-zero variance...")
    miss = analysis_2_missing_variance(X)

    print("3) Univariate AUC + MI...")
    uni = analysis_3_univariate(X, y)

    print("4) Production-quality correlations...")
    prod = analysis_4_prod_corr(frame, X)

    print("5) Creator heterogeneity (within + LOCO)...")
    within, loco = analysis_5_heterogeneity(X, y, groups, creators)

    print("6) Feature-set comparison (univariate + GroupKFold OOF)...")
    groups_cmp = analysis_6_group_signal(X, y, groups, uni)

    summary = build_summary(
        frame, X, label_bal, miss, uni, prod, within, loco, groups_cmp
    )

    print("\n=== KEY NUMBERS ===")
    print(f"global_pos_rate={summary['global_pos_rate']}")
    print(f"univariate_max_auc={summary['univariate_max_auc']} "
          f"median={summary['univariate_median_auc']}")
    print(f"framing+visual OOF AUC={summary['primary_framing_visual_oof_auc']}")
    print(f"all features OOF AUC={summary['all_features_oof_auc']}")
    print(f"audio OOF AUC={summary['audio_oof_auc']}")
    print(f"hook_early OOF AUC={summary['hook_early_oof_auc']}")
    print(f"within-creator median RF AUC={summary['within_creator']['median_within_rf_auc']}")
    print(f"LOCO median AUC={summary['loco_framing_visual']['median_auc']}")
    print(f"Wrote summary -> {REPORTS / 'phase_b_diagnosis_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
