"""Exp C5: within-creator ranking / continuous targets (val + train diagnostics).

Axis separate from residual-feature channel mode: can absolute (or within-creator
z-scored) framing+visual features rank videos by creator_relative_log_views?

Promising gate (val only; test untouched unless gate passes):
  mean within-creator Spearman >= 0.25 AND beats brightness baseline by >= 0.05.

    .venv\\Scripts\\python.exe scripts/experiments/run_within_creator_ranking.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.creator_splits import indices_for_split, load_creator_splits  # noqa: E402
from backend.training.feature_groups import select_group_features  # noqa: E402
from backend.training.model_dataset import load_model_dataset  # noqa: E402
from backend.training.model_specs import PRIMARY_SPEC  # noqa: E402
from backend.training.ranking_metrics import (  # noqa: E402
    DEFAULT_MIN_CREATOR_N,
    within_creator_metric,
    within_creator_zscore_features,
)
from backend.training.regressor import build_regressor_pipeline  # noqa: E402

OUT = PROJECT_ROOT / "data" / "reports" / "within_creator_ranking_val.json"
TARGET = "creator_relative_log_views"
BRIGHTNESS = "brightness_mean_full"
SEED = 42
PROMISING_SPEARMAN = 0.25
PROMISING_MARGIN_VS_BRIGHTNESS = 0.05


def _round_metrics(payload: dict) -> dict:
    """Round floats in a nested report for stable JSON."""

    def _r(v):
        if isinstance(v, float):
            return None if not np.isfinite(v) else round(v, 4)
        if isinstance(v, dict):
            return {k: _r(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_r(x) for x in v]
        return v

    return _r(payload)


def _summarize(scores: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    sp = within_creator_metric(y, scores, groups, metric="spearman")
    pw = within_creator_metric(y, scores, groups, metric="pairwise")
    return {
        "mean_within_creator_spearman": sp["mean"],
        "mean_within_creator_pairwise": pw["mean"],
        "n_creators_spearman": sp["n_creators"],
        "n_creators_pairwise": pw["n_creators"],
        "per_creator_spearman": sp["per_creator"],
        "per_creator_pairwise": pw["per_creator"],
    }


def _per_creator_cv_ranking(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 3,
) -> dict:
    """Within each train creator, KFold RF regressor OOF ranking metrics."""
    rows = []
    for creator in sorted(set(groups.astype(str))):
        idx = np.flatnonzero(groups.astype(str) == creator)
        if len(idx) < max(DEFAULT_MIN_CREATOR_N, n_splits + 1):
            continue
        X_c = X.iloc[idx].reset_index(drop=True)
        y_c = y[idx]
        if np.unique(y_c).size < 2:
            continue
        oof = np.full(len(y_c), np.nan)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        pipe0 = build_regressor_pipeline(X_c)
        for tr, te in kf.split(X_c):
            est = clone(pipe0)
            est.fit(X_c.iloc[tr], y_c[tr])
            oof[te] = est.predict(X_c.iloc[te])
        sp = within_creator_metric(y_c, oof, np.array([creator] * len(y_c)))
        pw = within_creator_metric(
            y_c, oof, np.array([creator] * len(y_c)), metric="pairwise"
        )
        rows.append(
            {
                "creator_username": creator,
                "n_videos": int(len(idx)),
                "spearman": sp["mean"],
                "pairwise": pw["mean"],
            }
        )

    spears = [r["spearman"] for r in rows if r["spearman"] is not None]
    pairs = [r["pairwise"] for r in rows if r["pairwise"] is not None]
    return {
        "n_creators": len(rows),
        "mean_within_creator_spearman": float(np.mean(spears)) if spears else None,
        "median_within_creator_spearman": float(np.median(spears)) if spears else None,
        "mean_within_creator_pairwise": float(np.mean(pairs)) if pairs else None,
        "per_creator": rows,
    }


def _fit_predict_global(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
) -> np.ndarray:
    pipe = build_regressor_pipeline(X_train)
    pipe.fit(X_train, y_train)
    return np.asarray(pipe.predict(X_eval), dtype=float)


def main() -> int:
    dataset = load_model_dataset()
    membership = load_creator_splits()
    feats = select_group_features(dataset.feature_names, PRIMARY_SPEC.feature_groups)
    if BRIGHTNESS not in feats and BRIGHTNESS in dataset.feature_names:
        # brightness is in visual group; keep available for baseline even if
        # somehow dropped from PRIMARY_SPEC (should not happen).
        pass

    X_all, y_s, groups = dataset.xy(TARGET)
    y = y_s.to_numpy(dtype=float)
    X_abs = X_all[feats].reset_index(drop=True)
    groups = np.asarray(groups).astype(str)
    # brightness column aligned to filtered rows
    bright = (
        X_all[BRIGHTNESS].to_numpy(dtype=float)
        if BRIGHTNESS in X_all.columns
        else dataset.X.loc[X_all.index, BRIGHTNESS].to_numpy(dtype=float)
    )

    train_idx = indices_for_split(groups, membership, "train")
    val_idx = indices_for_split(groups, membership, "val")

    X_z = within_creator_zscore_features(X_abs, groups)

    # --- Train-creator within-creator CV (diagnostic / train_cv axis) ---
    train_cv_abs = _per_creator_cv_ranking(
        X_abs.iloc[train_idx].reset_index(drop=True),
        y[train_idx],
        groups[train_idx],
    )
    train_cv_z = _per_creator_cv_ranking(
        X_z.iloc[train_idx].reset_index(drop=True),
        y[train_idx],
        groups[train_idx],
    )

    # --- Global regressor → val within-creator ranking ---
    pred_abs = _fit_predict_global(
        X_abs.iloc[train_idx].reset_index(drop=True),
        y[train_idx],
        X_abs.iloc[val_idx].reset_index(drop=True),
    )
    pred_z = _fit_predict_global(
        X_z.iloc[train_idx].reset_index(drop=True),
        y[train_idx],
        X_z.iloc[val_idx].reset_index(drop=True),
    )
    val_abs = _summarize(pred_abs, y[val_idx], groups[val_idx])
    val_z = _summarize(pred_z, y[val_idx], groups[val_idx])

    # --- Baselines on val ---
    rng = np.random.RandomState(SEED)
    val_bright = _summarize(bright[val_idx], y[val_idx], groups[val_idx])
    val_random = _summarize(
        rng.random_sample(len(val_idx)), y[val_idx], groups[val_idx]
    )

    # Best candidate = higher mean within-creator Spearman on val
    candidates = {
        "global_rf_absolute": val_abs["mean_within_creator_spearman"],
        "global_rf_zscored": val_z["mean_within_creator_spearman"],
    }
    best_name = max(
        candidates,
        key=lambda k: -np.inf if candidates[k] is None else candidates[k],
    )
    best_sp = candidates[best_name]
    bright_sp = val_bright["mean_within_creator_spearman"]
    margin = None
    if best_sp is not None and bright_sp is not None:
        margin = float(best_sp - bright_sp)

    promising = (
        best_sp is not None
        and bright_sp is not None
        and best_sp >= PROMISING_SPEARMAN
        and margin >= PROMISING_MARGIN_VS_BRIGHTNESS
    )

    report = {
        "experiment": "C5_within_creator_ranking",
        "target": TARGET,
        "features": "framing+visual (PRIMARY_SPEC)",
        "n_features": len(feats),
        "feature_names": feats,
        "min_creator_n": DEFAULT_MIN_CREATOR_N,
        "promising_criteria": {
            "val_mean_within_creator_spearman_gte": PROMISING_SPEARMAN,
            "margin_vs_brightness_gte": PROMISING_MARGIN_VS_BRIGHTNESS,
        },
        "train_cv_per_creator": {
            "absolute_features": train_cv_abs,
            "zscored_features": train_cv_z,
        },
        "val": {
            "global_rf_absolute": val_abs,
            "global_rf_zscored": val_z,
            "baseline_brightness": val_bright,
            "baseline_random": val_random,
        },
        "selection": {
            "best_candidate": best_name,
            "best_val_mean_spearman": best_sp,
            "brightness_val_mean_spearman": bright_sp,
            "margin_vs_brightness": margin,
            "promising": promising,
            "ran_test": False,
        },
    }
    report = _round_metrics(report)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def _fmt(v):
        return "nan" if v is None else f"{v:.4f}"

    print("=== Exp C5 — Within-creator ranking ===")
    print(
        f"train_cv abs  mean Spearman={_fmt(train_cv_abs['mean_within_creator_spearman'])} "
        f"median={_fmt(train_cv_abs['median_within_creator_spearman'])} "
        f"n_creators={train_cv_abs['n_creators']}"
    )
    print(
        f"train_cv z    mean Spearman={_fmt(train_cv_z['mean_within_creator_spearman'])} "
        f"median={_fmt(train_cv_z['median_within_creator_spearman'])}"
    )
    print(
        f"val RF abs    mean Spearman={_fmt(val_abs['mean_within_creator_spearman'])} "
        f"pairwise={_fmt(val_abs['mean_within_creator_pairwise'])}"
    )
    print(
        f"val RF z      mean Spearman={_fmt(val_z['mean_within_creator_spearman'])} "
        f"pairwise={_fmt(val_z['mean_within_creator_pairwise'])}"
    )
    print(
        f"val bright    mean Spearman={_fmt(val_bright['mean_within_creator_spearman'])}"
    )
    print(
        f"val random    mean Spearman={_fmt(val_random['mean_within_creator_spearman'])}"
    )
    print(
        f"best={best_name} margin_vs_bright={_fmt(margin)} "
        f"PROMISING={promising}"
    )
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
