"""Exp C4: label-variant screen on train_cv + val (no test unless promising).

Variants of creator-relative success; primary remains top_quartile_for_creator.

    python scripts/experiments/run_label_variants.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.baselines import build_rf_control_pipeline  # noqa: E402
from backend.training.creator_splits import indices_for_split, load_creator_splits  # noqa: E402
from backend.training.evaluate import precision_at_k  # noqa: E402
from backend.training.feature_groups import select_group_features  # noqa: E402
from backend.training.model_dataset import load_model_dataset  # noqa: E402
from backend.training.model_specs import PRIMARY_SPEC  # noqa: E402
from backend.training.validation import group_cv_splits, plan_group_cv  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

OUT = PROJECT_ROOT / "data" / "reports" / "label_variants_val.json"


def _metrics(y, scores):
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    out = {
        "positive_rate": float(y.mean()),
        "n_positive": n_pos,
        "precision_at_k": precision_at_k(y, scores, k=max(n_pos, 1)),
    }
    if len(np.unique(y)) < 2:
        out["roc_auc"] = float("nan")
    else:
        out["roc_auc"] = float(roc_auc_score(y, scores))
    return out


def _make_labels(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build alternate binary labels aligned to frame rows (no leakage into X)."""
    # Existing primary
    tq = frame["top_quartile_for_creator"].astype(str).str.lower().eq("true").to_numpy()

    # Top tercile within creator on log_views
    log_views = pd.to_numeric(frame["log_views"], errors="coerce")
    top_tercile = np.zeros(len(frame), dtype=int)
    for _, idx in frame.groupby("creator_username").groups.items():
        vals = log_views.loc[idx]
        thr = vals.quantile(2 / 3)
        top_tercile[list(idx)] = (vals >= thr).astype(int).to_numpy()

    # Relative z > 0.5 (robust z already in labels)
    z = pd.to_numeric(frame["creator_relative_z"], errors="coerce")
    z_pos = (z > 0.5).fillna(False).astype(int).to_numpy()

    # Outperformed median (already a column)
    outperformed = (
        frame["outperformed_creator_median"].astype(str).str.lower().eq("true").to_numpy()
    )

    return {
        "top_quartile_for_creator": tq.astype(int),
        "top_tercile_for_creator": top_tercile,
        "relative_z_gt_0_5": z_pos,
        "outperformed_creator_median": outperformed.astype(int),
    }


def _oof_or_heldout(X, y, groups, membership, mode: str):
    if mode == "train_cv":
        train_idx = indices_for_split(groups, membership, "train")
        X_t = X.iloc[train_idx].reset_index(drop=True)
        y_t = y[train_idx]
        g_t = groups[train_idx]
        plan = plan_group_cv(g_t, 5)
        splits = group_cv_splits(g_t, 5)
        oof = np.full(len(y_t), np.nan)
        pipe0 = build_rf_control_pipeline(X_t)
        for tr, te in splits:
            est = clone(pipe0)
            est.fit(X_t.iloc[tr], y_t[tr])
            oof[te] = est.predict_proba(X_t.iloc[te])[:, 1]
        return _metrics(y_t, oof), len(y_t), plan.n_splits

    # val heldout
    train_idx = indices_for_split(groups, membership, "train")
    val_idx = indices_for_split(groups, membership, "val")
    pipe = build_rf_control_pipeline(X.iloc[train_idx])
    pipe.fit(X.iloc[train_idx], y[train_idx])
    scores = pipe.predict_proba(X.iloc[val_idx])[:, 1]
    return _metrics(y[val_idx], scores), len(val_idx), 0


def main() -> int:
    dataset = load_model_dataset()
    membership = load_creator_splits()
    feats = select_group_features(dataset.feature_names, PRIMARY_SPEC.feature_groups)
    X = dataset.X[feats]
    groups = dataset.groups
    labels = _make_labels(dataset.frame)

    rows = []
    for label_name, y in labels.items():
        for mode in ("train_cv", "val"):
            metrics, n, folds = _oof_or_heldout(X, y, groups, membership, mode)
            print(
                f"{label_name:32s} {mode:8s} n={n:4d} "
                f"auc={metrics['roc_auc']:.4f} pos={metrics['positive_rate']:.3f}"
            )
            rows.append(
                {
                    "label": label_name,
                    "split": mode,
                    "n": n,
                    "n_folds": folds,
                    "metrics": metrics,
                }
            )

    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
