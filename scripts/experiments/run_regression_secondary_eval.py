"""Secondary: engagement/shareability Spearman with audio structure on val.

    python scripts/experiments/run_regression_secondary_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.features.audio_features import AUDIO_STRUCTURE_FEATURE_KEYS  # noqa: E402
from backend.training.creator_splits import indices_for_split, load_creator_splits  # noqa: E402
from backend.training.feature_groups import select_group_features  # noqa: E402
from backend.training.model_dataset import load_model_dataset  # noqa: E402
from backend.training.regressor import build_regressor_pipeline  # noqa: E402

OUT = PROJECT_ROOT / "data" / "reports" / "regression_secondary_val.json"


def _spearman(y, pred) -> float:
    return float(pd.Series(y).corr(pd.Series(pred), method="spearman") or 0.0)


def eval_reg(name, X, y, groups, membership, feats):
    mask = pd.Series(y).notna().to_numpy()
    X = X.loc[mask, feats].reset_index(drop=True)
    y = np.asarray(y)[mask]
    groups = groups[mask]
    tr = indices_for_split(groups, membership, "train")
    va = indices_for_split(groups, membership, "val")
    pipe = build_regressor_pipeline(X.iloc[tr])
    pipe.fit(X.iloc[tr], y[tr])
    pred = pipe.predict(X.iloc[va])
    # baselines: predict train mean; predict random rank
    mean_pred = np.full(len(va), float(np.mean(y[tr])))
    return {
        "name": name,
        "n_features": len(feats),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "spearman": _spearman(y[va], pred),
        "spearman_mean_baseline": _spearman(y[va], mean_pred),
    }


def main() -> int:
    dataset = load_model_dataset()
    membership = load_creator_splits()
    structure = [f for f in AUDIO_STRUCTURE_FEATURE_KEYS if f in dataset.feature_names]
    suites = {
        "engagement_framing_visual_audio": (
            "engagement_rate",
            select_group_features(dataset.feature_names, ("framing", "visual", "audio")),
        ),
        "engagement_plus_structure_same": (
            "engagement_rate",
            select_group_features(dataset.feature_names, ("framing", "visual", "audio")),
        ),
        "engagement_structure_only": ("engagement_rate", structure),
        "share_all_features": (
            "share_rate",
            list(dataset.feature_names),
        ),
        "share_structure_only": ("share_rate", structure),
        "creator_rel_all": (
            "creator_relative_log_views",
            list(dataset.feature_names),
        ),
        "creator_rel_structure": (
            "creator_relative_log_views",
            structure,
        ),
    }

    rows = []
    for name, (target, feats) in suites.items():
        if not feats:
            continue
        y = dataset.target(target)
        row = eval_reg(name, dataset.X, y, dataset.groups, membership, feats)
        row["target"] = target
        print(
            f"{name:40s} spearman={row['spearman']:.4f} "
            f"mean_base={row['spearman_mean_baseline']:.4f}"
        )
        rows.append(row)

    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
