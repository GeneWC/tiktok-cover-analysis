"""Exp C3: hook/early-window feature ablation on frozen val split (no test).

Uses existing first_1s / first_3s columns already in training_dataset.csv.

    python scripts/experiments/run_hook_feature_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.baselines import build_rf_control_pipeline  # noqa: E402
from backend.training.creator_splits import load_creator_splits  # noqa: E402
from backend.training.feature_groups import select_group_features  # noqa: E402
from backend.training.heldout_eval import evaluate_estimator_heldout  # noqa: E402
from backend.training.model_dataset import load_model_dataset  # noqa: E402
from backend.training.model_specs import PRIMARY_SPEC  # noqa: E402

OUT_JSON = PROJECT_ROOT / "data" / "reports" / "hook_feature_ablation_val.json"


def _early_features(all_features: list[str]) -> list[str]:
    return [f for f in all_features if "first_1s" in f or "first_3s" in f]


def main() -> int:
    dataset = load_model_dataset()
    membership = load_creator_splits()
    suites = {
        "framing_visual_control": select_group_features(
            dataset.feature_names, PRIMARY_SPEC.feature_groups
        ),
        "early_windows_only": _early_features(dataset.feature_names),
        "framing_visual_plus_early_motion_audio": sorted(
            set(
                select_group_features(dataset.feature_names, ("framing", "visual"))
                + [
                    f
                    for f in dataset.feature_names
                    if f.startswith(("motion_energy_first", "audio_energy_first"))
                ]
            )
        ),
        "all_features": list(dataset.feature_names),
    }

    results = []
    for name, feats in suites.items():
        if not feats:
            continue
        pipe = build_rf_control_pipeline(dataset.X[feats])
        result = evaluate_estimator_heldout(
            name,
            pipe,
            dataset,
            membership,
            eval_split="val",
            feature_names=feats,
        )
        results.append(
            {
                "name": result.name,
                "n_features": len(feats),
                "features": feats,
                "metrics": result.metrics,
                "n_train": result.n_train,
                "n_eval": result.n_eval,
            }
        )
        auc = result.metrics.get("roc_auc", float("nan"))
        print(f"{name:45s} n_feat={len(feats):3d}  val_auc={auc:.4f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
