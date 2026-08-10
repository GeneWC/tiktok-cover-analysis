"""Score the Exp C2 promising candidate on train_cv then test (once).

    python scripts/experiments/run_audio_structure_test_once.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.features.audio_features import AUDIO_STRUCTURE_FEATURE_KEYS  # noqa: E402
from backend.training.baselines import build_rf_control_pipeline  # noqa: E402
from backend.training.creator_splits import load_creator_splits  # noqa: E402
from backend.training.feature_groups import select_group_features  # noqa: E402
from backend.training.heldout_eval import (  # noqa: E402
    evaluate_estimator_heldout,
    evaluate_train_group_cv,
)
from backend.training.model_dataset import load_model_dataset  # noqa: E402

OUT = PROJECT_ROOT / "data" / "reports" / "audio_structure_cycle_end.json"


def main() -> int:
    dataset = load_model_dataset()
    membership = load_creator_splits()
    structure = [f for f in AUDIO_STRUCTURE_FEATURE_KEYS if f in dataset.feature_names]
    framing_visual = select_group_features(dataset.feature_names, ("framing", "visual"))
    candidate = framing_visual + structure
    control = framing_visual

    rows = []
    for name, feats in (
        ("rf_control", control),
        ("rf_framing_visual_structure", candidate),
    ):
        pipe = build_rf_control_pipeline(dataset.X[feats])
        cv = evaluate_train_group_cv(
            name, pipe, dataset, membership, feature_names=feats
        )
        val = evaluate_estimator_heldout(
            name, build_rf_control_pipeline(dataset.X[feats]), dataset, membership,
            eval_split="val", feature_names=feats,
        )
        test = evaluate_estimator_heldout(
            name, build_rf_control_pipeline(dataset.X[feats]), dataset, membership,
            eval_split="test", feature_names=feats,
        )
        for split_name, result in (("train_cv", cv), ("val", val), ("test", test)):
            print(
                f"{name:32s} {split_name:8s} "
                f"auc={result.metrics.get('roc_auc', float('nan')):.4f} "
                f"p@k={result.metrics.get('precision_at_k', float('nan')):.4f}"
            )
            rows.append(
                {
                    "name": name,
                    "split": split_name,
                    "n_features": len(feats),
                    "metrics": result.metrics,
                }
            )

    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
