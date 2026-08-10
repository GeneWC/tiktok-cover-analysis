"""Exp C2: evaluate audio structure features on frozen val (no test).

    python scripts/experiments/run_audio_structure_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.features.audio_features import AUDIO_STRUCTURE_FEATURE_KEYS  # noqa: E402
from backend.training.baselines import (  # noqa: E402
    build_rf_control_pipeline,
    build_simple_logistic_pipeline,
)
from backend.training.creator_splits import load_creator_splits  # noqa: E402
from backend.training.feature_groups import select_group_features  # noqa: E402
from backend.training.heldout_eval import evaluate_estimator_heldout  # noqa: E402
from backend.training.model_dataset import load_model_dataset  # noqa: E402

OUT_JSON = PROJECT_ROOT / "data" / "reports" / "audio_structure_val.json"


def main() -> int:
    dataset = load_model_dataset()
    membership = load_creator_splits()

    structure = [f for f in AUDIO_STRUCTURE_FEATURE_KEYS if f in dataset.feature_names]
    audio = select_group_features(dataset.feature_names, ("audio",))
    framing_visual = select_group_features(dataset.feature_names, ("framing", "visual"))
    fv_audio = select_group_features(
        dataset.feature_names, ("framing", "visual", "audio")
    )

    suites: dict[str, list[str]] = {
        "rf_framing_visual_control": framing_visual,
        "rf_audio_all": audio,
        "rf_structure_only": structure,
        "rf_framing_visual_audio": fv_audio,
        "rf_framing_visual_structure": framing_visual + structure,
        "logistic_structure": structure,
    }

    results = []
    for name, feats in suites.items():
        if not feats:
            print(f"SKIP {name}: no features")
            continue
        if name.startswith("logistic"):
            pipe = build_simple_logistic_pipeline(dataset.X[feats])
        else:
            pipe = build_rf_control_pipeline(dataset.X[feats])
        result = evaluate_estimator_heldout(
            name, pipe, dataset, membership, eval_split="val", feature_names=feats
        )
        auc = result.metrics.get("roc_auc", float("nan"))
        pk = result.metrics.get("precision_at_k", float("nan"))
        print(f"{name:40s} n={len(feats):3d}  auc={auc:.4f}  p@k={pk:.4f}")
        results.append(
            {
                "name": name,
                "n_features": len(feats),
                "features": feats,
                "metrics": result.metrics,
            }
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
