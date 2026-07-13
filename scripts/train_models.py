"""CLI: train, evaluate, and export all models (PRD 12 / 13).

    python scripts/train_models.py \
        --dataset data/training_dataset.csv \
        --out backend/models

Runs cross-creator (GroupKFold-by-creator) evaluation for every model spec, then
fits each model on all its available rows and writes the deployable artifacts
(pipelines, feature schema, importances, training metadata).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.config import settings  # noqa: E402
from backend.training.evaluate import evaluate_all  # noqa: E402
from backend.training.export_artifacts import export_models  # noqa: E402
from backend.training.model_dataset import load_model_dataset  # noqa: E402


def _format_metrics(result) -> str:
    if result.message:
        return result.message
    return "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in result.metrics.items())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train + evaluate + export models (PRD 12/13).")
    parser.add_argument("--dataset", default=str(settings.training_dataset_csv))
    parser.add_argument("--out", default=str(settings.models_dir))
    args = parser.parse_args(argv)

    dataset = load_model_dataset(args.dataset)
    print(f"Loaded {len(dataset.frame)} rows, {len(dataset.feature_names)} features.\n")

    print("Cross-creator (GroupKFold) evaluation:")
    evaluations = evaluate_all(dataset)
    for name, result in evaluations.items():
        print(f"  {name:16s} [{result.task:14s} n={result.n_samples:4d} folds={result.n_folds}]  {_format_metrics(result)}")

    print("\nFitting final models and writing artifacts...")
    written = export_models(dataset, out_dir=args.out, evaluations=evaluations)
    for key, path in written.items():
        print(f"  {key:20s} -> {path}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
