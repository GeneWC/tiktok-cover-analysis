"""CLI: print baseline + control metrics on frozen creator splits.

    python scripts/eval_report.py
    python scripts/eval_report.py --split val
    python scripts/eval_report.py --split test   # only at end of experiment cycle

Does not retrain production artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.creator_splits import (  # noqa: E402
    DEFAULT_SPLITS_PATH,
    generate_and_save_splits,
    load_creator_splits,
)
from backend.training.heldout_eval import run_baseline_suite  # noqa: E402
from backend.training.model_dataset import load_model_dataset  # noqa: E402


def _fmt(metrics: dict) -> str:
    parts = []
    for key in (
        "roc_auc",
        "pr_auc",
        "precision_at_k",
        "f1",
        "balanced_accuracy",
        "brier",
        "ece",
        "within_creator_spearman",
        "within_creator_pairwise",
        "positive_rate",
        "n_positive",
    ):
        if key not in metrics:
            continue
        val = metrics[key]
        if isinstance(val, float):
            parts.append(f"{key}={val:.4f}" if key != "n_positive" else f"{key}={int(val)}")
        else:
            parts.append(f"{key}={val}")
    return "  ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Held-out baseline eval report.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument(
        "--split",
        choices=("val", "test", "both"),
        default="val",
        help="Which held-out split to score (test only at cycle end).",
    )
    parser.add_argument("--train-cv", action="store_true", help="Also print train GroupKFold OOF.")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path for machine-readable results (e.g. data/reports/eval_report.json).",
    )
    args = parser.parse_args(argv)

    dataset = load_model_dataset(args.dataset)
    splits_path = Path(args.splits)
    if splits_path.exists():
        membership = load_creator_splits(splits_path)
        print(f"Loaded splits from {splits_path}")
    else:
        membership = generate_and_save_splits(dataset.groups, path=splits_path)
        print(f"Created new splits at {splits_path}")

    print(
        f"Creators train/val/test: "
        f"{len(membership.train_creators)}/"
        f"{len(membership.val_creators)}/"
        f"{len(membership.test_creators)}  "
        f"videos={membership.n_videos}"
    )

    eval_splits = ["val", "test"] if args.split == "both" else [args.split]
    all_results = []
    for split in eval_splits:
        print(f"\n=== Held-out: {split} ===")
        results = run_baseline_suite(
            dataset,
            membership,
            eval_split=split,
            include_train_cv=(args.train_cv and split == eval_splits[0]),
        )
        for r in results:
            if r.message:
                print(f"  {r.name:28s} [{r.split}]  {r.message}")
            else:
                print(
                    f"  {r.name:28s} [{r.split}]  n_train={r.n_train} n_eval={r.n_eval}  {_fmt(r.metrics)}"
                )
            all_results.append(
                {
                    "name": r.name,
                    "split": r.split,
                    "n_train": r.n_train,
                    "n_eval": r.n_eval,
                    "metrics": r.metrics,
                    "message": r.message,
                }
            )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "splits_path": str(splits_path),
            "train_creators": list(membership.train_creators),
            "val_creators": list(membership.val_creators),
            "test_creators": list(membership.test_creators),
            "n_videos": membership.n_videos,
            "results": all_results,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
