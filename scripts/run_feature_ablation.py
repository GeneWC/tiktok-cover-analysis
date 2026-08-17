"""CLI: feature-group ablation on the frozen validation creators.

    python scripts/run_feature_ablation.py
    python scripts/run_feature_ablation.py --json-out data/reports/ablation.json

Never scores the frozen test set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.ablation import ablation_table, run_group_ablation  # noqa: E402
from backend.training.creator_splits import (  # noqa: E402
    DEFAULT_SPLITS_PATH,
    generate_and_save_splits,
    load_creator_splits,
)
from backend.training.model_dataset import load_model_dataset  # noqa: E402
from backend.training.reproducibility import seed_everything  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Feature-group ablation on val.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    seed_everything()
    dataset = load_model_dataset(args.dataset)
    splits_path = Path(args.splits)
    if splits_path.exists():
        membership = load_creator_splits(splits_path)
    else:
        membership = generate_and_save_splits(dataset.groups, path=splits_path)

    results = run_group_ablation(dataset, membership, eval_split="val")
    rows = ablation_table(results)
    print("Feature-group ablation (train fit / val score)")
    for row in rows:
        if row.get("message"):
            print(f"  {row['name']:22s}  {row['message']}")
            continue
        auc = row.get("roc_auc")
        pr = row.get("pr_auc")
        brier = row.get("brier")
        print(
            f"  {row['name']:22s}  n_eval={row['n_eval']:<4}  "
            f"roc_auc={auc:.4f}  pr_auc={pr:.4f}  brier={brier:.4f}"
            if auc is not None
            else f"  {row['name']:22s}  n_eval={row['n_eval']}"
        )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"results": rows}, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
