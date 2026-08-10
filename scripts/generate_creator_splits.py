"""CLI: generate durable creator train/val/test splits (D-001).

    python scripts/generate_creator_splits.py
    python scripts/generate_creator_splits.py --dataset data/training_dataset.csv --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.creator_splits import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_SPLITS_PATH,
    generate_and_save_splits,
)
from backend.training.model_dataset import load_model_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate creator-grouped splits.")
    parser.add_argument("--dataset", default=None, help="Path to training_dataset.csv")
    parser.add_argument("--out", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    dataset = load_model_dataset(args.dataset)
    membership = generate_and_save_splits(dataset.groups, path=args.out, seed=args.seed)

    print(f"Wrote {args.out}")
    print(
        f"  train: {len(membership.train_creators)} creators, "
        f"{(membership.n_videos or {}).get('train', '?')} videos"
    )
    print(
        f"  val:   {len(membership.val_creators)} creators, "
        f"{(membership.n_videos or {}).get('val', '?')} videos"
    )
    print(
        f"  test:  {len(membership.test_creators)} creators, "
        f"{(membership.n_videos or {}).get('test', '?')} videos"
    )
    print("  train creators:", ", ".join(membership.train_creators))
    print("  val creators:  ", ", ".join(membership.val_creators))
    print("  test creators: ", ", ".join(membership.test_creators))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
