"""CLI: build the joined training dataset (PRD 20.3).

    python scripts/build_training_dataset.py \
        --raw data/raw_tiktok_training_data.csv \
        --features data/video_features.csv \
        --out data/training_dataset.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.config import settings  # noqa: E402
from backend.training.build_training_dataset import build_training_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the training dataset (PRD 20.3).")
    parser.add_argument("--raw", default=str(settings.raw_training_csv))
    parser.add_argument("--features", default=str(settings.video_features_csv))
    parser.add_argument("--out", default=str(settings.training_dataset_csv))
    args = parser.parse_args(argv)

    summary = build_training_dataset(
        raw_csv=args.raw, features_csv=args.features, out_path=args.out
    )

    print("Training dataset built:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"  output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
