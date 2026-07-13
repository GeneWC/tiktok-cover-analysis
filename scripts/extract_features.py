"""CLI: extract features for training videos (PRD 20.2).

    python scripts/extract_features.py \
        --input data/raw_tiktok_training_data.csv \
        --out data/video_features.csv

Resumable: re-running skips videos already in the output CSV, so the (multi-hour)
job can be run in chunks. Use --limit to process only the next N videos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.config import settings  # noqa: E402
from backend.training.extract_training_features import (  # noqa: E402
    build_video_features_csv,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract training video features (PRD 20.2).")
    parser.add_argument("--input", default=str(settings.raw_training_csv))
    parser.add_argument("--out", default=str(settings.video_features_csv))
    parser.add_argument(
        "--limit", type=int, default=None, help="process only the next N unprocessed videos"
    )
    args = parser.parse_args(argv)

    summary = build_video_features_csv(
        raw_csv=args.input, out_path=args.out, limit=args.limit
    )

    print("\nFeature extraction summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"  output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
