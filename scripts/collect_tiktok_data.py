"""CLI: collect raw TikTok training data (PRD 20.1).

    python scripts/collect_tiktok_data.py \
        --creators data/creators.csv \
        --out data/raw_tiktok_training_data.csv

Defaults come from app settings, so plain `python scripts/collect_tiktok_data.py`
works for this project's local dataset. Adds the project root to sys.path so it
runs from anywhere without PYTHONPATH.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.config import settings  # noqa: E402
from backend.training.collect_dataset import build_raw_training_csv  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect raw TikTok training data (PRD 20.1).")
    parser.add_argument("--creators", default=str(settings.creators_csv))
    parser.add_argument("--engagement", default=str(settings.engagement_csv))
    parser.add_argument("--downloads", default=str(settings.downloads_dir))
    parser.add_argument("--out", default=str(settings.raw_training_csv))
    args = parser.parse_args(argv)

    summary = build_raw_training_csv(
        creators_csv=args.creators,
        engagement_csv=args.engagement,
        downloads_dir=args.downloads,
        out_path=args.out,
    )

    print("Raw training CSV written:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
