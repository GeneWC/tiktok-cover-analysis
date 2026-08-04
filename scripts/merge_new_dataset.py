"""CLI: merge the spreadsheet video export into the canonical engagement CSV.

    python scripts/merge_new_dataset.py --min-videos 5

Rewrites `engagement.csv` from `data/new_video_data.xlsx`, keeping only videos
whose file exists in `downloads/` and creators with enough videos for
creator-relative labels to be meaningful. The pre-merge CSV is backed up once to
`engagement.pre_merge.csv`.

Re-run `scripts/collect_tiktok_data.py` afterwards to refresh the raw training
CSV, then `scripts/extract_features.py` (resumable) for the new videos.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.config import settings  # noqa: E402
from backend.training.merge_spreadsheet_dataset import (  # noqa: E402
    build_engagement_rows,
    write_engagement_csv,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge the spreadsheet video export into engagement.csv."
    )
    parser.add_argument("--xlsx", default=str(settings.data_dir / "new_video_data.xlsx"))
    parser.add_argument("--downloads", default=str(settings.downloads_dir))
    parser.add_argument("--out", default=str(settings.engagement_csv))
    parser.add_argument(
        "--min-videos",
        type=int,
        default=5,
        help="minimum downloaded videos a creator needs to be included (default: 5)",
    )
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    rows, summary = build_engagement_rows(
        xlsx_path=args.xlsx,
        downloads_dir=args.downloads,
        previous_engagement=out_path,
        min_videos_per_creator=args.min_videos,
    )

    backup = out_path.with_suffix(".pre_merge.csv")
    if out_path.exists() and not backup.exists():
        shutil.copy2(out_path, backup)
        print(f"Backed up previous engagement CSV -> {backup}")

    write_engagement_csv(rows, out_path)

    print("Merged engagement CSV written:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"  output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
