"""Raw training CSV generation (PRD 8.4 / 20.1).

Runs a collector over the creator seed list and writes the normalized
`raw_tiktok_training_data.csv` snapshot that the rest of the offline pipeline
(feature extraction, label computation, dataset build) reads from.

Kept as an importable function (separate from the CLI wrapper in `scripts/`) so
it can be unit-tested without spawning a subprocess.
"""

from __future__ import annotations

from pathlib import Path

from backend.collectors.base_collector import BaseCollector
from backend.collectors.local_collector import LocalDatasetCollector
from backend.training.creators import load_creators


def build_raw_training_csv(
    creators_csv: str | Path,
    engagement_csv: str | Path,
    downloads_dir: str | Path,
    out_path: str | Path,
    collector: BaseCollector | None = None,
) -> dict[str, object]:
    """Collect videos for every seeded creator and write the raw training CSV.

    `collector` can be injected (e.g. in tests or to swap in a different data
    source); by default a `LocalDatasetCollector` over the given files is used.
    Returns a summary dict.
    """
    creators = load_creators(creators_csv)
    if collector is None:
        collector = LocalDatasetCollector(engagement_csv, downloads_dir)

    videos = collector.collect([c.username for c in creators])
    collector.save_training_csv(videos, out_path)

    matched = sum(1 for v in videos if v.video_file)
    return {
        "creators": len(creators),
        "videos": len(videos),
        "matched_files": matched,
        "missing_files": len(videos) - matched,
        "output": str(out_path),
    }
