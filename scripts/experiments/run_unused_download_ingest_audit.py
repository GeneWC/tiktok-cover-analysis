"""Audit unused downloads that have engagement metrics but are outside training.

    python scripts/experiments/run_unused_download_ingest_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "data" / "reports" / "unused_download_ingest_audit.json"


def main() -> int:
    downloads = {p.stem: p for p in (PROJECT_ROOT / "downloads").glob("*.mp4")}
    raw_path = PROJECT_ROOT / "data" / "raw_tiktok_training_data.csv"
    eng_path = PROJECT_ROOT / "engagement.csv"
    train_ids: set[str] = set()
    if raw_path.exists():
        raw = pd.read_csv(raw_path, dtype=str)
        if "video_id" in raw.columns:
            train_ids = set(raw["video_id"].astype(str))

    eng = None
    if eng_path.exists():
        eng = pd.read_csv(eng_path, dtype=str)

    unused_files = sorted(set(downloads) - train_ids)
    with_metrics = []
    without_metrics = []
    if eng is not None and "video_id" in eng.columns:
        eng_ids = set(eng["video_id"].astype(str))
        for vid in unused_files:
            if vid in eng_ids:
                with_metrics.append(vid)
            else:
                without_metrics.append(vid)
    else:
        without_metrics = unused_files

    # Creator breakdown if engagement has creator column
    by_creator = {}
    if eng is not None and with_metrics:
        sub = eng[eng["video_id"].astype(str).isin(with_metrics)]
        creator_col = next(
            (c for c in ("creator_username", "creator", "username") if c in sub.columns),
            None,
        )
        if creator_col:
            by_creator = sub[creator_col].value_counts().to_dict()

    payload = {
        "downloads_mp4": len(downloads),
        "training_video_ids": len(train_ids),
        "unused_downloads": len(unused_files),
        "unused_with_engagement_metrics": len(with_metrics),
        "unused_without_metrics": len(without_metrics),
        "creators_among_unused_with_metrics": by_creator,
        "sample_unused_with_metrics": with_metrics[:20],
        "recommendation": (
            f"Ingest {len(with_metrics)} unused downloads that already have "
            "engagement rows, rebuild labels/features incrementally, re-run "
            "held-out eval — only if creator counts still meet min_videos=5."
            if with_metrics
            else "No unused downloads with engagement.csv rows; cannot expand "
            "labels without new metrics. Keep training set as-is."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2)[:2500])
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
