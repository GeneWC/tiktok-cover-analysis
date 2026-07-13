"""Local-dataset collector (PRD 8.3 "manual seed/mock data mode").

A concrete `BaseCollector` for the dataset shipped with this project: an
`engagement.csv` of public metrics plus a `downloads/` folder of `{video_id}.mp4`
files. It reads the CSV, matches each row to its video file, and emits normalized
`RawTikTokVideo` records - no network access, fully compliant with PRD 8.3.

This is the adapter the training pipeline uses in place of a live scraper. A real
API/scraping collector could be dropped in later behind the same interface
without touching anything downstream.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from backend.collectors.base_collector import (
    BaseCollector,
    RawTikTokVideo,
    parse_count,
)
from backend.core.config import settings

# Columns the engagement CSV must provide.
_REQUIRED_COLUMNS = (
    "creator",
    "video_id",
    "url",
    "likes",
    "comments",
    "favorites",
    "views",
    "shares",
)


class LocalDatasetCollector(BaseCollector):
    """Collector backed by a local engagement CSV + downloads folder."""

    def __init__(
        self,
        engagement_csv: str | Path | None = None,
        downloads_dir: str | Path | None = None,
    ):
        self.engagement_csv = Path(engagement_csv or settings.engagement_csv)
        self.downloads_dir = Path(downloads_dir or settings.downloads_dir)
        self._rows: list[dict[str, str]] | None = None

    def _load_rows(self) -> list[dict[str, str]]:
        """Read and validate the engagement CSV once, then cache it."""
        if self._rows is not None:
            return self._rows
        if not self.engagement_csv.exists():
            raise FileNotFoundError(f"Engagement CSV not found: {self.engagement_csv}")

        with self.engagement_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            missing = [c for c in _REQUIRED_COLUMNS if c not in fieldnames]
            if missing:
                raise ValueError(
                    f"{self.engagement_csv} missing column(s): {', '.join(missing)}"
                )
            self._rows = [row for row in reader if (row.get("video_id") or "").strip()]
        return self._rows

    def _video_file_for(self, video_id: str) -> str:
        """Relative path to the video if it exists in downloads/, else ''."""
        path = self.downloads_dir / f"{video_id}.mp4"
        # Stored relative to the project root so the raw CSV stays portable.
        return f"{self.downloads_dir.name}/{video_id}.mp4" if path.exists() else ""

    def _row_to_video(self, row: dict[str, str]) -> RawTikTokVideo:
        video_id = (row.get("video_id") or "").strip()
        return RawTikTokVideo(
            video_id=video_id,
            creator_username=(row.get("creator") or "").strip(),
            video_url=(row.get("url") or "").strip(),
            video_file=self._video_file_for(video_id),
            views=parse_count(row.get("views")),
            likes=parse_count(row.get("likes")),
            comments=parse_count(row.get("comments")),
            shares=parse_count(row.get("shares")),
            favorites=parse_count(row.get("favorites")),
        )

    def collect_creator_videos(self, creator_username: str) -> list[RawTikTokVideo]:
        return [
            self._row_to_video(row)
            for row in self._load_rows()
            if (row.get("creator") or "").strip() == creator_username
        ]

    def collect_video_metrics(self, video_url: str) -> RawTikTokVideo:
        for row in self._load_rows():
            if (row.get("url") or "").strip() == video_url:
                return self._row_to_video(row)
        raise KeyError(f"No video found for url: {video_url}")

    def collect_all(self) -> list[RawTikTokVideo]:
        """Collect every video in the engagement CSV (ignores the seed list)."""
        return [self._row_to_video(row) for row in self._load_rows()]

    def match_summary(self) -> dict[str, int]:
        """Report totals + how many rows matched a downloaded file (sanity)."""
        videos = self.collect_all()
        matched = sum(1 for v in videos if v.video_file)
        return {
            "total": len(videos),
            "matched_files": matched,
            "missing_files": len(videos) - matched,
            "creators": len(Counter(v.creator_username for v in videos)),
        }
