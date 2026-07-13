"""Data-collector adapter interface (PRD 8.3).

The collection layer is isolated behind an adapter so the rest of the pipeline
never depends on *how* data was obtained (live API, permitted public data, or a
local/manual dataset). Concrete collectors implement two methods; CSV writing is
a shared concrete helper (template-method style) so every collector emits the
same raw schema (PRD 8.4).

Compliance (PRD 8.3): collectors must not bypass login/CAPTCHA, collect private
videos, or scrape unnecessary personal data. This MVP uses a local-dataset
collector (see `local_collector.py`), which is fully compliant.

`RawTikTokVideo` is the normalized record shared by all collectors. Metric
fields are `int | None` because public counts are sometimes hidden/missing;
missing is represented as None and resolved later per the PRD 8.5 rules.
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path

# Raw training CSV schema (PRD 8.4: required + recommended fields). `favorites`
# (TikTok "saves") is an extra signal present in this project's dataset.
RAW_CSV_FIELDS: tuple[str, ...] = (
    "video_id",
    "creator_username",
    "video_url",
    "video_file",
    "views",
    "likes",
    "comments",
    "shares",
    "favorites",
    "scrape_timestamp",
    "hashtags",
    "duration_seconds",
    "width",
    "height",
)


@dataclass
class RawTikTokVideo:
    """One collected video's public metrics + metadata (normalized record)."""

    video_id: str
    creator_username: str
    video_url: str = ""
    video_file: str = ""
    views: int | None = None
    likes: int | None = None
    comments: int | None = None
    shares: int | None = None
    favorites: int | None = None
    scrape_timestamp: str = ""
    hashtags: str = ""
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None

    def to_csv_row(self) -> dict[str, str]:
        """Serialize to a CSV row dict (None -> empty string)."""
        row: dict[str, str] = {}
        for name in RAW_CSV_FIELDS:
            value = getattr(self, name, None)
            row[name] = "" if value is None else str(value)
        return row

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "RawTikTokVideo":
        """Parse a raw-CSV row back into a record (blanks -> None)."""
        return cls(
            video_id=(row.get("video_id") or "").strip(),
            creator_username=(row.get("creator_username") or "").strip(),
            video_url=(row.get("video_url") or "").strip(),
            video_file=(row.get("video_file") or "").strip(),
            views=parse_count(row.get("views")),
            likes=parse_count(row.get("likes")),
            comments=parse_count(row.get("comments")),
            shares=parse_count(row.get("shares")),
            favorites=parse_count(row.get("favorites")),
            scrape_timestamp=(row.get("scrape_timestamp") or "").strip(),
            hashtags=(row.get("hashtags") or "").strip(),
            duration_seconds=_parse_float(row.get("duration_seconds")),
            width=parse_count(row.get("width")),
            height=parse_count(row.get("height")),
        )


class BaseCollector(ABC):
    """Adapter interface for collecting public TikTok data (PRD 8.3)."""

    @abstractmethod
    def collect_creator_videos(self, creator_username: str) -> list[RawTikTokVideo]:
        """Return all collected videos for a single creator."""

    @abstractmethod
    def collect_video_metrics(self, video_url: str) -> RawTikTokVideo:
        """Return the metrics/metadata for a single video URL."""

    def collect(self, creators: Iterable[str]) -> list[RawTikTokVideo]:
        """Collect videos across many creators (concrete convenience)."""
        videos: list[RawTikTokVideo] = []
        for username in creators:
            videos.extend(self.collect_creator_videos(username))
        return videos

    def save_training_csv(
        self, videos: list[RawTikTokVideo], output_path: str | Path
    ) -> None:
        """Write collected videos to the raw training CSV (PRD 8.4)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(RAW_CSV_FIELDS))
            writer.writeheader()
            for video in videos:
                writer.writerow(video.to_csv_row())


def load_raw_videos(path: str | Path) -> list[RawTikTokVideo]:
    """Read a raw training CSV back into RawTikTokVideo records."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return [RawTikTokVideo.from_csv_row(row) for row in csv.DictReader(handle)]


def parse_count(value: object) -> int | None:
    """Parse a public count to int. Blank -> None; supports '1.2K'/'3M' suffixes."""
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    suffix = text[-1].lower()
    try:
        if suffix in multipliers:
            return int(float(text[:-1]) * multipliers[suffix])
        return int(float(text))
    except ValueError:
        return None


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


# Sanity check: keep the dataclass field set and CSV schema in sync.
assert set(RAW_CSV_FIELDS) <= {f.name for f in fields(RawTikTokVideo)}
