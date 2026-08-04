"""Spreadsheet -> engagement CSV merge (dataset expansion).

A second batch of videos arrived as a spreadsheet export (`new_video_data.xlsx`:
one row per TikTok video with public counts and a page URL) plus the matching
files in `downloads/`. This module normalizes that export into the canonical
`engagement.csv` schema the existing pipeline already consumes, so nothing
downstream (collector, labels, feature extraction, dataset build) has to change.

Two rules matter here:

- **Only videos with a file on disk.** A metrics row with no downloadable video
  can't be feature-extracted, so it is dropped at the source rather than
  becoming a `failed` row later.
- **Minimum videos per creator.** The training targets are creator-relative
  (PRD 6.3/10): a video is scored against its own creator's view distribution.
  A creator with one video is trivially its own median and 75th percentile, so
  `top_quartile_for_creator` would always be True - a degenerate label. Creators
  below the threshold are excluded.

`favorites` (TikTok "saves") is not in the spreadsheet export. It is carried
over from a previous engagement CSV where video_ids overlap and left blank
otherwise; blank parses to None, which the label computation already tolerates.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pandas as pd

# Canonical engagement CSV schema (what LocalDatasetCollector reads).
ENGAGEMENT_COLUMNS: tuple[str, ...] = (
    "creator",
    "video_id",
    "url",
    "likes",
    "comments",
    "favorites",
    "views",
    "shares",
)

# Spreadsheet column -> engagement CSV column.
METRIC_COLUMNS: dict[str, str] = {
    "Video View Count": "views",
    "Video Like Count": "likes",
    "Video Comment Count": "comments",
    "Video Share Count": "shares",
}

_URL_COLUMN = "Video Page URL"
_CREATOR_COLUMN = "Author Username"
_VIDEO_ID_PATTERN = r"/video/(\d+)"
_VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".webm", ".mkv")


def downloaded_video_ids(downloads_dir: str | Path) -> set[str]:
    """video_ids that have an actual media file in the downloads folder."""
    downloads_dir = Path(downloads_dir)
    if not downloads_dir.exists():
        return set()
    return {
        path.stem
        for path in downloads_dir.iterdir()
        if path.suffix.lower() in _VIDEO_EXTENSIONS
    }


def existing_favorites(engagement_csv: str | Path) -> dict[str, str]:
    """{video_id: favorites} from an existing engagement CSV ({} if absent)."""
    engagement_csv = Path(engagement_csv)
    if not engagement_csv.exists():
        return {}
    with engagement_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "favorites" not in (reader.fieldnames or []):
            return {}
        return {
            video_id: (row.get("favorites") or "").strip()
            for row in reader
            if (video_id := (row.get("video_id") or "").strip())
        }


def build_engagement_rows(
    xlsx_path: str | Path,
    downloads_dir: str | Path,
    previous_engagement: str | Path | None = None,
    min_videos_per_creator: int = 5,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Normalize the spreadsheet export into engagement rows.

    Returns `(rows, summary)`, where rows are sorted by (creator, video_id) and
    contain exactly `ENGAGEMENT_COLUMNS`.
    """
    frame = pd.read_excel(xlsx_path, dtype=str)

    required = (*METRIC_COLUMNS, _URL_COLUMN, _CREATOR_COLUMN)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{xlsx_path} missing column(s): {', '.join(missing)}")

    frame = frame.assign(
        video_id=frame[_URL_COLUMN].astype(str).str.extract(_VIDEO_ID_PATTERN)[0],
        creator=frame[_CREATOR_COLUMN].astype(str).str.strip(),
    )
    frame = frame.dropna(subset=["video_id"]).drop_duplicates("video_id")
    spreadsheet_videos = len(frame)

    frame = frame[frame["video_id"].isin(downloaded_video_ids(downloads_dir))]
    with_file = len(frame)

    counts = Counter(frame["creator"])
    kept_creators = {
        creator for creator, n in counts.items() if n >= min_videos_per_creator
    }
    frame = frame[frame["creator"].isin(kept_creators)]

    favorites = existing_favorites(previous_engagement) if previous_engagement else {}

    rows: list[dict[str, str]] = []
    for record in frame.to_dict("records"):
        video_id = str(record["video_id"])
        row = {
            "creator": record["creator"],
            "video_id": video_id,
            "url": str(record[_URL_COLUMN]).strip(),
            "favorites": favorites.get(video_id, ""),
        }
        for source, target in METRIC_COLUMNS.items():
            row[target] = _clean_count(record.get(source))
        rows.append(row)

    rows.sort(key=lambda row: (row["creator"], row["video_id"]))

    summary = {
        "spreadsheet_videos": spreadsheet_videos,
        "with_downloaded_file": with_file,
        "excluded_low_video_creators": with_file - len(rows),
        "written_videos": len(rows),
        "creators": len(kept_creators),
        "favorites_carried_over": sum(1 for row in rows if row["favorites"]),
    }
    return rows, summary


def write_engagement_csv(rows: list[dict[str, str]], out_path: str | Path) -> None:
    """Write engagement rows in the canonical schema."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ENGAGEMENT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _clean_count(value: object) -> str:
    """Spreadsheet cell -> CSV string ('' for missing; parsed downstream)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("nan", "none") else text
