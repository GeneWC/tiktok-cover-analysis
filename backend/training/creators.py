"""Creator seed input (PRD 8.1) - the entry point of the training pipeline.

The offline pipeline starts from a list of creators to collect videos for. The
canonical source is `data/creators.csv` with the PRD schema:

    creator_username,profile_url[,instrument,notes]

Because this project ships with a real local dataset (`engagement.csv` +
`downloads/`), we can also *derive* the seed list straight from the engagement
file instead of hand-authoring it, then write it out in the canonical format.

The model must not require `instrument` to be present (PRD 8.1), so it is
optional here.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Canonical TikTok profile URL for a username (used when deriving the seed list).
_PROFILE_URL_TEMPLATE = "https://www.tiktok.com/@{username}"
_REQUIRED_COLUMNS = ("creator_username", "profile_url")


@dataclass(frozen=True)
class Creator:
    """One creator in the seed list."""

    username: str
    profile_url: str
    instrument: str | None = None
    notes: str | None = None


def load_creators(path: str | Path) -> list[Creator]:
    """Load and validate the creator seed list from a CSV (PRD 8.1).

    Raises FileNotFoundError if the file is missing and ValueError if the
    required columns are absent. Blank rows are skipped and duplicate usernames
    are de-duplicated (first occurrence wins).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Creator seed list not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [col for col in _REQUIRED_COLUMNS if col not in fieldnames]
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): {', '.join(missing)}"
            )

        creators: list[Creator] = []
        seen: set[str] = set()
        for row in reader:
            username = (row.get("creator_username") or "").strip()
            profile_url = (row.get("profile_url") or "").strip()
            if not username:
                continue  # skip blank/padding rows
            if username in seen:
                continue  # de-dupe, keep first
            seen.add(username)
            creators.append(
                Creator(
                    username=username,
                    profile_url=profile_url or _PROFILE_URL_TEMPLATE.format(username=username),
                    instrument=_clean_optional(row.get("instrument")),
                    notes=_clean_optional(row.get("notes")),
                )
            )
    return creators


def derive_creators_from_engagement(
    engagement_path: str | Path, instrument: str | None = None
) -> list[Creator]:
    """Build a seed list from the unique creators in an engagement CSV.

    The engagement file's `creator` column is the source of truth for which
    creators exist in the local dataset. Returns creators sorted by username.
    """
    engagement_path = Path(engagement_path)
    if not engagement_path.exists():
        raise FileNotFoundError(f"Engagement CSV not found: {engagement_path}")

    with engagement_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "creator" not in reader.fieldnames:
            raise ValueError(f"{engagement_path} has no 'creator' column")
        usernames = {(row.get("creator") or "").strip() for row in reader}

    usernames.discard("")
    return [
        Creator(
            username=username,
            profile_url=_PROFILE_URL_TEMPLATE.format(username=username),
            instrument=instrument,
        )
        for username in sorted(usernames)
    ]


def video_counts_by_creator(engagement_path: str | Path) -> dict[str, int]:
    """Count videos per creator in the engagement CSV (for reporting/sanity)."""
    engagement_path = Path(engagement_path)
    with engagement_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        counts = Counter((row.get("creator") or "").strip() for row in reader)
    counts.pop("", None)
    return dict(counts)


def write_creators_csv(creators: list[Creator], path: str | Path) -> None:
    """Write a seed list to CSV in the canonical PRD 8.1 format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["creator_username", "profile_url", "instrument", "notes"])
        for creator in creators:
            writer.writerow(
                [
                    creator.username,
                    creator.profile_url,
                    creator.instrument or "",
                    creator.notes or "",
                ]
            )


def _clean_optional(value: str | None) -> str | None:
    """Normalize an optional CSV cell to a stripped string or None."""
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None
