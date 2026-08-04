"""Unit tests for the spreadsheet -> engagement CSV merge (dataset expansion)."""

from __future__ import annotations

import pandas as pd
import pytest

from backend.training.merge_spreadsheet_dataset import (
    ENGAGEMENT_COLUMNS,
    build_engagement_rows,
    downloaded_video_ids,
    existing_favorites,
    write_engagement_csv,
)


def _write_xlsx(path, records: list[dict]):
    """Write a minimal spreadsheet export with the columns the merge needs."""
    rows = [
        {
            "Video Page URL": f"https://www.tiktok.com/@{r['creator']}/video/{r['video_id']}",
            "Author Username": r["creator"],
            "Video View Count": r.get("views", "1000"),
            "Video Like Count": r.get("likes", "100"),
            "Video Comment Count": r.get("comments", "10"),
            "Video Share Count": r.get("shares", "5"),
        }
        for r in records
    ]
    pd.DataFrame(rows).to_excel(path, index=False)
    return path


def _touch_videos(downloads_dir, video_ids: list[str]):
    downloads_dir.mkdir(parents=True, exist_ok=True)
    for video_id in video_ids:
        (downloads_dir / f"{video_id}.mp4").write_bytes(b"fake")
    return downloads_dir


def test_maps_spreadsheet_columns_to_engagement_schema(tmp_path):
    xlsx = _write_xlsx(
        tmp_path / "new.xlsx",
        [
            {
                "creator": "amy",
                "video_id": "1",
                "views": "7400000",
                "likes": "824900",
                "comments": "2967",
                "shares": "33100",
            }
        ],
    )
    downloads = _touch_videos(tmp_path / "downloads", ["1"])

    rows, summary = build_engagement_rows(xlsx, downloads, min_videos_per_creator=1)

    assert rows == [
        {
            "creator": "amy",
            "video_id": "1",
            "url": "https://www.tiktok.com/@amy/video/1",
            "favorites": "",
            "views": "7400000",
            "likes": "824900",
            "comments": "2967",
            "shares": "33100",
        }
    ]
    assert set(rows[0]) == set(ENGAGEMENT_COLUMNS)
    assert summary["written_videos"] == 1
    assert summary["creators"] == 1


def test_videos_without_a_downloaded_file_are_dropped(tmp_path):
    xlsx = _write_xlsx(
        tmp_path / "new.xlsx",
        [{"creator": "amy", "video_id": "1"}, {"creator": "amy", "video_id": "2"}],
    )
    downloads = _touch_videos(tmp_path / "downloads", ["1"])

    rows, summary = build_engagement_rows(xlsx, downloads, min_videos_per_creator=1)

    assert [r["video_id"] for r in rows] == ["1"]
    assert summary["spreadsheet_videos"] == 2
    assert summary["with_downloaded_file"] == 1


def test_creators_below_min_videos_are_excluded(tmp_path):
    """Creator-relative labels are degenerate for a creator with too few videos."""
    records = [{"creator": "amy", "video_id": str(i)} for i in range(1, 4)]
    records.append({"creator": "solo", "video_id": "99"})
    xlsx = _write_xlsx(tmp_path / "new.xlsx", records)
    downloads = _touch_videos(tmp_path / "downloads", ["1", "2", "3", "99"])

    rows, summary = build_engagement_rows(xlsx, downloads, min_videos_per_creator=3)

    assert {r["creator"] for r in rows} == {"amy"}
    assert summary["excluded_low_video_creators"] == 1
    assert summary["creators"] == 1


def test_favorites_carried_over_from_previous_engagement(tmp_path):
    xlsx = _write_xlsx(
        tmp_path / "new.xlsx",
        [{"creator": "amy", "video_id": "1"}, {"creator": "amy", "video_id": "2"}],
    )
    downloads = _touch_videos(tmp_path / "downloads", ["1", "2"])
    previous = tmp_path / "engagement.csv"
    previous.write_text(
        "creator,video_id,url,likes,comments,favorites,views,shares\n"
        "amy,1,u,1,1,4567,1,1\n",
        encoding="utf-8",
    )

    rows, summary = build_engagement_rows(
        xlsx, downloads, previous_engagement=previous, min_videos_per_creator=1
    )

    by_id = {r["video_id"]: r["favorites"] for r in rows}
    assert by_id == {"1": "4567", "2": ""}
    assert summary["favorites_carried_over"] == 1


def test_duplicate_video_ids_deduped_and_rows_sorted(tmp_path):
    xlsx = _write_xlsx(
        tmp_path / "new.xlsx",
        [
            {"creator": "zoe", "video_id": "9"},
            {"creator": "amy", "video_id": "2"},
            {"creator": "amy", "video_id": "2"},
            {"creator": "amy", "video_id": "1"},
        ],
    )
    downloads = _touch_videos(tmp_path / "downloads", ["1", "2", "9"])

    rows, _ = build_engagement_rows(xlsx, downloads, min_videos_per_creator=1)

    assert [(r["creator"], r["video_id"]) for r in rows] == [
        ("amy", "1"),
        ("amy", "2"),
        ("zoe", "9"),
    ]


def test_missing_required_column_raises(tmp_path):
    path = tmp_path / "new.xlsx"
    pd.DataFrame([{"Video Page URL": "x", "Author Username": "amy"}]).to_excel(
        path, index=False
    )
    with pytest.raises(ValueError, match="Video View Count"):
        build_engagement_rows(path, tmp_path / "downloads")


def test_missing_counts_become_blank(tmp_path):
    path = tmp_path / "new.xlsx"
    pd.DataFrame(
        [
            {
                "Video Page URL": "https://www.tiktok.com/@amy/video/1",
                "Author Username": "amy",
                "Video View Count": "1000",
                "Video Like Count": None,
                "Video Comment Count": "10",
                "Video Share Count": "5",
            }
        ]
    ).to_excel(path, index=False)
    downloads = _touch_videos(tmp_path / "downloads", ["1"])

    rows, _ = build_engagement_rows(path, downloads, min_videos_per_creator=1)

    assert rows[0]["likes"] == ""
    assert rows[0]["views"] == "1000"


def test_downloaded_video_ids_ignores_non_video_files(tmp_path):
    downloads = _touch_videos(tmp_path / "downloads", ["1", "2"])
    (downloads / "notes.txt").write_text("x", encoding="utf-8")
    assert downloaded_video_ids(downloads) == {"1", "2"}


def test_downloaded_video_ids_missing_dir_is_empty(tmp_path):
    assert downloaded_video_ids(tmp_path / "nope") == set()


def test_existing_favorites_absent_file_is_empty(tmp_path):
    assert existing_favorites(tmp_path / "nope.csv") == {}


def test_write_engagement_csv_roundtrip(tmp_path):
    xlsx = _write_xlsx(tmp_path / "new.xlsx", [{"creator": "amy", "video_id": "1"}])
    downloads = _touch_videos(tmp_path / "downloads", ["1"])
    rows, _ = build_engagement_rows(xlsx, downloads, min_videos_per_creator=1)

    out = tmp_path / "out" / "engagement.csv"
    write_engagement_csv(rows, out)

    reloaded = pd.read_csv(out, dtype=str).fillna("")
    assert list(reloaded.columns) == list(ENGAGEMENT_COLUMNS)
    assert reloaded.to_dict("records") == rows
