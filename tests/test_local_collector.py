"""Tests for the local-dataset collector (PRD 8.3)."""

from __future__ import annotations

import pytest

from backend.collectors.local_collector import LocalDatasetCollector

_HEADER = "creator,video_id,url,likes,comments,favorites,views,shares\n"


@pytest.fixture
def dataset(tmp_path):
    """A tiny engagement CSV + downloads folder; one video file is missing."""
    eng = tmp_path / "engagement.csv"
    eng.write_text(
        _HEADER
        + "amy,1,https://t/v/1,200,10,5,1000,3\n"
        + "amy,2,https://t/v/2,50,,,500,1\n"  # blank comments/favorites
        + "bob,3,https://t/v/3,80,4,2,800,0\n",
        encoding="utf-8",
    )
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    for vid in ("1", "2"):  # video 3 is intentionally absent
        (downloads / f"{vid}.mp4").write_bytes(b"\x00")
    return LocalDatasetCollector(engagement_csv=eng, downloads_dir=downloads)


def test_collect_creator_videos_parses_metrics(dataset):
    videos = dataset.collect_creator_videos("amy")
    assert [v.video_id for v in videos] == ["1", "2"]
    first = videos[0]
    assert first.views == 1000
    assert first.likes == 200
    assert first.shares == 3
    # Blank cells become None (PRD 8.5 missing handling).
    assert videos[1].comments is None
    assert videos[1].favorites is None


def test_video_file_matched_only_when_present(dataset):
    by_id = {v.video_id: v for v in dataset.collect_all()}
    assert by_id["1"].video_file == "downloads/1.mp4"
    assert by_id["3"].video_file == ""  # no file on disk


def test_collect_video_metrics_by_url(dataset):
    video = dataset.collect_video_metrics("https://t/v/3")
    assert video.creator_username == "bob"
    assert video.views == 800
    with pytest.raises(KeyError):
        dataset.collect_video_metrics("https://t/v/does-not-exist")


def test_collect_across_seed_list(dataset):
    videos = dataset.collect(["amy", "bob"])
    assert [v.video_id for v in videos] == ["1", "2", "3"]


def test_match_summary(dataset):
    assert dataset.match_summary() == {
        "total": 3,
        "matched_files": 2,
        "missing_files": 1,
        "creators": 2,
    }


def test_missing_required_column_raises(tmp_path):
    eng = tmp_path / "engagement.csv"
    eng.write_text("creator,video_id,url,likes\namy,1,u,2\n", encoding="utf-8")
    collector = LocalDatasetCollector(engagement_csv=eng, downloads_dir=tmp_path)
    with pytest.raises(ValueError, match="views"):
        collector.collect_all()
