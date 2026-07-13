"""Tests for raw training CSV generation (PRD 8.4 / 20.1)."""

from __future__ import annotations

from backend.collectors.base_collector import load_raw_videos
from backend.training.collect_dataset import build_raw_training_csv


def _make_dataset(tmp_path):
    (tmp_path / "creators.csv").write_text(
        "creator_username,profile_url\n"
        "amy,https://www.tiktok.com/@amy\n"
        "bob,https://www.tiktok.com/@bob\n",
        encoding="utf-8",
    )
    (tmp_path / "engagement.csv").write_text(
        "creator,video_id,url,likes,comments,favorites,views,shares\n"
        "amy,1,https://t/v/1,200,10,5,1000,3\n"
        "bob,2,https://t/v/2,80,,,800,1\n",
        encoding="utf-8",
    )
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    (downloads / "1.mp4").write_bytes(b"\x00")  # video 2 missing on disk
    return downloads


def test_build_raw_training_csv_writes_rows_and_summary(tmp_path):
    downloads = _make_dataset(tmp_path)
    out = tmp_path / "raw.csv"

    summary = build_raw_training_csv(
        creators_csv=tmp_path / "creators.csv",
        engagement_csv=tmp_path / "engagement.csv",
        downloads_dir=downloads,
        out_path=out,
    )

    assert summary["creators"] == 2
    assert summary["videos"] == 2
    assert summary["matched_files"] == 1
    assert summary["missing_files"] == 1

    videos = load_raw_videos(out)
    assert {v.video_id for v in videos} == {"1", "2"}
    by_id = {v.video_id: v for v in videos}
    assert by_id["1"].video_file == "downloads/1.mp4"
    assert by_id["2"].video_file == ""
    assert by_id["2"].comments is None
