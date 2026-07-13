"""Tests for the collector adapter interface + raw record model (PRD 8.3/8.4)."""

from __future__ import annotations

import pytest

from backend.collectors.base_collector import (
    RAW_CSV_FIELDS,
    BaseCollector,
    RawTikTokVideo,
    load_raw_videos,
    parse_count,
)


def test_parse_count_handles_blanks_commas_and_suffixes():
    assert parse_count("") is None
    assert parse_count(None) is None
    assert parse_count("1600000") == 1_600_000
    assert parse_count("1,600,000") == 1_600_000
    assert parse_count("1.2K") == 1_200
    assert parse_count("3M") == 3_000_000
    assert parse_count("garbage") is None


def test_to_csv_row_serializes_none_as_blank():
    video = RawTikTokVideo(
        video_id="123", creator_username="amy", views=1000, comments=None
    )
    row = video.to_csv_row()
    assert set(row) == set(RAW_CSV_FIELDS)
    assert row["views"] == "1000"
    assert row["comments"] == ""  # missing metric -> blank, not "None"


def test_from_csv_row_roundtrip():
    original = RawTikTokVideo(
        video_id="123",
        creator_username="amy",
        video_url="https://t/v/123",
        video_file="downloads/123.mp4",
        views=1000,
        likes=200,
        comments=None,
        shares=5,
        favorites=None,
    )
    restored = RawTikTokVideo.from_csv_row(original.to_csv_row())
    assert restored == original


def test_base_collector_is_abstract():
    with pytest.raises(TypeError):
        BaseCollector()  # type: ignore[abstract]


class _StubCollector(BaseCollector):
    """Minimal concrete collector for exercising the shared base behavior."""

    def __init__(self, data: dict[str, list[RawTikTokVideo]]):
        self._data = data

    def collect_creator_videos(self, creator_username: str) -> list[RawTikTokVideo]:
        return self._data.get(creator_username, [])

    def collect_video_metrics(self, video_url: str) -> RawTikTokVideo:
        for videos in self._data.values():
            for video in videos:
                if video.video_url == video_url:
                    return video
        raise KeyError(video_url)


def test_collect_aggregates_across_creators():
    collector = _StubCollector(
        {
            "amy": [RawTikTokVideo("1", "amy"), RawTikTokVideo("2", "amy")],
            "bob": [RawTikTokVideo("3", "bob")],
        }
    )
    videos = collector.collect(["amy", "bob"])
    assert [v.video_id for v in videos] == ["1", "2", "3"]


def test_save_training_csv_then_reload(tmp_path):
    collector = _StubCollector(
        {"amy": [RawTikTokVideo("1", "amy", views=10, shares=2, comments=None)]}
    )
    out = tmp_path / "raw.csv"
    collector.save_training_csv(collector.collect(["amy"]), out)

    reloaded = load_raw_videos(out)
    assert len(reloaded) == 1
    assert reloaded[0].views == 10
    assert reloaded[0].shares == 2
    assert reloaded[0].comments is None
