"""Unit tests for the creator seed loader (PRD 8.1)."""

from __future__ import annotations

import pytest

from backend.training.creators import (
    Creator,
    derive_creators_from_engagement,
    load_creators,
    video_counts_by_creator,
    write_creators_csv,
)


def _write(path, text: str):
    path.write_text(text, encoding="utf-8")
    return path


def test_load_full_seed_list(tmp_path):
    csv_path = _write(
        tmp_path / "creators.csv",
        "creator_username,profile_url,instrument,notes\n"
        "geneviolin,https://www.tiktok.com/@geneviolin,violin,seed\n",
    )
    creators = load_creators(csv_path)
    assert creators == [
        Creator("geneviolin", "https://www.tiktok.com/@geneviolin", "violin", "seed")
    ]


def test_optional_columns_absent_are_none(tmp_path):
    csv_path = _write(
        tmp_path / "creators.csv",
        "creator_username,profile_url\npianoguy,https://www.tiktok.com/@pianoguy\n",
    )
    (creator,) = load_creators(csv_path)
    assert creator.instrument is None
    assert creator.notes is None


def test_missing_required_column_raises(tmp_path):
    csv_path = _write(tmp_path / "creators.csv", "creator_username\nfoo\n")
    with pytest.raises(ValueError, match="profile_url"):
        load_creators(csv_path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_creators(tmp_path / "nope.csv")


def test_blank_rows_skipped_and_deduped(tmp_path):
    csv_path = _write(
        tmp_path / "creators.csv",
        "creator_username,profile_url\n"
        "  alice  ,  https://www.tiktok.com/@alice  \n"
        ",\n"
        "alice,https://www.tiktok.com/@alice2\n"
        "bob,\n",
    )
    creators = load_creators(csv_path)
    assert [c.username for c in creators] == ["alice", "bob"]
    # Whitespace stripped; first occurrence of a duplicate username wins.
    assert creators[0].profile_url == "https://www.tiktok.com/@alice"
    # Blank profile_url falls back to the canonical URL.
    assert creators[1].profile_url == "https://www.tiktok.com/@bob"


def test_derive_from_engagement_sorted_unique(tmp_path):
    eng = _write(
        tmp_path / "engagement.csv",
        "creator,video_id,likes\n"
        "zoe,1,10\n"
        "amy,2,20\n"
        "zoe,3,30\n",
    )
    creators = derive_creators_from_engagement(eng, instrument="violin")
    assert [c.username for c in creators] == ["amy", "zoe"]
    assert all(c.instrument == "violin" for c in creators)
    assert creators[0].profile_url == "https://www.tiktok.com/@amy"


def test_video_counts_by_creator(tmp_path):
    eng = _write(
        tmp_path / "engagement.csv",
        "creator,video_id\nzoe,1\namy,2\nzoe,3\n",
    )
    assert video_counts_by_creator(eng) == {"zoe": 2, "amy": 1}


def test_write_then_load_roundtrip(tmp_path):
    creators = [
        Creator("amy", "https://www.tiktok.com/@amy", "violin", None),
        Creator("bob", "https://www.tiktok.com/@bob", None, "note"),
    ]
    out = tmp_path / "creators.csv"
    write_creators_csv(creators, out)
    assert load_creators(out) == creators
