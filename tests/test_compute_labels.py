"""Tests for success-metric / label computation (PRD 10)."""

from __future__ import annotations

import math

import pytest

from backend.collectors.base_collector import RawTikTokVideo
from backend.training.compute_labels import compute_labels


def _video(vid, creator, views, likes=0, comments=0, shares=0):
    return RawTikTokVideo(
        video_id=vid,
        creator_username=creator,
        views=views,
        likes=likes,
        comments=comments,
        shares=shares,
    )


def _by_id(labels):
    return {lab.video_id: lab for lab in labels}


def test_rate_metrics_and_log_views():
    (lab,) = compute_labels([_video("1", "amy", views=1000, likes=100, comments=10, shares=5)])
    assert lab.like_rate == pytest.approx(0.1)
    assert lab.comment_rate == pytest.approx(0.01)
    assert lab.share_rate == pytest.approx(0.005)
    assert lab.engagement_rate == pytest.approx(0.115)
    assert lab.log_views == pytest.approx(math.log(1001), abs=1e-6)


def test_zero_views_sets_rates_to_zero():
    (lab,) = compute_labels([_video("1", "amy", views=0, likes=5, comments=2, shares=1)])
    assert lab.like_rate == 0.0
    assert lab.comment_rate == 0.0
    assert lab.share_rate == 0.0
    assert lab.engagement_rate == 0.0
    assert lab.log_views == 0.0  # log(0+1)


def test_missing_views_yields_none_targets():
    (lab,) = compute_labels([RawTikTokVideo("1", "amy", views=None, likes=10)])
    assert lab.log_views is None
    assert lab.like_rate is None
    assert lab.top_quartile_for_creator is None
    assert lab.creator_relative_log_views is None


def test_missing_comment_excludes_engagement_only():
    (lab,) = compute_labels(
        [RawTikTokVideo("1", "amy", views=1000, likes=100, comments=None, shares=5)]
    )
    assert lab.comment_rate is None
    assert lab.engagement_rate is None  # needs all of likes+comments+shares
    assert lab.like_rate == pytest.approx(0.1)
    assert lab.share_rate == pytest.approx(0.005)


def test_creator_relative_labels():
    videos = [
        _video("1", "amy", views=100),
        _video("2", "amy", views=200),
        _video("3", "amy", views=300),
        _video("4", "amy", views=400),
    ]
    labels = _by_id(compute_labels(videos))

    # Median log-views -> the two smaller below, two larger above.
    assert labels["1"].creator_relative_log_views < 0
    assert labels["4"].creator_relative_log_views > 0
    assert labels["1"].outperformed_creator_median is False
    assert labels["4"].outperformed_creator_median is True
    # The largest is at/above the 75th percentile.
    assert labels["4"].top_quartile_for_creator is True
    assert labels["1"].top_quartile_for_creator is False


def test_z_score_falls_back_to_zero_when_no_spread():
    # All identical views -> IQR=0 and std=0 -> z=0 (PRD 10.6 fallback).
    videos = [_video(str(i), "amy", views=500) for i in range(4)]
    for lab in compute_labels(videos):
        assert lab.creator_relative_z == 0.0
        assert lab.creator_relative_log_views == 0.0


def test_creators_are_normalized_independently():
    # Big-audience creator vs small-audience creator: relative labels, not raw.
    videos = [
        _video("a1", "big", views=1_000_000),
        _video("a2", "big", views=2_000_000),
        _video("b1", "small", views=100),
        _video("b2", "small", views=200),
    ]
    labels = _by_id(compute_labels(videos))
    # The small creator's 200-view video outperforms its own median despite tiny
    # absolute numbers, proving normalization is per-creator.
    assert labels["b2"].outperformed_creator_median is True
    assert labels["b1"].outperformed_creator_median is False
    assert labels["a1"].creator_median_log_views != labels["b1"].creator_median_log_views
