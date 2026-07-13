"""Success-metric / label computation (PRD 10).

Turns raw public metrics into the training targets, the key idea being
**creator-relative** labels (PRD 6.3): raw views aren't comparable across
creators with different audience sizes, so we measure each video against its own
creator's distribution.

Per video we derive:
- rate metrics: like/comment/share/engagement rate (safe division; 0 when views=0)
- log_views = log(views + 1)
- creator_relative_log_views = log_views - creator_median_log_views  (main target)
- creator_relative_z = robust z-score with IQR->std->0 fallbacks (PRD 10.6)
- classification labels incl. the primary target top_quartile_for_creator (10.7)

Missing-data rules (PRD 8.5): a metric that is None makes the targets that depend
on it None (excluded later) rather than fabricating a value. Per-creator
percentiles ignore None values.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, fields

import numpy as np

from backend.collectors.base_collector import RawTikTokVideo


@dataclass
class VideoLabels:
    """All derived metrics/labels for one video (identifiers + targets)."""

    video_id: str
    creator_username: str
    # raw metrics (echoed for traceability; excluded as model features per 12.4)
    views: int | None
    likes: int | None
    comments: int | None
    shares: int | None
    favorites: int | None
    # rate metrics
    like_rate: float | None
    comment_rate: float | None
    share_rate: float | None
    engagement_rate: float | None
    # popularity / creator-relative
    log_views: float | None
    creator_median_log_views: float | None
    creator_relative_log_views: float | None
    creator_relative_z: float | None
    # classification labels
    outperformed_creator_median: bool | None
    top_quartile_for_creator: bool | None
    high_share_rate: bool | None
    high_engagement_rate: bool | None


LABEL_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(VideoLabels))


def _safe_rate(numerator: int | None, views: int | None) -> float | None:
    """numerator / views, with PRD rules: views=0 -> 0; missing inputs -> None."""
    if views is None or numerator is None:
        return None
    if views == 0:
        return 0.0
    return numerator / views


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


@dataclass
class _CreatorStats:
    median_log: float
    p75_log: float
    iqr_log: float
    std_log: float
    p75_share: float | None
    p75_engagement: float | None


def compute_labels(videos: list[RawTikTokVideo]) -> list[VideoLabels]:
    """Compute creator-relative labels for every video (order preserved)."""
    # --- pass 1: per-video base metrics ---
    base: list[dict] = []
    for video in videos:
        views = video.views
        log_views = math.log(views + 1) if views is not None else None
        likes, comments, shares = video.likes, video.comments, video.shares
        engagement_numerator = (
            (likes + comments + shares)
            if None not in (likes, comments, shares)
            else None
        )
        base.append(
            {
                "video": video,
                "log_views": log_views,
                "like_rate": _safe_rate(likes, views),
                "comment_rate": _safe_rate(comments, views),
                "share_rate": _safe_rate(shares, views),
                "engagement_rate": _safe_rate(engagement_numerator, views),
            }
        )

    # --- per-creator distributions ---
    by_creator: dict[str, list[dict]] = defaultdict(list)
    for item in base:
        by_creator[item["video"].creator_username].append(item)

    stats: dict[str, _CreatorStats] = {}
    for creator, items in by_creator.items():
        log_views = [i["log_views"] for i in items if i["log_views"] is not None]
        shares = [i["share_rate"] for i in items if i["share_rate"] is not None]
        engagement = [i["engagement_rate"] for i in items if i["engagement_rate"] is not None]
        if not log_views:
            continue
        arr = np.array(log_views, dtype=float)
        stats[creator] = _CreatorStats(
            median_log=float(np.median(arr)),
            p75_log=float(np.percentile(arr, 75)),
            iqr_log=float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            std_log=float(np.std(arr)),
            p75_share=_percentile(shares, 75),
            p75_engagement=_percentile(engagement, 75),
        )

    # --- pass 2: assign relative labels ---
    results: list[VideoLabels] = []
    for item in base:
        video = item["video"]
        log_views = item["log_views"]
        cstat = stats.get(video.creator_username)
        results.append(
            _assemble_labels(video, item, log_views, cstat)
        )
    return results


def _assemble_labels(video, item, log_views, cstat) -> VideoLabels:
    relative_log = median_log = relative_z = None
    outperformed = top_quartile = high_share = high_engagement = None

    if cstat is not None and log_views is not None:
        median_log = cstat.median_log
        relative_log = log_views - median_log
        relative_z = _robust_z(log_views, cstat)
        outperformed = relative_log > 0
        top_quartile = log_views >= cstat.p75_log
        if item["share_rate"] is not None and cstat.p75_share is not None:
            high_share = item["share_rate"] >= cstat.p75_share
        if item["engagement_rate"] is not None and cstat.p75_engagement is not None:
            high_engagement = item["engagement_rate"] >= cstat.p75_engagement

    return VideoLabels(
        video_id=video.video_id,
        creator_username=video.creator_username,
        views=video.views,
        likes=video.likes,
        comments=video.comments,
        shares=video.shares,
        favorites=video.favorites,
        like_rate=_round(item["like_rate"]),
        comment_rate=_round(item["comment_rate"]),
        share_rate=_round(item["share_rate"]),
        engagement_rate=_round(item["engagement_rate"]),
        log_views=_round(log_views),
        creator_median_log_views=_round(median_log),
        creator_relative_log_views=_round(relative_log),
        creator_relative_z=_round(relative_z),
        outperformed_creator_median=outperformed,
        top_quartile_for_creator=top_quartile,
        high_share_rate=high_share,
        high_engagement_rate=high_engagement,
    )


def _robust_z(log_views: float, cstat: _CreatorStats) -> float:
    """(log_views - median) / spread, with IQR -> std -> 0 fallbacks (PRD 10.6)."""
    if cstat.iqr_log > 0:
        return (log_views - cstat.median_log) / cstat.iqr_log
    if cstat.std_log > 0:
        return (log_views - cstat.median_log) / cstat.std_log
    return 0.0


def _round(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None
