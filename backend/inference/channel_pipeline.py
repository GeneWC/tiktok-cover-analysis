"""Background pipeline for labeled channel diagnostics (D-018)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from backend.core.config import settings
from backend.features.extract_features import extract_all_features
from backend.schemas.channel import (
    ChannelFeatureDelta,
    ChannelReportResponse,
    ChannelVideoRank,
)
from backend.services import channel_store
from backend.training.channel_diagnostics import diagnose_channel
from backend.training.creator_residuals import MIN_CREATOR_VIDEOS_FOR_RESIDUALS
from backend.training.feature_groups import select_group_features
from backend.training.model_specs import PRIMARY_SPEC

_BASE_LIMITATIONS = [
    "Channel diagnostics compare videos within this upload batch only.",
    "This is creative feedback from your own hit/miss patterns — not a forecast "
    "of TikTok views or virality.",
    "Top-quartile labels (when views are provided) are computed inside this "
    "batch, relative to your median/quartile — not against other creators.",
]


def labels_from_views(views: list[int | None]) -> np.ndarray | None:
    """Creator-relative top-quartile labels inside the batch.

    Requires views on every row (all-or-nothing). Uses a rank-based top ~25%
    so ties at the percentile boundary cannot label the entire batch as hits.
    """
    if any(v is None for v in views):
        return None
    if len(views) < MIN_CREATOR_VIDEOS_FOR_RESIDUALS:
        return None
    log_views = np.array([math.log(int(v) + 1.0) for v in views], dtype=float)
    n = len(log_views)
    n_top = max(1, int(math.ceil(n * 0.25)))
    # Highest log-views first; stable tie-break by index.
    order = np.lexsort((np.arange(n), -log_views))
    labels = np.zeros(n, dtype=int)
    labels[order[:n_top]] = 1
    if int(labels.sum()) == 0 or int(labels.sum()) == n:
        return None
    return labels


def run_channel_diagnose(channel_id: str) -> None:
    """Extract features for each video, then run within-batch diagnostics."""
    record = channel_store.get_channel_job(channel_id)
    if record is None:
        return

    videos = record["videos"]
    try:
        channel_store.set_channel_step(channel_id, "features", "running")
        feature_rows: list[dict] = []
        for index, video in enumerate(videos):
            path = video["path"]
            result = extract_all_features(path)
            row = dict(result.features)
            # Drop non-numeric status strings from the feature matrix.
            row = {
                k: v
                for k, v in row.items()
                if k
                not in {
                    "audio_feature_extraction_status",
                    "video_feature_extraction_status",
                }
                and not isinstance(v, str)
            }
            row["_video_id"] = video.get("video_id") or Path(path).stem
            row["_filename"] = video.get("filename")
            row["_views"] = video.get("views")
            feature_rows.append(row)
            channel_store.update_channel_job(channel_id, n_features_done=index + 1)

        channel_store.set_channel_step(channel_id, "features", "complete")
        channel_store.set_channel_step(channel_id, "diagnose", "running")

        frame = pd.DataFrame(feature_rows)
        feature_names = select_group_features(
            [c for c in frame.columns if not c.startswith("_")],
            PRIMARY_SPEC.feature_groups,
        )
        # Keep only columns that exist and are mostly numeric.
        feature_names = [c for c in feature_names if c in frame.columns]
        X = frame[feature_names].apply(pd.to_numeric, errors="coerce")

        views_norm: list[int | None] = []
        for v in frame["_views"].tolist():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                views_norm.append(None)
            else:
                try:
                    views_norm.append(int(v))
                except (TypeError, ValueError):
                    views_norm.append(None)

        labels = labels_from_views(views_norm)
        diagnostics = diagnose_channel(
            X,
            labels=labels,
            video_ids=frame["_video_id"].astype(str).tolist(),
            creator=None,
            feature_names=feature_names,
        )

        ranks = []
        for item in diagnostics.video_ranks:
            vid = item["video_id"]
            match = frame.loc[frame["_video_id"].astype(str) == str(vid)]
            filename = None
            views_i = None
            if not match.empty:
                filename = match.iloc[0].get("_filename")
                raw_views = match.iloc[0].get("_views")
                if raw_views is not None and not (
                    isinstance(raw_views, float) and np.isnan(raw_views)
                ):
                    try:
                        views_i = int(raw_views)
                    except (TypeError, ValueError):
                        views_i = None
            ranks.append(
                ChannelVideoRank(
                    video_id=str(vid),
                    filename=None if filename is None else str(filename),
                    views=views_i,
                    presentation_score=float(item["presentation_score"]),
                    residual_l2=float(item["residual_l2"]),
                    label=item.get("label"),
                )
            )

        limitations = list(_BASE_LIMITATIONS)
        n_labeled = sum(v is not None for v in views_norm)
        if labels is None:
            limitations.append(
                "Views were missing or too uniform to form hit/miss groups, so "
                "feature deltas are unavailable. Presentation ranks still use "
                "within-batch feature levels."
            )

        report = ChannelReportResponse(
            channel_id=channel_id,
            status="complete",
            n_videos=diagnostics.n_videos,
            n_labeled=n_labeled,
            n_hits=diagnostics.n_hits,
            positive_rate=diagnostics.positive_rate,
            top_feature_deltas=[
                ChannelFeatureDelta(
                    feature=d.feature,
                    hit_mean=d.hit_mean,
                    miss_mean=d.miss_mean,
                    delta=d.delta,
                )
                for d in diagnostics.top_feature_deltas
            ],
            video_ranks=ranks,
            recommendations=list(diagnostics.recommendations),
            limitations=limitations,
            message=diagnostics.message,
        )

        channel_store.set_channel_step(channel_id, "diagnose", "complete")
        channel_store.set_channel_step(channel_id, "report", "running")

        out_path = settings.reports_dir / f"{channel_id}.json"
        out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        channel_store.update_channel_job(
            channel_id,
            status="complete",
            report_json_path=str(out_path),
        )
        channel_store.set_channel_step(channel_id, "report", "complete")
    except Exception as exc:  # noqa: BLE001 - persist failure on the job
        channel_store.update_channel_job(
            channel_id, status="failed", error=f"{type(exc).__name__}: {exc}"
        )
        channel_store.set_channel_step(channel_id, "diagnose", "failed")
        channel_store.set_channel_step(channel_id, "report", "failed")
