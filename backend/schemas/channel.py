"""Pydantic schemas for labeled channel diagnostics (D-018)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from backend.schemas.analysis import AnalysisStatus, StepStatus

ChannelStepStatus = StepStatus
ChannelJobStatus = AnalysisStatus


class ChannelAnalyzeResponse(BaseModel):
    """Returned by POST /api/channel/diagnose when a job is accepted."""

    channel_id: str = Field(..., examples=["channel_8f3c1a2b"])
    status: ChannelJobStatus = Field(..., examples=["processing"])
    n_videos: int


class ChannelStatusResponse(BaseModel):
    channel_id: str
    status: ChannelJobStatus
    steps: dict[str, ChannelStepStatus]
    n_videos: int
    n_features_done: int = 0
    error: str | None = None


class ChannelFeatureDelta(BaseModel):
    feature: str
    hit_mean: float
    miss_mean: float
    delta: float


class ChannelVideoRank(BaseModel):
    video_id: str
    filename: str | None = None
    views: int | None = None
    presentation_score: float
    residual_l2: float
    label: int | None = None


class ChannelReportResponse(BaseModel):
    """Honest within-creator diagnostics — not a virality forecast."""

    channel_id: str
    status: ChannelJobStatus
    mode: Literal["diagnostics"] = "diagnostics"
    n_videos: int
    n_labeled: int
    n_hits: int | None = None
    positive_rate: float | None = None
    top_feature_deltas: list[ChannelFeatureDelta] = Field(default_factory=list)
    video_ranks: list[ChannelVideoRank] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    message: str | None = None
