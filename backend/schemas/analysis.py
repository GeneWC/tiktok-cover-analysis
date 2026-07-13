"""Pydantic schemas for the analysis API.

These models define the *contract* of the API: the exact JSON shapes that go
out (and, later, come in). FastAPI uses them to validate, serialize, and
document responses automatically. Shapes follow PRD section 19.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Predicted performance tiers (PRD 12.2). Preferred over fake exact numbers.
PerformanceTier = Literal["low", "medium", "medium_high", "high"]

# The five legal states for any pipeline step (PRD section 14.3).
StepStatus = Literal["pending", "running", "complete", "failed", "skipped"]

# Overall status of an analysis job.
AnalysisStatus = Literal["processing", "complete", "failed"]


class AnalyzeResponse(BaseModel):
    """Returned by POST /api/analyze when a job is accepted (PRD 19.1)."""

    analysis_id: str = Field(..., examples=["analysis_8f3c1a2b"])
    status: AnalysisStatus = Field(..., examples=["processing"])


class StatusResponse(BaseModel):
    """Returned by GET /api/analyze/{id}/status (PRD 19.2).

    `steps` maps each pipeline step name to its current status. Clients poll this
    endpoint to drive the processing UI.
    """

    analysis_id: str
    status: AnalysisStatus
    steps: dict[str, StepStatus]


class VideoMetadata(BaseModel):
    """Basic container metadata read during upload validation (PRD 11.2 / 16.2).

    This is the cheap metadata available straight from the container header.
    Richer per-window visual/audio features come later in the Phase 2 extractor.
    """

    duration_seconds: float
    width: int
    height: int
    fps: float | None = None
    has_audio: bool
    aspect_ratio: float | None = None
    is_vertical_video: bool
    is_square_video: bool


class ReportScores(BaseModel):
    """Predicted tiers and 0-100 presentation scores (PRD 14.4 / 15).

    All fields are nullable: until the feature-extraction + model pipeline runs,
    they are None rather than fabricated numbers (PRD forbids fake metrics).
    """

    top_quartile_probability: float | None = None
    view_performance_tier: PerformanceTier | None = None
    engagement_tier: PerformanceTier | None = None
    shareability_tier: PerformanceTier | None = None
    overall_presentation_score: float | None = None
    visual_quality_score: float | None = None
    audio_quality_score: float | None = None
    motion_score: float | None = None
    framing_score: float | None = None


class ReportExplanation(BaseModel):
    """Human-readable signal breakdown and recommendations (PRD 16.5-16.8)."""

    strong_signals: list[str] = Field(default_factory=list)
    weak_signals: list[str] = Field(default_factory=list)
    neutral_or_missing_signals: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class ReportResponse(BaseModel):
    """Full analysis report (PRD 19.3 / 16).

    In Phase 1 the metadata section is populated from upload validation; scores,
    features, and explanations are placeholders until later phases compute them.
    """

    analysis_id: str
    status: AnalysisStatus
    video_metadata: VideoMetadata | None = None
    scores: ReportScores = Field(default_factory=ReportScores)
    features: dict[str, float | int | bool | None] = Field(default_factory=dict)
    explanation: ReportExplanation = Field(default_factory=ReportExplanation)
    limitations: list[str] = Field(default_factory=list)
