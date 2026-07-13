"""Builds the user-facing analysis report (PRD 19.3 / 16).

Three entry points share one report shape (single source of truth):
- `build_pending_report` - job accepted but the pipeline hasn't run yet (nulls +
  a "pending" note). Used by the report endpoint before processing finishes.
- `build_analysis_report` - the full report: predicted probability + tiers,
  0-100 presentation scores, the signal/recommendation breakdown, and the
  standing exploratory disclaimers (+ low-confidence / no-audio caveats).
- `build_unusable_report` - the video couldn't be analyzed (undecodable, or a
  stage failed): nulls + a clear note, so the API still returns a coherent report.
"""

from __future__ import annotations

from backend.inference.prediction import Predictions
from backend.inference.presentation import PresentationScores
from backend.schemas.analysis import (
    ReportExplanation,
    ReportResponse,
    ReportScores,
    VideoMetadata,
)

# Always-present disclaimers (PRD 16.9 / 6.4: exploratory, never causal).
_BASE_LIMITATIONS = (
    "Predictions are exploratory and based on similarity to TikTok instrumental "
    "cover videos in the training dataset.",
    "Actual TikTok performance depends on additional factors such as audience "
    "size, platform distribution, viewer preferences, timing, and external trends.",
)

_PIPELINE_PENDING_NOTE = (
    "Prediction is not yet available: the analysis pipeline has not finished "
    "for this video."
)

_UNUSABLE_NOTE = (
    "This video could not be analyzed (it may be unreadable or too short). No "
    "predictions were generated."
)

# Added when any low-confidence tier is shown (PRD 16.9).
_LOW_CONFIDENCE_NOTE = (
    "View-performance and shareability tiers are low-confidence: these signals "
    "did not generalize well across creators and should be treated as exploratory."
)

# Shown when the uploaded video has no audio track (PRD 11.6).
_NO_AUDIO_WARNING = (
    "No audio track was detected. Audio-based predictions may be unavailable or "
    "lower confidence."
)


def _video_metadata(record: dict) -> VideoMetadata | None:
    metadata_dict = record.get("metadata")
    return VideoMetadata(**metadata_dict) if metadata_dict else None


def build_pending_report(record: dict) -> ReportResponse:
    """Report for a job whose pipeline hasn't produced predictions yet."""
    video_metadata = _video_metadata(record)
    limitations = [*_BASE_LIMITATIONS, _PIPELINE_PENDING_NOTE]
    explanation = ReportExplanation()
    if video_metadata is not None and not video_metadata.has_audio:
        limitations.append(_NO_AUDIO_WARNING)
        explanation.neutral_or_missing_signals.append("No audio track detected.")

    return ReportResponse(
        analysis_id=record["analysis_id"],
        status=record["status"],
        video_metadata=video_metadata,
        scores=ReportScores(),
        features={},
        explanation=explanation,
        limitations=limitations,
    )


# Kept as an alias so existing callers of build_report get the pending report.
build_report = build_pending_report


def build_unusable_report(
    record: dict, explanation: ReportExplanation | None = None
) -> ReportResponse:
    """Report for a video that could not be analyzed (nulls + clear note)."""
    return ReportResponse(
        analysis_id=record["analysis_id"],
        status="failed",
        video_metadata=_video_metadata(record),
        scores=ReportScores(),
        features={},
        explanation=explanation or ReportExplanation(),
        limitations=[*_BASE_LIMITATIONS, _UNUSABLE_NOTE],
    )


def _scores(predictions: Predictions, presentation: PresentationScores) -> ReportScores:
    tiers = predictions.tiers

    def tier(name: str) -> str | None:
        entry = tiers.get(name)
        return entry.tier if entry else None

    return ReportScores(
        top_quartile_probability=predictions.top_quartile_probability,
        view_performance_tier=tier("view_performance_tier"),
        engagement_tier=tier("engagement_tier"),
        shareability_tier=tier("shareability_tier"),
        overall_presentation_score=presentation.overall_presentation_score,
        visual_quality_score=presentation.visual_quality_score,
        audio_quality_score=presentation.audio_quality_score,
        motion_score=presentation.motion_score,
        framing_score=presentation.framing_score,
    )


def build_analysis_report(
    record: dict,
    predictions: Predictions,
    presentation: PresentationScores,
    explanation: ReportExplanation,
    features: dict | None = None,
) -> ReportResponse:
    """Assemble the full report from computed predictions/scores/signals."""
    video_metadata = _video_metadata(record)
    limitations = list(_BASE_LIMITATIONS)

    if any(t.low_confidence for t in predictions.tiers.values()):
        limitations.append(_LOW_CONFIDENCE_NOTE)
    if video_metadata is not None and not video_metadata.has_audio:
        limitations.append(_NO_AUDIO_WARNING)

    return ReportResponse(
        analysis_id=record["analysis_id"],
        status="complete",
        video_metadata=video_metadata,
        scores=_scores(predictions, presentation),
        features=features or {},
        explanation=explanation,
        limitations=limitations,
    )
