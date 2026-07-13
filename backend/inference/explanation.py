"""Signal breakdown + recommendations for the report (PRD 16.5-16.9).

Turns a video's features into human-readable `strong_signals` / `weak_signals` /
`neutral_or_missing_signals` and actionable `recommendations`.

Approach (kept honest and non-causal, PRD 16.6): we only describe features with
an unambiguous "more/less is better" direction, comparing this video against the
training percentiles - "stronger/weaker than most comparable covers", never
"this caused more views". Signals are prioritized by the classifier's feature
importances so the most model-relevant dimensions surface first, then capped so
the report stays focused.

Missing/absent inputs are framed neutrally, never as faults (PRD 16.7): no audio
track and no on-screen text (optional for instrumental covers) go into the
neutral list, and audio signals are simply not evaluated when there's no audio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from backend.schemas.analysis import ReportExplanation

MAX_SIGNALS = 5  # cap per list so the report stays focused (PRD 16.8)


@dataclass(frozen=True)
class SignalDef:
    """An interpretable, directional feature and how to talk about it."""

    feature: str
    direction: str        # "up" = higher is better, "down" = lower is better
    label: str
    recommendation: str


# Only features with a clear production-quality direction (mirrors the
# presentation subscores, plus text/audio where meaningful).
SIGNAL_CATALOG: tuple[SignalDef, ...] = (
    SignalDef("sharpness_full", "up", "Footage sharpness",
              "Shoot in sharper focus / higher quality so details stay crisp."),
    SignalDef("contrast_full", "up", "Visual contrast",
              "Improve lighting contrast so the performer stands out from the background."),
    SignalDef("colorfulness_full", "up", "Color vibrancy",
              "Add color or lighting variety to make the frame more vibrant."),
    SignalDef("person_visible_ratio", "up", "Performer visibility",
              "Keep the performer in frame for more of the video."),
    SignalDef("face_visible_ratio", "up", "Face visibility",
              "Show your face more consistently to build connection."),
    SignalDef("subject_centering_score", "up", "Subject centering",
              "Center the performer in the frame."),
    SignalDef("subject_size_ratio", "up", "Subject size in frame",
              "Fill more of the frame with the performer (move closer or crop tighter)."),
    SignalDef("motion_consistency", "up", "Motion steadiness",
              "Keep motion smooth and consistent; avoid abrupt changes."),
    SignalDef("camera_stability_score", "up", "Camera stability",
              "Stabilize the camera (tripod or gimbal) to reduce shake."),
    SignalDef("audio_dynamic_range", "up", "Audio dynamics",
              "Preserve dynamic range; avoid over-compressing the audio."),
    SignalDef("audio_clipping_ratio", "down", "Audio clipping",
              "Lower recording levels to avoid clipping and distortion."),
    SignalDef("audio_silence_ratio", "down", "Dead air",
              "Trim long silent sections so the audio stays engaging."),
)


def _to_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def _position(value: float, stats: dict, direction: str) -> str:
    """Classify a value vs peers as strong / weak / neutral (uses p25, p75)."""
    high = value >= stats["p75"]
    low = value <= stats["p25"]
    if direction == "up":
        return "strong" if high else "weak" if low else "neutral"
    return "strong" if low else "weak" if high else "neutral"


def build_explanation(
    features: dict,
    calibration: dict,
    importances: dict | None = None,
    has_audio: bool = True,
) -> ReportExplanation:
    """Assemble strong/weak/neutral signals + recommendations for one video."""
    percentiles = calibration.get("feature_percentiles", {})
    clf_importance = (importances or {}).get("top_quartile", {})
    # Most model-relevant features first (unknown -> lowest priority).
    ordered = sorted(
        SIGNAL_CATALOG,
        key=lambda s: clf_importance.get(s.feature, 0.0),
        reverse=True,
    )

    strong: list[str] = []
    weak: list[str] = []
    neutral: list[str] = []
    recommendations: list[str] = []

    for sig in ordered:
        if sig.feature.startswith("audio_") and not has_audio:
            continue  # not evaluated; covered by the neutral no-audio note
        value = _to_float(features.get(sig.feature))
        stats = percentiles.get(sig.feature)
        if value is None or stats is None:
            continue

        position = _position(value, stats, sig.direction)
        if position == "strong" and len(strong) < MAX_SIGNALS:
            strong.append(f"{sig.label}: stronger than most comparable covers.")
        elif position == "weak" and len(weak) < MAX_SIGNALS:
            weak.append(f"{sig.label}: weaker than most comparable covers.")
            recommendations.append(sig.recommendation)

    if not has_audio:
        neutral.append("No audio track detected; audio-based signals were not evaluated.")

    if _to_float(features.get("ocr_failed")) == 1.0:
        neutral.append("On-screen text detection was unavailable for this video.")
    elif _to_float(features.get("text_present_anywhere")) == 0.0:
        neutral.append("No on-screen text detected (optional for instrumental covers).")

    return ReportExplanation(
        strong_signals=strong,
        weak_signals=weak,
        neutral_or_missing_signals=neutral,
        recommendations=recommendations[:MAX_SIGNALS],
    )
