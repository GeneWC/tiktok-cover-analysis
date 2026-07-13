"""Presentation subscores (PRD 15).

Model-independent, transparent 0-100 scores for the four production dimensions
(visual quality, audio quality, motion, framing) plus an overall average. These
are *not* ML predictions: each score normalizes a small set of clearly-directional
features against their training percentiles (p5 -> 0, p95 -> 100, clipped), so a
"70" means "around the 70th percentile of the training videos on that dimension".

Only features with an unambiguous "more/less is better" direction are used, so the
scores stay explainable (PRD 15 / 16): e.g. sharper/steadier/less-clipped is
better; brightness (where a mid-range is ideal) is deliberately excluded.

Missing inputs are skipped, not penalized: a video with no audio track yields
`audio_quality_score = None` (not 0), and the overall score averages only the
dimensions that could be computed (PRD 11.6 / 16.7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean

# Per subscore: (feature, direction). "up" = higher is better, "down" = lower.
PRESENTATION_FEATURES: dict[str, list[tuple[str, str]]] = {
    "visual_quality_score": [
        ("sharpness_full", "up"),
        ("contrast_full", "up"),
        ("colorfulness_full", "up"),
    ],
    "audio_quality_score": [
        ("audio_dynamic_range", "up"),
        ("audio_clipping_ratio", "down"),
        ("audio_silence_ratio", "down"),
    ],
    "motion_score": [
        ("motion_consistency", "up"),
        ("camera_stability_score", "up"),
    ],
    "framing_score": [
        ("person_visible_ratio", "up"),
        ("face_visible_ratio", "up"),
        ("subject_centering_score", "up"),
        ("subject_size_ratio", "up"),
    ],
}

_NEUTRAL_SCORE = 50.0


@dataclass(frozen=True)
class PresentationScores:
    """0-100 presentation subscores + overall (None where not computable)."""

    visual_quality_score: float | None
    audio_quality_score: float | None
    motion_score: float | None
    framing_score: float | None
    overall_presentation_score: float | None


def _to_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def _normalize(value: float, p5: float, p95: float, direction: str) -> float:
    """Map a value onto 0-100 by its training p5..p95 spread, honoring direction."""
    if p95 <= p5:
        return _NEUTRAL_SCORE  # no spread to rank against
    frac = min(1.0, max(0.0, (value - p5) / (p95 - p5)))
    if direction == "down":
        frac = 1.0 - frac
    return frac * 100.0


def _subscore(
    features: dict, percentiles: dict, spec: list[tuple[str, str]]
) -> float | None:
    """Average the normalized, directional features available for one dimension."""
    scores: list[float] = []
    for feature, direction in spec:
        value = _to_float(features.get(feature))
        stats = percentiles.get(feature)
        if value is None or stats is None:
            continue
        scores.append(_normalize(value, stats["p5"], stats["p95"], direction))
    return round(mean(scores), 1) if scores else None


def compute_presentation_scores(features: dict, calibration: dict) -> PresentationScores:
    """Compute all presentation subscores from a raw feature dict + calibration."""
    percentiles = calibration.get("feature_percentiles", {})
    sub = {
        name: _subscore(features, percentiles, spec)
        for name, spec in PRESENTATION_FEATURES.items()
    }

    computed = [score for score in sub.values() if score is not None]
    overall = round(mean(computed), 1) if computed else None

    return PresentationScores(
        visual_quality_score=sub["visual_quality_score"],
        audio_quality_score=sub["audio_quality_score"],
        motion_score=sub["motion_score"],
        framing_score=sub["framing_score"],
        overall_presentation_score=overall,
    )
