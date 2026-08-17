"""Model scoring for one assembled video (PRD 12.1 / 12.2 / 16.9).

Runs the loaded pipelines on the schema-aligned feature row:

- the primary classifier -> `top_quartile_probability` (0-1), the headline
  "does this upload resemble a creator's top-quartile videos" signal;
- each regressor -> a raw prediction mapped to a report tier
  (low/medium/medium_high/high) via the calibration thresholds.

Each pipeline preprocesses its own feature subset internally, so we just hand it
the columns its schema entry lists. Low-confidence regressors (creator-relative
views, shareability) still produce a tier, but the flag rides along so the report
can present them as exploratory rather than authoritative (PRD 16.9).
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.inference.feature_assembly import FeatureSchemaError
from backend.inference.model_registry import ModelRegistry, get_registry
from backend.training.calibration import tier_for


@dataclass(frozen=True)
class TierPrediction:
    """A regressor's tier output plus provenance for the report."""

    model: str            # model name (e.g. "engagement")
    tier_name: str        # report field name (e.g. "engagement_tier")
    tier: str             # low | medium | medium_high | high
    raw_value: float      # underlying regression prediction
    low_confidence: bool  # weak cross-creator signal -> present as exploratory


@dataclass(frozen=True)
class Predictions:
    """All model outputs for one video."""

    top_quartile_probability: float
    tiers: dict[str, TierPrediction]  # keyed by tier_name (report field name)


def _feature_row(assembled_or_frame):
    """Accept either an AssembledFeatures or a raw feature DataFrame."""
    return getattr(assembled_or_frame, "X", assembled_or_frame)


def predict(assembled, registry: ModelRegistry | None = None) -> Predictions:
    """Score one assembled video into a probability + per-tier predictions."""
    registry = registry or get_registry()
    X = _feature_row(assembled)
    expected = list(registry.all_features)
    if list(X.columns) != expected:
        raise FeatureSchemaError(
            "Prediction input columns do not match the loaded feature schema."
        )

    clf = registry.classifier
    missing = [name for name in clf.features if name not in X.columns]
    if missing:
        raise FeatureSchemaError(
            f"Classifier is missing required features: {missing}"
        )
    probability = float(clf.pipeline.predict_proba(X[clf.features])[0, 1])

    tiers: dict[str, TierPrediction] = {}
    for reg in registry.regressors:
        raw = float(reg.pipeline.predict(X[reg.features])[0])
        thresholds = registry.calibration["regressor_tiers"][reg.name]["thresholds"]
        tiers[reg.tier_name] = TierPrediction(
            model=reg.name,
            tier_name=reg.tier_name,
            tier=tier_for(raw, thresholds),
            raw_value=round(raw, 6),
            low_confidence=reg.low_confidence,
        )

    return Predictions(
        top_quartile_probability=round(probability, 4),
        tiers=tiers,
    )
