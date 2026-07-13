"""Per-model training specifications (PRD 12.1 / 12.2).

Declares each model the trainer builds: its target, task, the feature-group
subset it consumes, its artifact filename, and whether it is a low-confidence
signal. Feature subsets are driven by the Phase-4 signal-search results:

- `top_quartile` (primary classifier) uses framing + visual - the subset that
  gave the best honest cross-creator ranking (all-features was ~chance).
- `engagement` (regressor) uses framing + visual + audio - the target with the
  strongest, most stable generalization (LOCO R^2 ~ 0.10, AUC ~ 0.64 as a class).
- `creator_relative` and `shareability` regressors keep all features but are
  flagged `low_confidence`: cross-creator signal was ~chance / negative R^2, so
  the report must present their tiers as exploratory (PRD 16.9).

Keeping this as declarative data (not scattered constants) means the trainer,
the artifact metadata, and inference all read the same source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """One model to train + persist."""

    name: str                       # short key (also used in feature_schema)
    target: str                     # label column to predict
    task: str                       # "classification" | "regression"
    artifact: str                   # .pkl filename under models_dir
    feature_groups: tuple[str, ...] = ()   # empty = all features
    low_confidence: bool = False    # weak cross-creator signal (exploratory tier)
    tier_name: str | None = None    # report tier this model feeds, if any


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="top_quartile",
        target="top_quartile_for_creator",
        task="classification",
        artifact="top_quartile_classifier.pkl",
        feature_groups=("framing", "visual"),
    ),
    ModelSpec(
        name="engagement",
        target="engagement_rate",
        task="regression",
        artifact="engagement_model.pkl",
        feature_groups=("framing", "visual", "audio"),
        tier_name="engagement_tier",
    ),
    ModelSpec(
        name="creator_relative",
        target="creator_relative_log_views",
        task="regression",
        artifact="creator_relative_regressor.pkl",
        feature_groups=(),  # all features
        low_confidence=True,
        tier_name="view_performance_tier",
    ),
    ModelSpec(
        name="shareability",
        target="share_rate",
        task="regression",
        artifact="shareability_model.pkl",
        feature_groups=(),  # all features
        low_confidence=True,
        tier_name="shareability_tier",
    ),
)

# Convenience lookups.
SPECS_BY_NAME: dict[str, ModelSpec] = {spec.name: spec for spec in MODEL_SPECS}
PRIMARY_SPEC: ModelSpec = SPECS_BY_NAME["top_quartile"]
