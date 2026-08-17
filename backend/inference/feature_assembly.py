"""Single-video feature assembly for inference (PRD 6.2 / 12.4).

Bridges the Phase-2 extractor and the trained models: runs `extract_all_features`
on an uploaded video and reshapes its flat feature dict into the exact input the
pipelines expect - a one-row DataFrame whose columns are the schema's
`all_features`, in order, coerced to numeric with the *same* conventions used to
build the training matrix (bool -> 1/0, missing/blank -> NaN). Imputation and
scaling happen inside each model pipeline, so we only need consistent columns and
types here.

Also surfaces the per-group extraction `steps` (for the job status map) and
`has_audio` (so the orchestrator can mark audio/ocr skipped rather than failed
when there's simply no audio track).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backend.features.extract_features import (
    FrameSample,
    extract_all_features,
)
from backend.inference.model_registry import ModelRegistry, get_registry
from backend.training.export_artifacts import SCHEMA_VERSION, feature_fingerprint
from backend.training.model_dataset import _coerce_numeric


class FeatureSchemaError(ValueError):
    """Raised when an assembled vector does not match the loaded model contract."""


@dataclass
class AssembledFeatures:
    """A schema-aligned feature row plus extraction context for one video."""

    X: pd.DataFrame                 # 1 row, columns = all_features (schema order)
    raw: dict[str, object]          # raw extractor output (native types)
    steps: dict[str, str]           # per-group extraction status
    has_audio: bool
    frames_sampled: int
    duration_seconds: float | None

    @property
    def usable(self) -> bool:
        """True if frames were decoded (a completely undecodable file is not)."""
        return self.frames_sampled > 0


def assert_feature_schema_compatible(
    X,
    expected: list[str],
    *,
    schema_version: int | None = None,
    expected_fingerprint: str | None = None,
) -> None:
    """Reject a feature frame that does not match the serving contract.

    ColumnTransformer pipelines select by name, but a renamed or reordered
    contract can still silently impute missing values. Fail fast instead.
    """
    cols = list(X.columns)
    expected = list(expected)
    if cols != expected:
        missing = [c for c in expected if c not in cols]
        extra = [c for c in cols if c not in expected]
        raise FeatureSchemaError(
            "Feature schema mismatch: column names or order differ from the "
            f"loaded model (missing={missing or 'none'}, extra={extra or 'none'})."
        )
    if schema_version is not None and schema_version != SCHEMA_VERSION:
        raise FeatureSchemaError(
            f"Feature schema version {schema_version} is not supported "
            f"(expected {SCHEMA_VERSION}). Retrain or regenerate artifacts."
        )
    if expected_fingerprint:
        actual = feature_fingerprint(cols)
        if actual != expected_fingerprint:
            raise FeatureSchemaError(
                "Feature fingerprint does not match the loaded model contract. "
                "The extractor and the trained pipelines disagree on feature names."
            )


def to_feature_frame(
    features: dict[str, object], feature_order: list[str]
) -> pd.DataFrame:
    """Reshape a raw feature dict into a 1-row numeric frame in schema order.

    Any feature the schema expects but the extractor didn't emit becomes NaN, so
    the column set always matches what the pipelines were trained on.
    """
    row = {name: features.get(name) for name in feature_order}
    frame = pd.DataFrame([row], columns=list(feature_order))
    return pd.DataFrame(
        {name: _coerce_numeric(frame[name]) for name in feature_order},
        index=frame.index,
    )


def assemble_features(
    video_path: str,
    registry: ModelRegistry | None = None,
    sample: FrameSample | None = None,
) -> AssembledFeatures:
    """Extract features for one video and align them to the model input schema."""
    registry = registry or get_registry()
    result = extract_all_features(video_path, sample=sample)

    X = to_feature_frame(result.features, registry.all_features)
    assert_feature_schema_compatible(
        X,
        registry.all_features,
        schema_version=getattr(registry, "schema_version", None),
        expected_fingerprint=getattr(registry, "feature_fingerprint", None) or None,
    )
    return AssembledFeatures(
        X=X,
        raw=result.features,
        steps=result.steps,
        has_audio=bool(result.features.get("has_audio")),
        frames_sampled=result.frames_sampled,
        duration_seconds=result.duration_seconds,
    )
