"""Loads and caches the trained model artifacts for serving (PRD 13 / 18.2).

Reads everything `scripts/train_models.py` exported - the four fitted pipelines
plus `feature_schema.json`, `feature_importances.json`, and `calibration.json` -
into one in-memory `ModelRegistry`. `get_registry()` caches it so the artifacts
are read from disk once per process, not per request.

The registry is driven entirely by `feature_schema.json` (the inference
contract): serving never imports the training model specs, it just loads what the
schema says was produced. Each pipeline is self-contained (preprocessing embedded
over its own feature subset), so inference hands a model the full feature frame
and the pipeline selects the columns it needs.

If any artifact is missing, loading fails fast with a message pointing at the
training command, rather than surfacing a confusing error deep in a request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from backend.core.config import settings
from backend.training.export_artifacts import (
    CALIBRATION_FILE,
    FEATURE_SCHEMA_FILE,
    IMPORTANCES_FILE,
)

_TRAIN_HINT = "Run `python scripts/train_models.py` to generate model artifacts."


@dataclass(frozen=True)
class LoadedModel:
    """One trained pipeline plus the schema metadata describing its use."""

    name: str
    pipeline: Pipeline
    target: str
    task: str                       # "classification" | "regression"
    features: list[str]             # feature-subset columns this model consumes
    low_confidence: bool
    tier_name: str | None


@dataclass(frozen=True)
class ModelRegistry:
    """All loaded artifacts needed to score an upload."""

    models: dict[str, LoadedModel]
    all_features: list[str]         # full feature order the extractor must produce
    importances: dict               # per-model feature importances (report signals)
    calibration: dict               # tier thresholds + feature percentiles

    @property
    def classifier(self) -> LoadedModel:
        """The primary classification model (top-quartile probability)."""
        return next(m for m in self.models.values() if m.task == "classification")

    @property
    def regressors(self) -> list[LoadedModel]:
        """The tier regressors (engagement / view-performance / shareability)."""
        return [m for m in self.models.values() if m.task == "regression"]


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}. {_TRAIN_HINT}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_registry(models_dir: str | Path | None = None) -> ModelRegistry:
    """Load all artifacts from `models_dir` into a `ModelRegistry` (uncached)."""
    models_dir = Path(models_dir or settings.models_dir)

    schema = _read_json(models_dir / FEATURE_SCHEMA_FILE)
    importances = _read_json(models_dir / IMPORTANCES_FILE)
    calibration = _read_json(models_dir / CALIBRATION_FILE)

    models: dict[str, LoadedModel] = {}
    for name, entry in schema["models"].items():
        artifact_path = models_dir / entry["artifact"]
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Missing model artifact: {artifact_path}. {_TRAIN_HINT}"
            )
        models[name] = LoadedModel(
            name=name,
            pipeline=joblib.load(artifact_path),
            target=entry["target"],
            task=entry["task"],
            features=list(entry["features"]),
            low_confidence=entry["low_confidence"],
            tier_name=entry.get("tier_name"),
        )

    return ModelRegistry(
        models=models,
        all_features=list(schema["all_features"]),
        importances=importances,
        calibration=calibration,
    )


@lru_cache(maxsize=1)
def get_registry() -> ModelRegistry:
    """Process-wide cached registry loaded from the configured models dir."""
    return load_registry()
