"""Spec-driven model training (PRD 12.1 / 12.2).

Fits every model declared in `MODEL_SPECS`. For each spec it:
1. selects the spec's feature-group subset (Phase-4 signal-search choice),
2. drops rows missing that target (per-target exclusion, PRD 8.5),
3. fits a `Pipeline(preprocessor-on-subset, estimator)` on the available rows.

Each fitted model is a self-contained pipeline: it preprocesses (impute/scale)
its own feature subset internally, so inference can hand every model the full
feature frame and each one selects the columns it needs. Evaluation (Step 6) and
artifact export (Step 7) build on the objects returned here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from backend.training.classifier import build_classifier_pipeline
from backend.training.feature_groups import select_group_features
from backend.training.model_dataset import ModelDataset
from backend.training.model_specs import MODEL_SPECS, ModelSpec
from backend.training.regressor import build_regressor_pipeline


@dataclass
class FittedModel:
    """A trained model pipeline plus the spec and data footprint behind it."""

    spec: ModelSpec
    pipeline: Pipeline
    features: list[str]        # the feature-subset columns this model consumes
    n_samples: int             # rows it was trained on (after target exclusion)


def build_pipeline_for_spec(spec: ModelSpec, X: pd.DataFrame) -> Pipeline:
    """Build the (unfitted) pipeline appropriate to a spec's task."""
    if spec.task == "classification":
        return build_classifier_pipeline(X)
    if spec.task == "regression":
        return build_regressor_pipeline(X)
    raise ValueError(f"Unknown task '{spec.task}' for model '{spec.name}'")


def _select_xy(spec: ModelSpec, dataset: ModelDataset):
    """Return (X_subset, y) for a spec: its feature subset + rows with a target."""
    features = select_group_features(dataset.feature_names, spec.feature_groups)
    y = dataset.target(spec.target)
    mask = y.notna().to_numpy()
    X = dataset.X.loc[mask, features]
    y = y[mask]
    if spec.task == "classification":
        y = y.astype(int)
    return features, X, y


def fit_model_for_spec(spec: ModelSpec, dataset: ModelDataset) -> FittedModel:
    """Train one model on its feature subset and available rows."""
    features, X, y = _select_xy(spec, dataset)
    pipeline = build_pipeline_for_spec(spec, X)
    pipeline.fit(X, np.asarray(y))
    return FittedModel(
        spec=spec, pipeline=pipeline, features=features, n_samples=len(X)
    )


def train_all_models(
    dataset: ModelDataset, specs: tuple[ModelSpec, ...] = MODEL_SPECS
) -> dict[str, FittedModel]:
    """Fit every spec's model; returns {spec.name: FittedModel}."""
    return {spec.name: fit_model_for_spec(spec, dataset) for spec in specs}
