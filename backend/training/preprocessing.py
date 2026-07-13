"""Preprocessing pipeline + persisted artifacts (PRD 12.5 / 13).

Builds the sklearn transformer that turns the leakage-safe feature matrix into a
clean numeric array the models can train/predict on, and that the inference app
reloads so training and serving preprocess identically.

Per PRD 12.5 we split columns by type:
- **numeric** features -> median imputation, then standard scaling.
- **boolean / flag** features (values in {0, 1}: `has_audio`, `text_present_*`,
  and the detection-failure indicators like `ocr_failed`) -> zero-fill, no
  scaling. Absence is meaningful (0), and the explicit `*_failed` / `*_missing`
  flags already encode "detection failed" vs "feature absent", so we don't add
  sklearn missing-indicators on top.

Columns are classified by their observed values at fit time and the split is
recorded in `feature_schema.json`, so inference applies the exact same handling.

Artifacts written (PRD 13):
- `preprocessor.pkl`   - the fitted ColumnTransformer used at inference.
- `imputer.pkl`        - the fitted numeric median imputer (component).
- `scaler.pkl`         - the fitted numeric standard scaler (component).
- `feature_schema.json`- the exact input feature order + numeric/boolean split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.core.config import settings

_PREPROCESSOR_FILE = "preprocessor.pkl"
_IMPUTER_FILE = "imputer.pkl"
_SCALER_FILE = "scaler.pkl"
_SCHEMA_FILE = "feature_schema.json"

_NUMERIC = "numeric"
_BOOLEAN = "boolean"


def classify_features(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split columns into (numeric, boolean) by their observed values.

    A column whose non-null values are all in {0, 1} is treated as boolean/flag;
    everything else is numeric. Original column order is preserved within each
    group so the schema is deterministic.
    """
    numeric: list[str] = []
    boolean: list[str] = []
    for col in X.columns:
        values = {float(v) for v in X[col].dropna().unique()}
        if values and values <= {0.0, 1.0}:
            boolean.append(col)
        else:
            numeric.append(col)
    return numeric, boolean


def build_preprocessor_for(X: pd.DataFrame) -> ColumnTransformer:
    """Build an *unfitted* preprocessor for a feature frame (classify + assemble).

    Used to drop a fresh preprocessor into a model Pipeline so it is refit inside
    each CV fold (no preprocessing leakage). Column classification uses only
    column semantics, not learned parameters, so doing it up front is safe.
    """
    return build_preprocessor(*classify_features(X))


def build_preprocessor(numeric: list[str], boolean: list[str]) -> ColumnTransformer:
    """ColumnTransformer: median-impute+scale numeric, zero-fill boolean."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    boolean_imputer = SimpleImputer(strategy="constant", fill_value=0)
    return ColumnTransformer(
        transformers=[
            (_NUMERIC, numeric_pipeline, numeric),
            (_BOOLEAN, boolean_imputer, boolean),
        ],
        remainder="drop",
    )


@dataclass
class FittedPreprocessor:
    """A fitted preprocessor plus the schema describing its expected inputs."""

    preprocessor: ColumnTransformer
    schema: dict

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the fitted preprocessing to a feature frame (aligned by schema)."""
        return self.preprocessor.transform(_align(X, self.schema["features"]))

    @property
    def output_features(self) -> list[str]:
        """Column order of the transformed array (numeric block, then boolean)."""
        return self.schema["transformed_feature_order"]


def _align(X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Reorder/select columns to the schema's expected feature order."""
    return X.reindex(columns=features)


def fit_preprocessor(X: pd.DataFrame) -> FittedPreprocessor:
    """Classify columns, build, and fit the preprocessor on a feature frame."""
    numeric, boolean = classify_features(X)
    preprocessor = build_preprocessor(numeric, boolean)
    preprocessor.fit(X)
    schema = {
        "features": list(X.columns),
        "numeric_features": numeric,
        "boolean_features": boolean,
        # ColumnTransformer concatenates outputs in transformer order.
        "transformed_feature_order": numeric + boolean,
    }
    return FittedPreprocessor(preprocessor=preprocessor, schema=schema)


def save_preprocessing_artifacts(
    fitted: FittedPreprocessor, models_dir: str | Path | None = None
) -> dict[str, Path]:
    """Persist preprocessor + component imputer/scaler + feature schema (PRD 13)."""
    models_dir = Path(models_dir or settings.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    preprocessor_path = models_dir / _PREPROCESSOR_FILE
    joblib.dump(fitted.preprocessor, preprocessor_path)

    # Surface the numeric imputer/scaler as separate files per the artifact list.
    numeric_pipeline = fitted.preprocessor.named_transformers_[_NUMERIC]
    joblib.dump(numeric_pipeline.named_steps["imputer"], models_dir / _IMPUTER_FILE)
    joblib.dump(numeric_pipeline.named_steps["scaler"], models_dir / _SCALER_FILE)

    schema_path = models_dir / _SCHEMA_FILE
    schema_path.write_text(json.dumps(fitted.schema, indent=2), encoding="utf-8")

    return {
        "preprocessor": preprocessor_path,
        "imputer": models_dir / _IMPUTER_FILE,
        "scaler": models_dir / _SCALER_FILE,
        "feature_schema": schema_path,
    }


def load_feature_schema(models_dir: str | Path | None = None) -> dict:
    """Load the saved feature schema (the inference feature contract, PRD 13.1)."""
    models_dir = Path(models_dir or settings.models_dir)
    return json.loads((models_dir / _SCHEMA_FILE).read_text(encoding="utf-8"))


def load_preprocessor(models_dir: str | Path | None = None) -> FittedPreprocessor:
    """Reload the fitted preprocessor + schema for inference."""
    models_dir = Path(models_dir or settings.models_dir)
    preprocessor = joblib.load(models_dir / _PREPROCESSOR_FILE)
    schema = load_feature_schema(models_dir)
    return FittedPreprocessor(preprocessor=preprocessor, schema=schema)
