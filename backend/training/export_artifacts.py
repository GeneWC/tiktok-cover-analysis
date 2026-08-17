"""Trained-model artifact export (PRD 13 / 18.2).

Fits every model on all its available rows and writes the deployable artifacts
the inference app loads:

- `<model>.pkl` (one per spec) - a self-contained `Pipeline(preprocess, model)`.
  Each pipeline embeds its own imputer/scaler over its feature subset, so
  training and serving preprocess identically and inference just calls
  `pipeline.predict[/_proba]` on the full feature frame (the pipeline selects the
  columns it needs). This is why we don't ship a separate global preprocessor.
- `feature_schema.json` - the inference contract: the full feature order the
  extractor must produce, plus, per model, its feature subset / target / task /
  tier / confidence. Serving reads this to build inputs and route outputs.
- `feature_importances.json` - per-model RandomForest importances (descending),
  which drive the report's strong/weak-signal explanations (PRD 16).
- `training_metadata.json` - reproducibility record: timestamp, dataset, library
  versions, model hyperparameters, per-model row counts, and the cross-creator
  evaluation metrics from `evaluate.py`.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

from backend.core.config import settings
from backend.training.calibration import compute_calibration
from backend.training.classifier import CLASSIFIER_PARAMS
from backend.training.evaluate import EvaluationResult
from backend.training.model_dataset import GROUP_COLUMN, ModelDataset
from backend.training.model_specs import MODEL_SPECS, ModelSpec
from backend.training.regressor import REGRESSOR_PARAMS
from backend.training.reproducibility import seed_record
from backend.training.train_models import FittedModel, train_all_models
from backend.training.validation import DEFAULT_N_SPLITS

# Bump when the inference feature contract changes meaning or order.
SCHEMA_VERSION = 1


def feature_fingerprint(names: list[str]) -> str:
    """Stable short hash of the ordered feature-name contract."""
    payload = "\n".join(names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]

FEATURE_SCHEMA_FILE = "feature_schema.json"
IMPORTANCES_FILE = "feature_importances.json"
METADATA_FILE = "training_metadata.json"
CALIBRATION_FILE = "calibration.json"

_PREPROCESS_BLOCKS = ("numeric", "boolean")


def _transformed_feature_order(pipeline) -> list[str]:
    """Feature names in the order the fitted preprocessor emits them."""
    pre = pipeline.named_steps["preprocess"]
    order: list[str] = []
    for name, _, cols in pre.transformers_:
        if name in _PREPROCESS_BLOCKS:
            order.extend(cols)
    return order


def _model_importances(fitted: FittedModel) -> dict[str, float]:
    """RandomForest feature importances mapped to feature names, descending."""
    order = _transformed_feature_order(fitted.pipeline)
    importances = fitted.pipeline.named_steps["model"].feature_importances_
    pairs = sorted(zip(order, importances.tolist()), key=lambda p: p[1], reverse=True)
    return {name: round(value, 6) for name, value in pairs}


def _schema_entry(fitted: FittedModel) -> dict:
    spec = fitted.spec
    return {
        "target": spec.target,
        "task": spec.task,
        "artifact": spec.artifact,
        "feature_groups": list(spec.feature_groups),
        "features": fitted.features,
        "low_confidence": spec.low_confidence,
        "tier_name": spec.tier_name,
        "n_samples": fitted.n_samples,
    }


def _metadata(
    fitted_models: dict[str, FittedModel],
    evaluations: dict[str, EvaluationResult] | None,
) -> dict:
    evaluations = evaluations or {}
    models_meta = {}
    for name, fitted in fitted_models.items():
        result = evaluations.get(name)
        models_meta[name] = {
            "target": fitted.spec.target,
            "task": fitted.spec.task,
            "n_samples": fitted.n_samples,
            "n_features": len(fitted.features),
            "low_confidence": fitted.spec.low_confidence,
            "metrics": (result.metrics or None) if result else None,
            "evaluation_message": result.message if result else None,
        }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_dataset": str(settings.training_dataset_csv),
        "sklearn_version": sklearn.__version__,
        "validation": {
            "scheme": "GroupKFold by creator",
            "group_column": GROUP_COLUMN,
            "n_splits": DEFAULT_N_SPLITS,
        },
        "classifier_params": CLASSIFIER_PARAMS,
        "regressor_params": REGRESSOR_PARAMS,
        "reproducibility": seed_record(),
        "models": models_meta,
    }


def export_models(
    dataset: ModelDataset,
    out_dir: str | Path | None = None,
    evaluations: dict[str, EvaluationResult] | None = None,
    specs: tuple[ModelSpec, ...] = MODEL_SPECS,
    fit_dataset: ModelDataset | None = None,
    calibration_dataset: ModelDataset | None = None,
) -> dict[str, Path]:
    """Fit all models on available rows and write every artifact (PRD 13).

    ``fit_dataset`` / ``calibration_dataset`` let callers restrict fit and
    threshold estimation to train (or train+val) creators. Default remains
    the full dataset so existing artifacts and tests stay unchanged.
    """
    out_dir = Path(out_dir or settings.models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = fit_dataset or dataset
    calib_ds = calibration_dataset or train_ds
    fitted_models = train_all_models(train_ds, specs)

    schema = {
        "schema_version": SCHEMA_VERSION,
        "feature_fingerprint": feature_fingerprint(list(dataset.feature_names)),
        "all_features": list(dataset.feature_names),
        "group_column": GROUP_COLUMN,
        "models": {},
    }
    importances = {}
    written: dict[str, Path] = {}

    for name, fitted in fitted_models.items():
        artifact_path = out_dir / fitted.spec.artifact
        joblib.dump(fitted.pipeline, artifact_path)
        written[name] = artifact_path
        schema["models"][name] = _schema_entry(fitted)
        importances[name] = _model_importances(fitted)

    calibration = compute_calibration(calib_ds, fitted_models)

    schema_path = out_dir / FEATURE_SCHEMA_FILE
    importances_path = out_dir / IMPORTANCES_FILE
    metadata_path = out_dir / METADATA_FILE
    calibration_path = out_dir / CALIBRATION_FILE
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    importances_path.write_text(json.dumps(importances, indent=2), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(_metadata(fitted_models, evaluations), indent=2), encoding="utf-8"
    )
    calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    written["feature_schema"] = schema_path
    written["feature_importances"] = importances_path
    written["training_metadata"] = metadata_path
    written["calibration"] = calibration_path
    return written


def load_model_schema(out_dir: str | Path | None = None) -> dict:
    """Load the inference feature/model contract (PRD 13.1)."""
    out_dir = Path(out_dir or settings.models_dir)
    return json.loads((out_dir / FEATURE_SCHEMA_FILE).read_text(encoding="utf-8"))


def load_calibration(out_dir: str | Path | None = None) -> dict:
    """Load the serving calibration (tier thresholds + feature percentiles)."""
    out_dir = Path(out_dir or settings.models_dir)
    return json.loads((out_dir / CALIBRATION_FILE).read_text(encoding="utf-8"))
