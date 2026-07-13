"""Primary classifier: top-quartile-for-creator (PRD 12.1 / 12.3).

The headline user-facing model. It predicts `top_quartile_for_creator` from only
video-derived features and outputs `predicted_top_quartile_probability` - the
probability that an upload resembles videos that were top-quartile performers for
their own creators in the training data.

Model choice (PRD 12.3): `RandomForestClassifier` - simple, explainable (feature
importances drive the report's strong/weak signals), robust to mixed feature
scales, and a sensible default for ~hundreds of rows. Configured for a small,
imbalanced dataset (~25% positive):
- `class_weight="balanced"` so the minority (top-quartile) class isn't ignored,
- `min_samples_leaf` > 1 and `max_features="sqrt"` to limit overfitting,
- a fixed `random_state` for reproducibility.

Two ways to use it:
- `build_classifier_pipeline(X)` -> a `Pipeline(preprocess, model)` whose
  preprocessor refits inside each CV fold (leakage-free evaluation, Step 6).
- `build_classifier()` -> the bare estimator, trained on the shared
  preprocessor's transformed matrix for the final saved artifact (Step 7).
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from backend.training.preprocessing import build_preprocessor_for

# Recorded in training_metadata.json (Step 7) for reproducibility.
CLASSIFIER_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


def build_classifier(oob_score: bool = False, **overrides) -> RandomForestClassifier:
    """Construct the configured RandomForest classifier.

    `oob_score=True` enables the out-of-bag estimate (a free internal validation
    score from bootstrap sampling); kept off by default since GroupKFold is the
    real generalization metric.
    """
    params = {**CLASSIFIER_PARAMS, **overrides}
    if oob_score:
        params["oob_score"] = True
        params["bootstrap"] = True
    return RandomForestClassifier(**params)


def build_classifier_pipeline(X: pd.DataFrame, **overrides) -> Pipeline:
    """Preprocess + classifier as one estimator (preprocessor refits per fold)."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor_for(X)),
            ("model", build_classifier(**overrides)),
        ]
    )
