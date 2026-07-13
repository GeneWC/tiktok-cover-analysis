"""Secondary regressors (PRD 12.2 / 12.3).

RandomForestRegressor for the continuous creator-relative targets
(`engagement_rate`, `share_rate`, `creator_relative_log_views`). Their
predictions are later bucketed into tiers (low / medium / medium_high / high)
rather than shown as exact numbers (PRD 12.2 / 14.4).

Mirrors `classifier.py`: same regularized RandomForest settings for a small
dataset, a bare estimator for the final artifact, and a `Pipeline(preprocess,
model)` builder whose preprocessor refits inside each CV fold (leakage-free
evaluation).
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from backend.training.preprocessing import build_preprocessor_for

# Recorded in training_metadata.json (Step 7) for reproducibility.
REGRESSOR_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}


def build_regressor(**overrides) -> RandomForestRegressor:
    """Construct the configured RandomForest regressor."""
    return RandomForestRegressor(**{**REGRESSOR_PARAMS, **overrides})


def build_regressor_pipeline(X: pd.DataFrame, **overrides) -> Pipeline:
    """Preprocess + regressor as one estimator (preprocessor refits per fold)."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor_for(X)),
            ("model", build_regressor(**overrides)),
        ]
    )
