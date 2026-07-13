"""Serving calibration artifact (PRD 12.2 / 14.4 / 15).

The models output raw numbers, but the report shows **tiers**
(low/medium/medium_high/high) and **0-100 presentation scores**. Both need
reference points derived from the training population - computed once here and
saved so serving never invents thresholds at request time.

Two things are calibrated:

- **Regressor tier thresholds.** RandomForest regression predictions regress
  toward the mean, so bucketing them by the *target's* quantiles would pile
  almost everything into the middle. Instead we bucket by quantiles of each
  model's *predicted* distribution over the training videos (its peer group):
  the 25/50/75 percentiles become the low | medium | medium_high | high cut
  points, giving a balanced, meaningful spread.

- **Per-feature percentiles.** For every feature we store the 5/25/50/75/95
  percentiles from the training feature matrix. Step 4 turns these into 0-100
  presentation subscores (normalize a video's feature against its training
  spread) and Step 5 uses them to describe a feature as high/typical/low
  relative to peers - all from one saved distribution, so train and serve agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.training.model_dataset import ModelDataset
from backend.training.train_models import FittedModel

# Report tiers, worst -> best (PRD 12.2). Index = number of thresholds cleared.
TIER_ORDER: tuple[str, ...] = ("low", "medium", "medium_high", "high")

# Quantiles (percent) for tier cut points and for feature normalization.
_TIER_QUANTILES = (25, 50, 75)
_FEATURE_QUANTILES = (5, 25, 50, 75, 95)


def tier_for(value: float, thresholds: dict[str, float]) -> str:
    """Map a predicted value to a tier using stored q25/q50/q75 cut points."""
    cutoffs = (thresholds["q25"], thresholds["q50"], thresholds["q75"])
    rank = sum(value >= cut for cut in cutoffs)  # 0..3 -> tier index
    return TIER_ORDER[rank]


def _tier_thresholds(fitted: FittedModel, X: pd.DataFrame) -> dict[str, float]:
    """25/50/75 percentiles of a regressor's predictions over training videos."""
    predictions = fitted.pipeline.predict(X[fitted.features])
    q25, q50, q75 = np.percentile(predictions, _TIER_QUANTILES)
    return {"q25": float(q25), "q50": float(q50), "q75": float(q75)}


def _feature_percentiles(X: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-feature training percentiles (NaN-aware), for later normalization."""
    out: dict[str, dict[str, float]] = {}
    for col in X.columns:
        values = X[col].to_numpy(dtype=float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        percentiles = np.percentile(values, _FEATURE_QUANTILES)
        out[col] = {
            f"p{q}": float(round(p, 6))
            for q, p in zip(_FEATURE_QUANTILES, percentiles)
        }
    return out


def compute_calibration(
    dataset: ModelDataset, fitted_models: dict[str, FittedModel]
) -> dict:
    """Build the calibration payload from fitted models + training features."""
    regressor_tiers: dict[str, dict] = {}
    for name, fitted in fitted_models.items():
        if fitted.spec.task != "regression":
            continue
        regressor_tiers[name] = {
            "target": fitted.spec.target,
            "tier_name": fitted.spec.tier_name,
            "low_confidence": fitted.spec.low_confidence,
            "thresholds": _tier_thresholds(fitted, dataset.X),
        }
    return {
        "tier_order": list(TIER_ORDER),
        "regressor_tiers": regressor_tiers,
        "feature_percentiles": _feature_percentiles(dataset.X),
    }
