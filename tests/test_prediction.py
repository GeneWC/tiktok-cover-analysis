"""Tests for the prediction service (PRD 12.1 / 12.2 / 16.9)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.inference.model_registry import LoadedModel, ModelRegistry
from backend.inference.prediction import predict


class _StubClassifier:
    def __init__(self, positive_proba: float):
        self._p = positive_proba

    def predict_proba(self, X):
        return np.array([[1.0 - self._p, self._p]])


class _StubRegressor:
    def __init__(self, value: float):
        self._v = value

    def predict(self, X):
        return np.array([self._v])


def _registry(engagement_value: float, share_value: float) -> ModelRegistry:
    models = {
        "top_quartile": LoadedModel(
            "top_quartile", _StubClassifier(0.73), "top_quartile_for_creator",
            "classification", ["f"], False, None,
        ),
        "engagement": LoadedModel(
            "engagement", _StubRegressor(engagement_value), "engagement_rate",
            "regression", ["f"], False, "engagement_tier",
        ),
        "shareability": LoadedModel(
            "shareability", _StubRegressor(share_value), "share_rate",
            "regression", ["f"], True, "shareability_tier",
        ),
    }
    calibration = {
        "regressor_tiers": {
            "engagement": {"thresholds": {"q25": 1.0, "q50": 2.0, "q75": 3.0}},
            "shareability": {"thresholds": {"q25": 0.1, "q50": 0.2, "q75": 0.3}},
        }
    }
    return ModelRegistry(models=models, all_features=["f"], importances={}, calibration=calibration)


def _frame():
    return pd.DataFrame({"f": [0.5]})


def test_predict_returns_probability_and_tiers():
    preds = predict(_frame(), registry=_registry(engagement_value=2.5, share_value=0.05))
    assert preds.top_quartile_probability == 0.73
    assert set(preds.tiers) == {"engagement_tier", "shareability_tier"}


def test_regressor_values_map_to_expected_tiers():
    preds = predict(_frame(), registry=_registry(engagement_value=2.5, share_value=0.05))
    assert preds.tiers["engagement_tier"].tier == "medium_high"  # 2.5 in [q50,q75)
    assert preds.tiers["shareability_tier"].tier == "low"        # 0.05 < q25


def test_low_confidence_flag_propagates():
    preds = predict(_frame(), registry=_registry(engagement_value=0.0, share_value=0.5))
    assert preds.tiers["engagement_tier"].low_confidence is False
    assert preds.tiers["shareability_tier"].low_confidence is True
    assert preds.tiers["shareability_tier"].tier == "high"        # 0.5 >= q75


def test_predict_accepts_assembled_like_object():
    class _Assembled:
        X = _frame()

    preds = predict(_Assembled(), registry=_registry(1.5, 0.15))
    assert preds.tiers["engagement_tier"].tier == "medium"        # 1.5 in [q25,q50)
