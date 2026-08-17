"""Preprocessing must be fit on training rows only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.training.model_dataset import ModelDataset, subset_dataset
from backend.training.preprocessing import fit_preprocessor


def test_scaler_statistics_ignore_held_out_rows():
    train = pd.DataFrame({"num": [0.0, 2.0, 4.0], "flag": [0.0, 1.0, 0.0]})
    full = pd.DataFrame(
        {"num": [0.0, 2.0, 4.0, 100.0], "flag": [0.0, 1.0, 0.0, 1.0]}
    )
    fitted_train = fit_preprocessor(train)
    fitted_full = fit_preprocessor(full)
    scaler_train = fitted_train.preprocessor.named_transformers_["numeric"].named_steps[
        "scaler"
    ]
    scaler_full = fitted_full.preprocessor.named_transformers_["numeric"].named_steps[
        "scaler"
    ]
    assert abs(scaler_train.mean_[0] - 2.0) < 1e-9
    assert scaler_full.mean_[0] > scaler_train.mean_[0]


def test_subset_dataset_preserves_feature_contract():
    frame = pd.DataFrame(
        {
            "creator_username": ["a", "a", "b", "b"],
            "num": [1.0, 2.0, 3.0, 4.0],
        }
    )
    X = frame[["num"]]
    dataset = ModelDataset(frame=frame, X=X, feature_names=["num"])
    subset = subset_dataset(dataset, np.array([True, True, False, False]))
    assert len(subset.frame) == 2
    assert subset.feature_names == ["num"]
    assert set(subset.groups) == {"a"}
