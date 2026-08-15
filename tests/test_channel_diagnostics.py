"""Tests for channel diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.training.channel_diagnostics import diagnose_channel
from backend.training.creator_residuals import MIN_CREATOR_VIDEOS_FOR_RESIDUALS


def test_diagnostics_requires_min_videos():
    X = pd.DataFrame(
        {
            "brightness_mean_full": [0.1, 0.2],
            "contrast_full": [0.1, 0.2],
            "sharpness_full": [0.1, 0.2],
            "face_visible_ratio": [0.1, 0.2],
            "subject_centering_score": [0.1, 0.2],
        }
    )
    report = diagnose_channel(X, feature_names=list(X.columns))
    assert report.message is not None


def test_diagnostics_hit_miss_deltas():
    rng = np.random.RandomState(0)
    n = MIN_CREATOR_VIDEOS_FOR_RESIDUALS + 3
    bright = np.concatenate([rng.rand(n // 2) * 0.3, 0.7 + rng.rand(n - n // 2) * 0.3])
    labels = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)]).astype(int)
    # shuffle together
    order = rng.permutation(n)
    bright, labels = bright[order], labels[order]
    X = pd.DataFrame(
        {
            "brightness_mean_full": bright,
            "contrast_full": rng.rand(n),
            "sharpness_full": rng.rand(n),
            "face_visible_ratio": rng.rand(n),
            "subject_centering_score": rng.rand(n),
        }
    )
    report = diagnose_channel(X, labels=labels, feature_names=list(X.columns))
    assert report.message is None
    assert report.n_hits == int(labels.sum())
    assert report.top_feature_deltas
    top = report.top_feature_deltas[0]
    assert top.feature == "brightness_mean_full"
    assert top.delta > 0
