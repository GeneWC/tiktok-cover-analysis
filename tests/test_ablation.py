"""Tests for feature-group ablation (val only, never test)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.training.ablation import ablation_feature_sets, run_group_ablation
from backend.training.creator_splits import make_creator_splits
from backend.training.feature_groups import GROUP_NAMES
from backend.training.model_dataset import ModelDataset


def _toy_dataset(n_creators: int = 10, n_per: int = 8) -> ModelDataset:
    rng = np.random.RandomState(1)
    rows = []
    for creator in range(n_creators):
        for i in range(n_per):
            bright = rng.random()
            rows.append(
                {
                    "creator_username": f"c{creator}",
                    "top_quartile_for_creator": "true" if bright > 0.6 else "false",
                    "creator_relative_log_views": str(bright - 0.5),
                    "engagement_rate": str(0.05 * bright),
                    "share_rate": str(0.01 * bright),
                    "brightness_mean_full": str(bright),
                    "contrast_full": str(rng.random()),
                    "sharpness_full": str(rng.random()),
                    "person_visible_ratio": str(rng.random()),
                    "face_visible_ratio": str(rng.random()),
                    "motion_energy_full": str(rng.random()),
                    "camera_stability_score": str(rng.random()),
                    "audio_rms_mean": str(rng.random()),
                    "text_present_anywhere": "0",
                    "duration_seconds": "15",
                }
            )
    frame = pd.DataFrame(rows)
    feature_names = [
        c
        for c in frame.columns
        if c
        not in {
            "creator_username",
            "top_quartile_for_creator",
            "creator_relative_log_views",
            "engagement_rate",
            "share_rate",
        }
    ]
    X = frame[feature_names].apply(pd.to_numeric, errors="coerce")
    return ModelDataset(frame=frame, X=X, feature_names=feature_names)


def test_ablation_sets_cover_groups_and_leave_one_out():
    sets = ablation_feature_sets(_toy_dataset().feature_names)
    assert "all" in sets
    for group in GROUP_NAMES:
        only_key = f"only_{group}"
        minus_key = f"all_minus_{group}"
        if only_key in sets:
            assert minus_key in sets
            assert set(sets[only_key]).isdisjoint(sets[minus_key])


def test_run_group_ablation_scores_val_only():
    dataset = _toy_dataset()
    membership = make_creator_splits(dataset.groups, seed=42)
    results = run_group_ablation(dataset, membership, eval_split="val")
    assert any(r.name == "all" for r in results)
    for result in results:
        assert result.split == "val"
        assert result.n_eval > 0
        assert "roc_auc" in result.metrics


def test_ablation_refuses_test_split():
    dataset = _toy_dataset()
    membership = make_creator_splits(dataset.groups, seed=42)
    with pytest.raises(ValueError, match="test"):
        run_group_ablation(dataset, membership, eval_split="test")
