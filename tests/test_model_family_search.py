"""Unit tests for Exp C1 model-family search helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.training.creator_splits import make_creator_splits
from backend.training.feature_groups import select_group_features
from backend.training.model_dataset import ModelDataset
from backend.training.model_family_search import (
    flag_promising,
    resolve_feature_set,
    run_model_family_search,
    select_early_window_features,
)
from backend.training.model_specs import PRIMARY_SPEC


def _toy_dataset(n_creators: int = 12, n_per: int = 10) -> ModelDataset:
    rows = []
    rng = np.random.RandomState(1)
    for c in range(n_creators):
        for i in range(n_per):
            bright = rng.random()
            rows.append(
                {
                    "creator_username": f"creator_{c}",
                    "video_id": f"{c}_{i}",
                    "top_quartile_for_creator": "true" if bright > 0.7 else "false",
                    "creator_relative_log_views": str(bright - 0.5),
                    "engagement_rate": str(0.05),
                    "share_rate": str(0.001),
                    "brightness_mean_full": str(bright),
                    "contrast_full": str(rng.random()),
                    "sharpness_full": str(rng.random()),
                    "face_visible_ratio": str(rng.random()),
                    "subject_centering_score": str(rng.random()),
                    "brightness_mean_first_3s": str(bright * 0.9),
                    "contrast_first_3s": str(rng.random()),
                    "sharpness_first_1s": str(rng.random()),
                    "person_visible_ratio": str(rng.random()),
                    "face_visible_ratio_first_3s": str(rng.random()),
                    "hand_visible_ratio": str(rng.random()),
                    "subject_size_ratio": str(rng.random()),
                    "face_size_ratio": str(rng.random()),
                    "audio_rms_mean": str(rng.random()),
                    "audio_energy_first_3s": str(rng.random()),
                    "motion_energy_mean": str(rng.random()),
                    "motion_energy_first_1s": str(rng.random()),
                    "colorfulness_full": str(rng.random()),
                }
            )
    frame = pd.DataFrame(rows)
    feature_names = [
        c
        for c in frame.columns
        if c
        not in {
            "creator_username",
            "video_id",
            "top_quartile_for_creator",
            "creator_relative_log_views",
            "engagement_rate",
            "share_rate",
        }
    ]
    X = frame[feature_names].apply(pd.to_numeric, errors="coerce")
    return ModelDataset(frame=frame, X=X, feature_names=feature_names)


def test_select_early_window_features():
    feats = [
        "brightness_mean_full",
        "brightness_mean_first_3s",
        "motion_energy_first_1s",
        "audio_rms_mean",
    ]
    early = select_early_window_features(feats)
    assert early == ["brightness_mean_first_3s", "motion_energy_first_1s"]


def test_resolve_feature_set_aliases():
    dataset = _toy_dataset()
    all_f = dataset.feature_names
    control = resolve_feature_set(all_f, "framing+visual")
    assert control == select_group_features(all_f, PRIMARY_SPEC.feature_groups)
    assert resolve_feature_set(all_f, "all") == all_f
    audio = resolve_feature_set(all_f, "audio-only")
    assert audio and all("audio" in f for f in audio)
    mv = resolve_feature_set(all_f, "motion+visual")
    assert mv
    early = resolve_feature_set(all_f, "early-window-only")
    assert early == select_early_window_features(all_f)


def test_flag_promising_margin():
    assert flag_promising(0.60, 0.55, 0.50) is True
    assert flag_promising(0.569, 0.55, 0.50) is False  # < +0.02
    assert flag_promising(0.60, 0.55, 0.61) is False  # loses to simple logistic
    assert flag_promising(None, 0.55, 0.50) is False


def test_run_model_family_search_val_only_toy():
    dataset = _toy_dataset()
    membership = make_creator_splits(dataset.groups, seed=42)
    rows = run_model_family_search(dataset, membership, eval_split="val")
    names = {r.name for r in rows}
    assert "rf_control" in names
    assert "logistic_balanced" in names
    assert "hist_gradient_boosting" in names
    assert "rf_regularized" in names
    assert "select_from_model_logistic" in names
    assert any(r.kind == "feature_ablation" for r in rows)
    for r in rows:
        assert r.n_eval > 0
        assert r.message is None
        assert "roc_auc" in r.metrics
