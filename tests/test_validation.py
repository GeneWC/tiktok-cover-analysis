"""Tests for validation strategies (PRD 12.7)."""

from __future__ import annotations

import numpy as np
import pytest

from backend.training.validation import (
    INSUFFICIENT_CREATORS_MESSAGE,
    group_cv_splits,
    plan_group_cv,
    random_split_indices,
)


def _groups(n_per_creator=10, n_creators=8):
    return np.repeat([f"creator_{i}" for i in range(n_creators)], n_per_creator)


def test_plan_clamps_splits_to_creator_count():
    plan = plan_group_cv(_groups(n_creators=8), n_splits=5)
    assert plan.available and plan.n_creators == 8 and plan.n_splits == 5

    plan = plan_group_cv(_groups(n_creators=3), n_splits=5)
    assert plan.available and plan.n_splits == 3  # clamped to #creators


def test_plan_flags_insufficient_creators():
    plan = plan_group_cv(_groups(n_creators=1))
    assert not plan.available
    assert plan.message == INSUFFICIENT_CREATORS_MESSAGE


def test_group_splits_keep_creators_disjoint():
    groups = _groups(n_creators=8)
    splits = group_cv_splits(groups, n_splits=5)
    assert len(splits) == 5

    all_test_creators: set[str] = set()
    for train_idx, test_idx in splits:
        train_creators = set(groups[train_idx])
        test_creators = set(groups[test_idx])
        # no creator appears in both train and test of a fold (no leakage)
        assert train_creators.isdisjoint(test_creators)
        all_test_creators |= test_creators
    # every creator is held out exactly once across the folds
    assert all_test_creators == set(groups)


def test_group_splits_raise_when_too_few_creators():
    with pytest.raises(ValueError, match="insufficient creator count"):
        group_cv_splits(_groups(n_creators=1))


def test_random_split_is_deterministic_and_disjoint():
    a = random_split_indices(100, seed=42)
    b = random_split_indices(100, seed=42)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])

    train_idx, test_idx = a
    assert set(train_idx).isdisjoint(test_idx)
    assert len(train_idx) + len(test_idx) == 100
    assert len(test_idx) == 20  # default test_size=0.2


def test_random_split_stratifies_on_labels():
    labels = np.array([1] * 25 + [0] * 75)  # 25% positive
    train_idx, test_idx = random_split_indices(100, stratify_labels=labels, test_size=0.2)
    # stratification preserves the positive rate in the test fold
    assert labels[test_idx].mean() == pytest.approx(0.25)
