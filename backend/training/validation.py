"""Validation strategies (PRD 12.7).

Two evaluation modes the trainer uses:

- **Random train/test split** - quick sanity/debugging. For the classifier we
  stratify on the label so both splits keep the (~25%) positive rate.
- **GroupKFold by creator** - the *preferred* generalization estimate. Folds are
  split so a creator's videos are entirely in train or entirely in test, never
  both. This prevents the model from "memorizing" a creator and then being graded
  on that same creator (creator leakage), which would inflate scores.

Guard (PRD 12.7): cross-creator CV needs at least two creators. With too few, the
trainer should report `INSUFFICIENT_CREATORS_MESSAGE` and skip grouped CV instead
of crashing. GroupKFold also can't use more folds than there are creators, so the
requested split count is clamped to the creator count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold, train_test_split

DEFAULT_N_SPLITS = 5
MIN_CREATORS_FOR_GROUP_CV = 2
INSUFFICIENT_CREATORS_MESSAGE = (
    "Cross-creator validation unavailable due to insufficient creator count."
)

# (train_indices, test_indices) into the row axis of X / y / groups.
Split = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class GroupCVPlan:
    """Whether grouped CV is possible for a dataset, and at how many folds."""

    available: bool
    n_creators: int
    n_splits: int
    message: str | None = None


def plan_group_cv(groups, n_splits: int = DEFAULT_N_SPLITS) -> GroupCVPlan:
    """Decide if GroupKFold-by-creator is possible and how many folds to use."""
    n_creators = int(np.unique(np.asarray(groups)).size)
    if n_creators < MIN_CREATORS_FOR_GROUP_CV:
        return GroupCVPlan(
            available=False,
            n_creators=n_creators,
            n_splits=0,
            message=INSUFFICIENT_CREATORS_MESSAGE,
        )
    # GroupKFold requires n_splits <= number of groups.
    return GroupCVPlan(
        available=True,
        n_creators=n_creators,
        n_splits=min(n_splits, n_creators),
    )


def group_cv_splits(groups, n_splits: int = DEFAULT_N_SPLITS) -> list[Split]:
    """GroupKFold splits keyed by creator (no creator spans train and test).

    Raises ValueError if there are too few creators for grouped CV. The fold
    count is clamped to the number of creators.
    """
    groups = np.asarray(groups)
    plan = plan_group_cv(groups, n_splits)
    if not plan.available:
        raise ValueError(INSUFFICIENT_CREATORS_MESSAGE)

    splitter = GroupKFold(n_splits=plan.n_splits)
    placeholder = np.zeros(len(groups))  # GroupKFold only needs the row count
    return list(splitter.split(placeholder, groups=groups))


def random_split_indices(
    n_samples: int,
    stratify_labels=None,
    test_size: float = 0.2,
    seed: int = 42,
) -> Split:
    """Deterministic random train/test index split.

    Pass `stratify_labels` (the classification target) to preserve class balance
    across the split; omit it for regression targets.
    """
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_labels,
    )
    return train_idx, test_idx
