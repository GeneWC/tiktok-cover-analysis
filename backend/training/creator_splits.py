"""Durable creator-grouped train / validation / test splits.

Holds out entire creators so no creator's videos appear in more than one split
(see docs/DECISIONS.md D-001). GroupKFold remains available for *within-train*
model selection via `validation.group_cv_splits`.

Default quotas for 24 creators: 14 train / 5 val / 5 test (~60/20/20).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backend.training.model_dataset import GROUP_COLUMN

DEFAULT_SEED = 42
DEFAULT_TRAIN_FRAC = 0.60
DEFAULT_VAL_FRAC = 0.20
DEFAULT_TEST_FRAC = 0.20
DEFAULT_SPLITS_PATH = Path("data/splits/creator_splits.json")


@dataclass(frozen=True)
class CreatorSplitMembership:
    """Creator usernames assigned to each split (disjoint)."""

    seed: int
    train_creators: tuple[str, ...]
    val_creators: tuple[str, ...]
    test_creators: tuple[str, ...]
    train_frac: float = DEFAULT_TRAIN_FRAC
    val_frac: float = DEFAULT_VAL_FRAC
    test_frac: float = DEFAULT_TEST_FRAC
    n_videos: dict[str, int] | None = None  # optional counts per split

    def creators_for(self, split: str) -> tuple[str, ...]:
        if split == "train":
            return self.train_creators
        if split == "val":
            return self.val_creators
        if split == "test":
            return self.test_creators
        raise ValueError(f"Unknown split '{split}' (expected train|val|test)")

    def all_creators(self) -> set[str]:
        return set(self.train_creators) | set(self.val_creators) | set(self.test_creators)

    def assert_disjoint(self) -> None:
        train, val, test = set(self.train_creators), set(self.val_creators), set(self.test_creators)
        if train & val or train & test or val & test:
            raise ValueError("Creator splits are not disjoint")


def creator_quotas(
    n_creators: int,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> tuple[int, int, int]:
    """Exact creator counts for train/val/test (remainder goes to train)."""
    if n_creators < 3:
        raise ValueError("Need at least 3 creators for train/val/test splits")
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must equal 1")

    n_test = max(1, int(round(n_creators * test_frac)))
    n_val = max(1, int(round(n_creators * val_frac)))
    # Keep at least one train creator; shrink val/test if needed.
    while n_test + n_val >= n_creators and (n_test > 1 or n_val > 1):
        if n_test >= n_val and n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break
    n_train = n_creators - n_val - n_test
    if n_train < 1:
        raise ValueError(f"Cannot allocate train/val/test for n_creators={n_creators}")
    return n_train, n_val, n_test


def _video_counts(groups: np.ndarray) -> dict[str, int]:
    creators, counts = np.unique(groups, return_counts=True)
    return {str(c): int(n) for c, n in zip(creators, counts)}


def make_creator_splits(
    groups: np.ndarray | list[str],
    seed: int = DEFAULT_SEED,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> CreatorSplitMembership:
    """Assign creators to train/val/test with seeded greedy video-count balancing.

    Algorithm:
    1. Compute exact creator quotas from fractions.
    2. Shuffle creators with `seed` (stable across platforms for the same set).
    3. Greedy: place each creator into the open split whose current video total
       is farthest below its proportional video target (among splits that still
       need creators). This keeps creator quotas exact while roughly balancing N.
    """
    groups = np.asarray(groups)
    counts = _video_counts(groups)
    creators = sorted(counts.keys())
    n_train, n_val, n_test = creator_quotas(
        len(creators), train_frac=train_frac, val_frac=val_frac, test_frac=test_frac
    )

    rng = np.random.RandomState(seed)
    order = creators[:]
    rng.shuffle(order)

    total_videos = sum(counts.values())
    targets = {
        "train": total_videos * (n_train / len(creators)),
        "val": total_videos * (n_val / len(creators)),
        "test": total_videos * (n_test / len(creators)),
    }
    remaining = {"train": n_train, "val": n_val, "test": n_test}
    videos_so_far = {"train": 0, "val": 0, "test": 0}
    assigned: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for creator in order:
        open_splits = [s for s, left in remaining.items() if left > 0]
        if not open_splits:
            break
        # Prefer the split most under its video target; tie-break: most creator
        # slots remaining, then name order for determinism.
        def deficit(split: str) -> tuple[float, int, str]:
            return (
                targets[split] - videos_so_far[split],
                remaining[split],
                split,
            )

        split = max(open_splits, key=deficit)
        assigned[split].append(creator)
        remaining[split] -= 1
        videos_so_far[split] += counts[creator]

    membership = CreatorSplitMembership(
        seed=seed,
        train_creators=tuple(sorted(assigned["train"])),
        val_creators=tuple(sorted(assigned["val"])),
        test_creators=tuple(sorted(assigned["test"])),
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        n_videos={
            "train": videos_so_far["train"],
            "val": videos_so_far["val"],
            "test": videos_so_far["test"],
        },
    )
    membership.assert_disjoint()
    if membership.all_creators() != set(creators):
        raise ValueError("Split assignment dropped or invented creators")
    return membership


def membership_to_dict(membership: CreatorSplitMembership) -> dict:
    payload = asdict(membership)
    # tuples -> lists for JSON
    payload["train_creators"] = list(membership.train_creators)
    payload["val_creators"] = list(membership.val_creators)
    payload["test_creators"] = list(membership.test_creators)
    return payload


def membership_from_dict(payload: dict) -> CreatorSplitMembership:
    return CreatorSplitMembership(
        seed=int(payload["seed"]),
        train_creators=tuple(payload["train_creators"]),
        val_creators=tuple(payload["val_creators"]),
        test_creators=tuple(payload["test_creators"]),
        train_frac=float(payload.get("train_frac", DEFAULT_TRAIN_FRAC)),
        val_frac=float(payload.get("val_frac", DEFAULT_VAL_FRAC)),
        test_frac=float(payload.get("test_frac", DEFAULT_TEST_FRAC)),
        n_videos=payload.get("n_videos"),
    )


def save_creator_splits(membership: CreatorSplitMembership, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(membership_to_dict(membership), indent=2), encoding="utf-8")
    return path


def load_creator_splits(path: str | Path = DEFAULT_SPLITS_PATH) -> CreatorSplitMembership:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    membership = membership_from_dict(payload)
    membership.assert_disjoint()
    return membership


def row_mask_for_split(
    frame: pd.DataFrame,
    membership: CreatorSplitMembership,
    split: str,
    group_column: str = GROUP_COLUMN,
) -> np.ndarray:
    """Boolean mask selecting rows whose creator belongs to `split`."""
    creators = set(membership.creators_for(split))
    return frame[group_column].astype(str).isin(creators).to_numpy()


def indices_for_split(
    groups: np.ndarray,
    membership: CreatorSplitMembership,
    split: str,
) -> np.ndarray:
    creators = set(membership.creators_for(split))
    groups = np.asarray(groups).astype(str)
    return np.flatnonzero(np.isin(groups, list(creators)))


def generate_and_save_splits(
    groups: np.ndarray | list[str],
    path: str | Path = DEFAULT_SPLITS_PATH,
    seed: int = DEFAULT_SEED,
    **fracs,
) -> CreatorSplitMembership:
    membership = make_creator_splits(groups, seed=seed, **fracs)
    save_creator_splits(membership, path)
    return membership
