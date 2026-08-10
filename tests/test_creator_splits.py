"""Tests for durable creator train/val/test splits (D-001)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.training.creator_splits import (
    creator_quotas,
    indices_for_split,
    load_creator_splits,
    make_creator_splits,
    save_creator_splits,
)


def _groups(n_per=(10, 20, 8, 15, 12, 9, 7, 11, 14, 6)):
    creators = [f"c{i}" for i in range(len(n_per))]
    return np.concatenate([np.repeat(c, n) for c, n in zip(creators, n_per)])


def test_quotas_for_24_creators():
    assert creator_quotas(24) == (14, 5, 5)


def test_quotas_small_sets():
    assert creator_quotas(3) == (1, 1, 1)
    with pytest.raises(ValueError):
        creator_quotas(2)


def test_splits_are_disjoint_and_cover_all():
    groups = _groups()
    membership = make_creator_splits(groups, seed=42)
    membership.assert_disjoint()
    assert membership.all_creators() == set(np.unique(groups))
    n = len(membership.all_creators())
    assert (
        len(membership.train_creators)
        + len(membership.val_creators)
        + len(membership.test_creators)
        == n
    )


def test_splits_deterministic():
    groups = _groups()
    a = make_creator_splits(groups, seed=7)
    b = make_creator_splits(groups, seed=7)
    assert a.train_creators == b.train_creators
    assert a.val_creators == b.val_creators
    assert a.test_creators == b.test_creators


def test_indices_respect_creators():
    groups = _groups()
    membership = make_creator_splits(groups, seed=42)
    for split in ("train", "val", "test"):
        idx = indices_for_split(groups, membership, split)
        assert set(groups[idx]).issubset(set(membership.creators_for(split)))
        # no leakage into other splits
        other = membership.all_creators() - set(membership.creators_for(split))
        assert set(groups[idx]).isdisjoint(other)


def test_roundtrip_json(tmp_path: Path):
    groups = _groups()
    membership = make_creator_splits(groups, seed=42)
    path = tmp_path / "creator_splits.json"
    save_creator_splits(membership, path)
    loaded = load_creator_splits(path)
    assert loaded.train_creators == membership.train_creators
    assert loaded.val_creators == membership.val_creators
    assert loaded.test_creators == membership.test_creators
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["seed"] == 42
