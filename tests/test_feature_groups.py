"""Tests for feature-group partition + per-model specs (PRD 12.1 / 12.2)."""

from __future__ import annotations

from backend.training.extract_training_features import feature_names
from backend.training.feature_groups import (
    GROUP_NAMES,
    assign_group,
    select_group_features,
)
from backend.training.model_dataset import select_feature_columns
from backend.training.model_specs import MODEL_SPECS, PRIMARY_SPEC, ModelSpec


def _model_features():
    """The canonical 70 model-input features (schema probe -> drop labels/status)."""
    return select_feature_columns(feature_names())


def test_every_feature_maps_to_exactly_one_group():
    features = _model_features()
    assert features, "expected a non-empty feature schema"
    # every feature is assigned to a known group (a clean partition)
    for feat in features:
        group = assign_group(feat)
        assert group in GROUP_NAMES, f"{feat} -> {group}"


def test_groups_are_disjoint_and_cover_all_features():
    features = _model_features()
    seen: list[str] = []
    for group in GROUP_NAMES:
        seen += select_group_features(features, (group,))
    assert sorted(seen) == sorted(features)  # partition: no overlaps, no gaps


def test_select_group_features_empty_means_all():
    features = _model_features()
    assert select_group_features(features, ()) == features


def test_select_group_features_subset_and_order():
    features = _model_features()
    framing_visual = select_group_features(features, ("framing", "visual"))
    assert 0 < len(framing_visual) < len(features)
    # preserves original ordering
    assert framing_visual == [f for f in features if f in set(framing_visual)]
    # only framing/visual members
    assert all(assign_group(f) in {"framing", "visual"} for f in framing_visual)


def test_model_specs_are_wellformed():
    assert PRIMARY_SPEC.name == "top_quartile"
    assert PRIMARY_SPEC.feature_groups == ("framing", "visual")
    names = [s.name for s in MODEL_SPECS]
    assert names == ["top_quartile", "engagement", "creator_relative", "shareability"]
    for spec in MODEL_SPECS:
        assert isinstance(spec, ModelSpec)
        assert spec.task in {"classification", "regression"}
        assert spec.artifact.endswith(".pkl")
        assert all(g in GROUP_NAMES for g in spec.feature_groups)
    # the weak targets are flagged low-confidence
    assert {s.name for s in MODEL_SPECS if s.low_confidence} == {
        "creator_relative",
        "shareability",
    }
