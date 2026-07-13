"""Tests for single-video feature assembly (PRD 6.2 / 12.4)."""

from __future__ import annotations

import numpy as np

import backend.inference.feature_assembly as fa
from backend.features.extract_features import FeatureExtractionResult
from backend.inference.feature_assembly import assemble_features, to_feature_frame
from backend.inference.model_registry import ModelRegistry

_ORDER = ["duration_seconds", "has_audio", "person_visible_ratio", "audio_rms_mean"]


def test_to_feature_frame_orders_and_coerces_types():
    features = {
        "audio_rms_mean": 0.42,        # out of order in the dict on purpose
        "duration_seconds": 30.0,
        "has_audio": True,             # bool -> 1.0
        "person_visible_ratio": "0.8",  # numeric string -> float
        "extra_key": 999,              # not in schema -> dropped
    }
    X = to_feature_frame(features, _ORDER)

    assert list(X.columns) == _ORDER            # schema order preserved
    assert X.shape == (1, 4)
    assert X.loc[0, "has_audio"] == 1.0
    assert X.loc[0, "person_visible_ratio"] == 0.8
    assert X.loc[0, "audio_rms_mean"] == 0.42


def test_to_feature_frame_fills_missing_with_nan():
    X = to_feature_frame({"duration_seconds": 12.0}, _ORDER)
    assert X.loc[0, "duration_seconds"] == 12.0
    assert np.isnan(X.loc[0, "has_audio"])
    assert np.isnan(X.loc[0, "person_visible_ratio"])


def _fake_registry() -> ModelRegistry:
    return ModelRegistry(models={}, all_features=_ORDER, importances={}, calibration={})


def test_assemble_features_wires_extractor_output(monkeypatch):
    fake_result = FeatureExtractionResult(
        features={
            "duration_seconds": 45.0, "has_audio": False,
            "person_visible_ratio": 0.5, "audio_rms_mean": None,
        },
        steps={"metadata": "complete", "audio": "failed"},
        frames_sampled=120,
        duration_seconds=45.0,
    )
    monkeypatch.setattr(fa, "extract_all_features", lambda path, sample=None: fake_result)

    assembled = assemble_features("video.mp4", registry=_fake_registry())

    assert list(assembled.X.columns) == _ORDER
    assert assembled.has_audio is False
    assert assembled.frames_sampled == 120
    assert assembled.duration_seconds == 45.0
    assert assembled.steps["audio"] == "failed"
    assert assembled.usable is True
    assert np.isnan(assembled.X.loc[0, "audio_rms_mean"])  # None -> NaN


def test_assemble_features_flags_unusable_when_no_frames(monkeypatch):
    fake_result = FeatureExtractionResult(
        features={"has_audio": True}, steps={"frame_sampling": "failed"},
        frames_sampled=0, duration_seconds=None,
    )
    monkeypatch.setattr(fa, "extract_all_features", lambda path, sample=None: fake_result)

    assembled = assemble_features("bad.mp4", registry=_fake_registry())
    assert assembled.usable is False
    assert assembled.has_audio is True
