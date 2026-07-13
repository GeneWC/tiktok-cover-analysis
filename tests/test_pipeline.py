"""Tests for the analysis pipeline orchestrator (PRD 14.3 / 17.3 / 19)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import backend.inference.pipeline as pipeline
from backend.core.database import init_db
from backend.inference.feature_assembly import AssembledFeatures
from backend.inference.model_registry import LoadedModel, ModelRegistry
from backend.services import analysis_store


class _StubClassifier:
    def __init__(self, proba=0.6, raise_on_predict=False):
        self._p = proba
        self._raise = raise_on_predict

    def predict_proba(self, X):
        if self._raise:
            raise RuntimeError("boom")
        return np.array([[1.0 - self._p, self._p]])


class _StubRegressor:
    def __init__(self, value):
        self._v = value

    def predict(self, X):
        return np.array([self._v])


def _registry(classifier=None) -> ModelRegistry:
    models = {
        "top_quartile": LoadedModel(
            "top_quartile", classifier or _StubClassifier(), "top_quartile_for_creator",
            "classification", ["f"], False, None,
        ),
        "engagement": LoadedModel(
            "engagement", _StubRegressor(2.5), "engagement_rate",
            "regression", ["f"], False, "engagement_tier",
        ),
        "creator_relative": LoadedModel(
            "creator_relative", _StubRegressor(0.5), "creator_relative_log_views",
            "regression", ["f"], True, "view_performance_tier",
        ),
        "shareability": LoadedModel(
            "shareability", _StubRegressor(0.05), "share_rate",
            "regression", ["f"], True, "shareability_tier",
        ),
    }
    calibration = {
        "regressor_tiers": {
            "engagement": {"thresholds": {"q25": 1.0, "q50": 2.0, "q75": 3.0}},
            "creator_relative": {"thresholds": {"q25": 0.0, "q50": 1.0, "q75": 2.0}},
            "shareability": {"thresholds": {"q25": 0.1, "q50": 0.2, "q75": 0.3}},
        },
        "feature_percentiles": {},
    }
    return ModelRegistry(models=models, all_features=["f"], importances={}, calibration=calibration)


def _assembled(has_audio=True, frames=100, steps=None) -> AssembledFeatures:
    steps = steps or {
        "metadata": "complete", "frame_sampling": "complete",
        "visual_quality": "complete", "framing": "complete",
        "motion": "complete", "audio": "complete", "ocr": "complete",
    }
    return AssembledFeatures(
        X=pd.DataFrame({"f": [0.5]}),
        raw={"f": 0.5, "has_audio": has_audio},
        steps=steps,
        has_audio=has_audio,
        frames_sampled=frames,
        duration_seconds=30.0,
    )


@pytest.fixture
def db(tmp_path, monkeypatch):
    # analysis_store + database both read the shared settings singleton at call
    # time, so pointing it at a temp DB/reports dir fully isolates the test.
    monkeypatch.setattr(pipeline.settings, "db_path", tmp_path / "test.db")
    monkeypatch.setattr(pipeline.settings, "reports_dir", tmp_path / "reports")
    init_db()
    return tmp_path


def _make_job(has_audio=True) -> str:
    record = analysis_store.create_analysis(
        video_file_path="video.mp4", original_filename="video.mp4"
    )
    analysis_store.update_analysis(
        record["analysis_id"],
        metadata={
            "duration_seconds": 30.0, "width": 1080, "height": 1920,
            "fps": 30.0, "has_audio": has_audio, "aspect_ratio": 0.5625,
            "is_vertical_video": True, "is_square_video": False,
        },
    )
    return record["analysis_id"]


def test_successful_run_populates_report_and_status(db, monkeypatch):
    monkeypatch.setattr(pipeline, "assemble_features", lambda p, registry=None: _assembled())
    analysis_id = _make_job()

    report = pipeline.run_analysis(analysis_id, registry=_registry())

    assert report.status == "complete"
    assert report.scores.top_quartile_probability == 0.6
    assert report.scores.engagement_tier == "medium_high"   # 2.5 in [q50,q75)
    assert report.scores.shareability_tier == "low"          # 0.05 < q25

    record = analysis_store.get_analysis(analysis_id)
    assert record["status"] == "complete"
    assert record["steps"]["prediction"] == "complete"
    assert record["steps"]["report"] == "complete"
    assert record["steps"]["audio"] == "complete"


def test_report_is_persisted_to_disk(db, monkeypatch):
    monkeypatch.setattr(pipeline, "assemble_features", lambda p, registry=None: _assembled())
    analysis_id = _make_job()
    pipeline.run_analysis(analysis_id, registry=_registry())

    report_path = db / "reports" / f"{analysis_id}.json"
    assert report_path.exists()
    saved = json.loads(report_path.read_text())
    assert saved["analysis_id"] == analysis_id
    assert saved["scores"]["top_quartile_probability"] == 0.6


def test_no_audio_marks_audio_skipped(db, monkeypatch):
    steps = {
        "metadata": "complete", "frame_sampling": "complete",
        "visual_quality": "complete", "framing": "complete",
        "motion": "complete", "audio": "failed", "ocr": "complete",
    }
    monkeypatch.setattr(
        pipeline, "assemble_features",
        lambda p, registry=None: _assembled(has_audio=False, steps=steps),
    )
    analysis_id = _make_job(has_audio=False)
    pipeline.run_analysis(analysis_id, registry=_registry())

    record = analysis_store.get_analysis(analysis_id)
    assert record["steps"]["audio"] == "skipped"
    assert record["status"] == "complete"


def test_undecodable_video_marks_failed(db, monkeypatch):
    monkeypatch.setattr(
        pipeline, "assemble_features",
        lambda p, registry=None: _assembled(frames=0, steps={"frame_sampling": "failed"}),
    )
    analysis_id = _make_job()
    report = pipeline.run_analysis(analysis_id, registry=_registry())

    assert report.status == "failed"
    assert report.scores.top_quartile_probability is None
    record = analysis_store.get_analysis(analysis_id)
    assert record["status"] == "failed"
    assert record["steps"]["prediction"] == "skipped"


def test_prediction_error_marks_failed(db, monkeypatch):
    monkeypatch.setattr(pipeline, "assemble_features", lambda p, registry=None: _assembled())
    registry = _registry(classifier=_StubClassifier(raise_on_predict=True))
    analysis_id = _make_job()

    report = pipeline.run_analysis(analysis_id, registry=registry)

    assert report.status == "failed"
    record = analysis_store.get_analysis(analysis_id)
    assert record["steps"]["prediction"] == "failed"
    assert record["status"] == "failed"
