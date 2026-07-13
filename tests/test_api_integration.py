"""Golden-sample API integration test (PRD 23.2 / 23.3).

Uploads a real cover video through the live API and asserts a fully populated
report comes back - exercising the whole Phase-5 path over HTTP: upload ->
validation -> background pipeline (extraction + models + scoring) -> persisted
report. Marked `slow` (decodes video, runs MediaPipe/EAST, loads model
artifacts) and skips automatically when no sample video or trained models are
present.

Run only the fast suite with:   pytest -m "not slow"
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.core.config import settings

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Prefer the training corpus, fall back to any stored/upload sample.
_SAMPLES = (
    sorted((_PROJECT_ROOT / "downloads").glob("*.mp4"))
    or sorted((_PROJECT_ROOT / "data" / "videos").glob("*.mp4"))
    or sorted((_PROJECT_ROOT / "videos").glob("*.mp4"))
)
_MODELS_READY = (settings.models_dir / "feature_schema.json").exists()

_TIERS = {"low", "medium", "medium_high", "high"}


@pytest.mark.slow
@pytest.mark.skipif(not _SAMPLES, reason="no sample video available")
@pytest.mark.skipif(not _MODELS_READY, reason="trained model artifacts not found")
def test_upload_to_report_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", tmp_path / "test.db")
    monkeypatch.setattr(settings, "videos_dir", tmp_path / "videos")
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "reports")
    (tmp_path / "videos").mkdir()
    (tmp_path / "reports").mkdir()

    from backend.app import app

    with TestClient(app) as client:
        with open(_SAMPLES[0], "rb") as handle:
            post = client.post(
                "/api/analyze",
                files={"video_file": ("cover.mp4", handle, "video/mp4")},
            )
        assert post.status_code == 202
        analysis_id = post.json()["analysis_id"]

        # TestClient runs the background task before returning the POST response,
        # so the job is already finished by the time we poll.
        status = client.get(f"/api/analyze/{analysis_id}/status").json()
        assert status["status"] == "complete"
        assert all(v == "complete" for v in status["steps"].values())

        report = client.get(f"/api/analyze/{analysis_id}/report").json()

    assert report["status"] == "complete"

    scores = report["scores"]
    assert 0.0 <= scores["top_quartile_probability"] <= 1.0
    assert scores["engagement_tier"] in _TIERS
    assert scores["view_performance_tier"] in _TIERS
    assert scores["shareability_tier"] in _TIERS
    for key in (
        "overall_presentation_score", "visual_quality_score",
        "audio_quality_score", "motion_score", "framing_score",
    ):
        assert 0.0 <= scores[key] <= 100.0

    # metadata, feature snapshot, signals, and disclaimers are all populated
    assert report["video_metadata"]["duration_seconds"] > 0
    assert report["features"]
    assert report["limitations"]
    explanation = report["explanation"]
    assert explanation["strong_signals"] or explanation["weak_signals"]

    # the report was persisted to disk as the serving source of truth
    assert (settings.reports_dir / f"{analysis_id}.json").exists()
