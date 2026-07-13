"""API endpoint tests for the analyze routes (PRD 19.1-19.3).

Uses FastAPI's TestClient with an isolated temp DB / storage dirs. The heavy
pipeline (`run_analysis`) and video decode (`validate_video`) are stubbed so we
test routing, job registration, background scheduling, and report serving - not
model inference (covered by the pipeline tests).
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import backend.api.analyze_routes as routes
from backend.core.config import settings
from backend.schemas.analysis import (
    ReportExplanation,
    ReportResponse,
    ReportScores,
    VideoMetadata,
)
from backend.services import analysis_store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", tmp_path / "test.db")
    monkeypatch.setattr(settings, "videos_dir", tmp_path / "videos")
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "reports")
    (tmp_path / "videos").mkdir()
    (tmp_path / "reports").mkdir()
    from backend.app import app

    with TestClient(app) as test_client:
        yield test_client


def _metadata(has_audio=True) -> VideoMetadata:
    return VideoMetadata(
        duration_seconds=30.0, width=1080, height=1920, fps=30.0,
        has_audio=has_audio, aspect_ratio=0.5625,
        is_vertical_video=True, is_square_video=False,
    )


def test_upload_registers_job_and_schedules_pipeline(client, monkeypatch):
    scheduled: list[str] = []
    monkeypatch.setattr(routes, "validate_video", lambda path: _metadata())
    monkeypatch.setattr(routes, "run_analysis", lambda analysis_id: scheduled.append(analysis_id))

    response = client.post(
        "/api/analyze",
        files={"video_file": ("cover.mp4", b"fake-bytes", "video/mp4")},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "processing"
    analysis_id = body["analysis_id"]

    # background task ran (TestClient executes it after the response)
    assert scheduled == [analysis_id]

    record = analysis_store.get_analysis(analysis_id)
    assert record is not None
    assert record["steps"]["metadata"] == "complete"


def test_upload_rejects_unsupported_extension(client):
    response = client.post(
        "/api/analyze",
        files={"video_file": ("cover.txt", b"nope", "text/plain")},
    )
    assert response.status_code == 400


def test_report_returns_pending_before_pipeline_completes(client):
    record = analysis_store.create_analysis(
        video_file_path="v.mp4", original_filename="v.mp4"
    )
    analysis_id = record["analysis_id"]

    response = client.get(f"/api/analyze/{analysis_id}/report")
    assert response.status_code == 200
    body = response.json()
    assert body["scores"]["top_quartile_probability"] is None
    assert any("not yet available" in note for note in body["limitations"])


def test_report_serves_persisted_report_when_available(client):
    record = analysis_store.create_analysis(
        video_file_path="v.mp4", original_filename="v.mp4"
    )
    analysis_id = record["analysis_id"]

    report = ReportResponse(
        analysis_id=analysis_id, status="complete",
        video_metadata=_metadata(),
        scores=ReportScores(top_quartile_probability=0.42, engagement_tier="high"),
        explanation=ReportExplanation(strong_signals=["Footage sharpness: strong."]),
        limitations=["exploratory"],
    )
    report_path = settings.reports_dir / f"{analysis_id}.json"
    report_path.write_text(report.model_dump_json(), encoding="utf-8")
    analysis_store.update_analysis(analysis_id, report_json_path=str(report_path))

    response = client.get(f"/api/analyze/{analysis_id}/report")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "complete"
    assert body["scores"]["top_quartile_probability"] == 0.42
    assert body["scores"]["engagement_tier"] == "high"


def test_status_and_report_404_for_unknown_id(client):
    assert client.get("/api/analyze/nope/status").status_code == 404
    assert client.get("/api/analyze/nope/report").status_code == 404
