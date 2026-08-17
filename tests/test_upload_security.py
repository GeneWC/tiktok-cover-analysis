"""Security regression tests for the upload and job-id surface."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.api.analyze_routes as routes
from backend.core.config import settings
from backend.schemas.analysis import VideoMetadata
from backend.services import analysis_store
from backend.services.job_cleanup import cleanup_expired_jobs, delete_job_video
from backend.services.video_validation import VideoValidationError


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", tmp_path / "test.db")
    monkeypatch.setattr(settings, "videos_dir", tmp_path / "videos")
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "reports")
    monkeypatch.setattr(settings, "upload_rate_limit", 10_000)
    (tmp_path / "videos").mkdir()
    (tmp_path / "reports").mkdir()
    from backend.app import app

    with TestClient(app) as test_client:
        yield test_client


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        duration_seconds=30.0,
        width=1080,
        height=1920,
        fps=30.0,
        has_audio=True,
        aspect_ratio=0.5625,
        is_vertical_video=True,
        is_square_video=False,
    )


def test_rejects_path_traversal_job_id(client):
    assert client.get("/api/analyze/..%2f..%2fetc%2fpasswd/status").status_code == 404
    assert client.get("/api/analyze/analysis_../secret/status").status_code == 404


def test_rejects_malformed_job_id(client):
    assert client.get("/api/analyze/not-a-real-id/status").status_code == 404
    assert client.get("/api/analyze/analysis_ZZZZZZZZZZZZ/report").status_code == 404


def test_malicious_filename_is_not_used_as_path(client, monkeypatch, tmp_path):
    scheduled: list[str] = []
    monkeypatch.setattr(routes, "validate_video", lambda path: _metadata())
    monkeypatch.setattr(routes, "run_analysis", lambda analysis_id: scheduled.append(analysis_id))

    response = client.post(
        "/api/analyze",
        files={"video_file": ("../../evil.mp4", b"fake-bytes", "video/mp4")},
    )
    assert response.status_code == 202
    analysis_id = response.json()["analysis_id"]
    record = analysis_store.get_analysis(analysis_id)
    assert record is not None
    assert record["original_filename"] == "evil.mp4"
    saved = Path(record["video_file_path"])
    assert saved.parent == settings.videos_dir
    assert saved.name.startswith(analysis_id)


def test_oversized_upload_is_rejected(client, monkeypatch):
    monkeypatch.setattr(settings, "max_file_size_mb", 1)

    def _fail(_path):
        raise AssertionError("should not probe an oversized file")

    monkeypatch.setattr(routes, "validate_video", _fail)
    payload = b"x" * (2 * 1024 * 1024)
    response = client.post(
        "/api/analyze",
        files={"video_file": ("cover.mp4", payload, "video/mp4")},
    )
    assert response.status_code == 413
    assert list(settings.videos_dir.glob("*")) == []


def test_cleanup_after_validation_failure(client, monkeypatch):
    def _reject(_path):
        raise VideoValidationError(
            code="undecodable",
            message="Could not decode the video.",
            status_code=400,
        )

    monkeypatch.setattr(routes, "validate_video", _reject)
    response = client.post(
        "/api/analyze",
        files={"video_file": ("cover.mp4", b"fake-bytes", "video/mp4")},
    )
    assert response.status_code == 400
    assert list(settings.videos_dir.glob("*")) == []


def test_upload_rate_limit_returns_429():
    from fastapi import FastAPI
    from backend.core.rate_limit import UploadRateLimitMiddleware

    tiny = FastAPI()
    tiny.add_middleware(UploadRateLimitMiddleware, limit=2, window_seconds=600)

    @tiny.post("/api/analyze")
    def _ok() -> dict[str, bool]:
        return {"ok": True}

    with TestClient(tiny) as limited:
        assert limited.post("/api/analyze").status_code == 200
        assert limited.post("/api/analyze").status_code == 200
        blocked = limited.post("/api/analyze")
        assert blocked.status_code == 429
        assert "Too many uploads" in blocked.json()["detail"]


def test_delete_job_video_and_expired_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", tmp_path / "test.db")
    monkeypatch.setattr(settings, "videos_dir", tmp_path / "videos")
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "reports")
    settings.videos_dir.mkdir()
    settings.reports_dir.mkdir()
    from backend.core.database import init_db

    init_db()
    video = settings.videos_dir / "analysis_aaaaaaaaaaaa.mp4"
    video.write_bytes(b"data")
    record = analysis_store.create_analysis(str(video), "clip.mp4")
    analysis_store.update_analysis(
        record["analysis_id"],
        created_at=(datetime.now(timezone.utc) - timedelta(hours=48)).isoformat(),
    )
    # created_at is not in COLUMN_MAP — write it directly for the test
    from backend.core.database import get_connection

    old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    with get_connection() as conn:
        conn.execute(
            "UPDATE analyses SET created_at = ? WHERE id = ?",
            (old, record["analysis_id"]),
        )

    delete_job_video(record["analysis_id"])
    assert not video.exists()
    removed = cleanup_expired_jobs(max_age_hours=24)
    assert removed == 1
    assert analysis_store.get_analysis(record["analysis_id"]) is None
