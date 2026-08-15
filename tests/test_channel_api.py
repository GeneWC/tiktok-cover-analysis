"""API tests for channel diagnostics routes (D-018)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import backend.api.channel_routes as channel_routes
from backend.core.config import settings
from backend.inference.channel_pipeline import labels_from_views
from backend.schemas.analysis import VideoMetadata
from backend.services import channel_store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", tmp_path / "test.db")
    monkeypatch.setattr(settings, "videos_dir", tmp_path / "videos")
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "reports")
    (tmp_path / "videos").mkdir()
    (tmp_path / "reports").mkdir()
    from backend.app import app
    from backend.core.database import init_db

    init_db()
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


def test_labels_from_views_top_quartile():
    views = [100, 100, 100, 100, 10000]
    labels = labels_from_views(views)
    assert labels is not None
    assert int(labels.sum()) >= 1
    assert labels[-1] == 1


def test_labels_from_views_requires_all_views():
    assert labels_from_views([1, 2, 3, None, 5]) is None


def test_channel_diagnose_rejects_too_few(client):
    files = [
        ("video_files", ("a.mp4", b"x", "video/mp4")),
        ("video_files", ("b.mp4", b"x", "video/mp4")),
    ]
    response = client.post("/api/channel/diagnose", files=files)
    assert response.status_code == 400


def test_channel_diagnose_registers_job(client, monkeypatch):
    scheduled: list[str] = []
    monkeypatch.setattr(channel_routes, "validate_video", lambda path: _metadata())
    monkeypatch.setattr(
        channel_routes,
        "run_channel_diagnose",
        lambda channel_id: scheduled.append(channel_id),
    )

    files = [
        ("video_files", (f"v{i}.mp4", b"fake", "video/mp4")) for i in range(5)
    ]
    metrics = json.dumps([{"views": 100 * (i + 1)} for i in range(5)])
    response = client.post(
        "/api/channel/diagnose",
        files=files,
        data={"metrics_json": metrics},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["n_videos"] == 5
    assert body["channel_id"].startswith("channel_")
    assert scheduled == [body["channel_id"]]

    record = channel_store.get_channel_job(body["channel_id"])
    assert record is not None
    assert record["steps"]["upload"] == "complete"
    assert len(record["videos"]) == 5


def test_channel_status_and_pending_report(client, monkeypatch):
    monkeypatch.setattr(channel_routes, "validate_video", lambda path: _metadata())
    monkeypatch.setattr(channel_routes, "run_channel_diagnose", lambda channel_id: None)

    files = [("video_files", (f"v{i}.mp4", b"fake", "video/mp4")) for i in range(5)]
    response = client.post("/api/channel/diagnose", files=files)
    channel_id = response.json()["channel_id"]

    status = client.get(f"/api/channel/{channel_id}/status")
    assert status.status_code == 200
    assert status.json()["n_videos"] == 5

    report = client.get(f"/api/channel/{channel_id}/report")
    assert report.status_code == 200
    assert report.json()["mode"] == "diagnostics"
