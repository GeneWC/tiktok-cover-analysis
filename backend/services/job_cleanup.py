"""Delete leftover upload media after a job finishes or expires."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.core.config import settings
from backend.services import analysis_store


def delete_job_video(analysis_id: str) -> None:
    """Remove the uploaded video; keep the report JSON for later reads."""
    record = analysis_store.get_analysis(analysis_id)
    if record is None:
        return
    path = record.get("video_file_path")
    if path:
        Path(path).unlink(missing_ok=True)
        analysis_store.update_analysis(analysis_id, video_file_path="")


def cleanup_expired_jobs(max_age_hours: float | None = None) -> int:
    """Delete videos (and optionally reports) older than the retention window."""
    hours = settings.job_retention_hours if max_age_hours is None else max_age_hours
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    removed = 0
    for record in analysis_store.list_analyses():
        created = record.get("created_at")
        if not created:
            continue
        try:
            created_at = datetime.fromisoformat(created)
        except ValueError:
            continue
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if created_at > cutoff:
            continue
        delete_job_video(record["analysis_id"])
        report = record.get("report_json_path")
        if report:
            Path(report).unlink(missing_ok=True)
        analysis_store.delete_analysis(record["analysis_id"])
        removed += 1
    return removed
