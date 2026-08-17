"""Persistent store for analysis jobs (SQLite-backed).

Tracks each uploaded video's analysis: its status, per-step progress, file
location, and basic metadata. The public functions here are the only way the
rest of the app touches storage, so the backing store (SQLite, per PRD 21) can
change without affecting the API layer.

Nested data (`steps`, `metadata`) is stored as JSON text columns and rehydrated
to dicts on read, so callers keep working with plain Python dicts.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from backend.core.database import get_connection

# Canonical ordered pipeline steps (PRD 19.2 / 17.3). The keys here are the
# contract the frontend's processing page relies on.
PIPELINE_STEPS: tuple[str, ...] = (
    "upload",
    "metadata",
    "frame_sampling",
    "visual_quality",
    "framing",
    "motion",
    "audio",
    "ocr",
    "prediction",
    "report",
)

# Maps record fields callers use -> DB columns. JSON-serialized fields are noted.
_COLUMN_MAP = {
    "status": "status",
    "video_file_path": "video_file_path",
    "original_filename": "original_filename",
    "report_json_path": "report_json_path",
    "error": "error",
    "metadata": "metadata_json",
    "steps": "steps_json",
}
_JSON_FIELDS = {"metadata", "steps"}


def _new_id() -> str:
    """Short, collision-resistant, URL-safe job id."""
    return f"analysis_{uuid.uuid4().hex[:12]}"


def _row_to_record(row: sqlite3.Row) -> dict:
    """Convert a DB row into the dict shape the app works with."""
    return {
        "analysis_id": row["id"],
        "created_at": row["created_at"],
        "status": row["status"],
        "video_file_path": row["video_file_path"],
        "original_filename": row["original_filename"],
        "steps": json.loads(row["steps_json"]),
        "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else None,
        "report_json_path": row["report_json_path"],
        "error": row["error"],
    }


def create_analysis(video_file_path: str, original_filename: str) -> dict:
    """Register a new analysis job with all steps pending except 'upload'."""
    analysis_id = _new_id()
    created_at = datetime.now(timezone.utc).isoformat()
    steps = {name: "pending" for name in PIPELINE_STEPS}
    steps["upload"] = "complete"  # the file is already saved by the time we're here

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO analyses
                (id, created_at, status, video_file_path, original_filename,
                 steps_json, metadata_json, report_json_path, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_id,
                created_at,
                "processing",
                video_file_path,
                original_filename,
                json.dumps(steps),
                None,
                None,
                None,
            ),
        )
    return get_analysis(analysis_id)  # type: ignore[return-value]


def get_analysis(analysis_id: str) -> dict | None:
    """Fetch a job record, or None if the id is unknown."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
    return _row_to_record(row) if row is not None else None


def update_analysis(analysis_id: str, **fields) -> None:
    """Patch one or more fields on a job record.

    Only whitelisted fields are written. Column names come from our own map
    (never user input) and all values are passed as bound parameters, so this is
    safe from SQL injection.
    """
    assignments: list[str] = []
    values: list = []
    for key, value in fields.items():
        column = _COLUMN_MAP.get(key)
        if column is None:
            continue
        if key in _JSON_FIELDS:
            value = json.dumps(value)
        assignments.append(f"{column} = ?")
        values.append(value)

    if not assignments:
        return

    values.append(analysis_id)
    with get_connection() as conn:
        conn.execute(
            f"UPDATE analyses SET {', '.join(assignments)} WHERE id = ?", values
        )


def set_step(analysis_id: str, step: str, step_status: str) -> None:
    """Update a single pipeline step's status for a job (read-modify-write)."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT steps_json FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
        if row is None:
            return
        steps = json.loads(row["steps_json"])
        if step in steps:
            steps[step] = step_status
            conn.execute(
                "UPDATE analyses SET steps_json = ? WHERE id = ?",
                (json.dumps(steps), analysis_id),
            )


def delete_analysis(analysis_id: str) -> None:
    """Remove a job record entirely (e.g. when validation rejects the upload)."""
    with get_connection() as conn:
        conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))


def list_analyses() -> list[dict]:
    """All job records (used by retention cleanup)."""
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM analyses").fetchall()
    return [_row_to_record(row) for row in rows]
