"""SQLite-backed store for multi-video channel diagnostic jobs (D-018)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from backend.core.database import get_connection

CHANNEL_STEPS: tuple[str, ...] = (
    "upload",
    "features",
    "diagnose",
    "report",
)

_COLUMN_MAP = {
    "status": "status",
    "steps": "steps_json",
    "videos": "videos_json",
    "report_json_path": "report_json_path",
    "error": "error",
    "n_features_done": "n_features_done",
}
_JSON_FIELDS = {"steps", "videos"}


def _new_id() -> str:
    return f"channel_{uuid.uuid4().hex[:12]}"


def _row_to_record(row: sqlite3.Row) -> dict:
    return {
        "channel_id": row["id"],
        "created_at": row["created_at"],
        "status": row["status"],
        "steps": json.loads(row["steps_json"]),
        "videos": json.loads(row["videos_json"]),
        "report_json_path": row["report_json_path"],
        "error": row["error"],
        "n_features_done": int(row["n_features_done"] or 0),
    }


def create_channel_job(videos: list[dict]) -> dict:
    """Register a channel job. `videos` items: path, filename, views (optional)."""
    channel_id = _new_id()
    created_at = datetime.now(timezone.utc).isoformat()
    steps = {name: "pending" for name in CHANNEL_STEPS}
    steps["upload"] = "complete"

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO channel_jobs
                (id, created_at, status, steps_json, videos_json,
                 report_json_path, error, n_features_done)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                channel_id,
                created_at,
                "processing",
                json.dumps(steps),
                json.dumps(videos),
                None,
                None,
                0,
            ),
        )
    return get_channel_job(channel_id)  # type: ignore[return-value]


def get_channel_job(channel_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM channel_jobs WHERE id = ?", (channel_id,)
        ).fetchone()
    return _row_to_record(row) if row is not None else None


def update_channel_job(channel_id: str, **fields) -> None:
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
    values.append(channel_id)
    with get_connection() as conn:
        conn.execute(
            f"UPDATE channel_jobs SET {', '.join(assignments)} WHERE id = ?",
            values,
        )


def set_channel_step(channel_id: str, step: str, step_status: str) -> None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT steps_json FROM channel_jobs WHERE id = ?", (channel_id,)
        ).fetchone()
        if row is None:
            return
        steps = json.loads(row["steps_json"])
        if step in steps:
            steps[step] = step_status
            conn.execute(
                "UPDATE channel_jobs SET steps_json = ? WHERE id = ?",
                (json.dumps(steps), channel_id),
            )


def delete_channel_job(channel_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM channel_jobs WHERE id = ?", (channel_id,))
