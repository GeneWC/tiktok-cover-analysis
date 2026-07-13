"""SQLite persistence layer.

SQLite is an embedded, serverless, single-file ACID database built into the
Python standard library (`sqlite3`) - ideal for an MVP because there is no
server to run. This module owns connection management and schema creation; the
`analysis_store` service builds on top of it.

Design notes:
- A fresh connection is opened per operation via the `get_connection()` context
  manager. `sqlite3` connections are not safe to share across threads, and
  FastAPI/Starlette may dispatch work on different threads, so per-operation
  connections keep us thread-safe without locks.
- WAL (write-ahead logging) journal mode is enabled for better read/write
  concurrency.
- The connection context manager commits on success and rolls back on error,
  giving each operation transactional (atomic) behavior.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from backend.core.config import settings

# Phase 1 stores the `analyses` table (PRD 21.1) plus JSON columns for the
# nested step/metadata data. The `analysis_features` and `analysis_predictions`
# tables (PRD 21.2 / 21.3) are added in later phases when we actually compute them.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id                TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    status            TEXT NOT NULL,
    video_file_path   TEXT,
    original_filename TEXT,
    steps_json        TEXT NOT NULL,
    metadata_json     TEXT,
    report_json_path  TEXT,
    error             TEXT
);
"""


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection, committing on success and rolling back on error."""
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row  # rows behave like dicts (access by column name)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with get_connection() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(_SCHEMA)
