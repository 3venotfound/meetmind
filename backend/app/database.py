import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    client_org TEXT NOT NULL,
    target_industry TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meetings (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    meeting_date TEXT NOT NULL CHECK (
        meeting_date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
    ),
    meeting_number INTEGER CHECK (meeting_number IS NULL OR meeting_number >= 1),
    status TEXT NOT NULL DEFAULT 'created' CHECK (
        status IN ('created', 'uploaded', 'processing', 'processed', 'failed')
    ),
    recording_path TEXT,
    transcript TEXT,
    summary TEXT,
    processing_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (project_id, meeting_number)
);

CREATE TABLE IF NOT EXISTS participants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    role TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    speaker TEXT NOT NULL,
    text TEXT NOT NULL,
    start_time_seconds INTEGER CHECK (
        start_time_seconds IS NULL OR start_time_seconds >= 0
    ),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS visual_evidence (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    timestamp_seconds INTEGER NOT NULL CHECK (timestamp_seconds >= 0),
    evidence_type TEXT NOT NULL CHECK (
        evidence_type IN ('whiteboard', 'slide', 'unknown')
    ),
    raw_ocr_text TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    image_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    field_name TEXT NOT NULL CHECK (
        field_name IN ('budget', 'deadline', 'owner', 'decision_text')
    ),
    field_value TEXT NOT NULL,
    normalized_value TEXT,
    budget_amount_minor INTEGER,
    currency_code TEXT CHECK (currency_code IS NULL OR length(currency_code) = 3),
    decided_by TEXT,
    timestamp_seconds INTEGER CHECK (
        timestamp_seconds IS NULL OR timestamp_seconds >= 0
    ),
    source_type TEXT NOT NULL CHECK (source_type IN ('transcript', 'visual')),
    reasoning_snippet TEXT,
    visual_evidence_id TEXT REFERENCES visual_evidence(id) ON DELETE SET NULL,
    is_canonical INTEGER NOT NULL CHECK (is_canonical IN (0, 1)),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    field_name TEXT NOT NULL CHECK (field_name IN ('budget', 'deadline', 'owner')),
    old_value TEXT NOT NULL,
    new_value TEXT NOT NULL,
    old_budget_amount_minor INTEGER,
    new_budget_amount_minor INTEGER,
    currency_code TEXT CHECK (currency_code IS NULL OR length(currency_code) = 3),
    from_meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    to_meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    reason TEXT,
    changed_by TEXT,
    source_type TEXT NOT NULL CHECK (source_type IN ('transcript', 'visual')),
    timestamp_seconds INTEGER CHECK (
        timestamp_seconds IS NULL OR timestamp_seconds >= 0
    ),
    detected_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS action_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    description TEXT NOT NULL,
    owner TEXT NOT NULL,
    due_date TEXT CHECK (
        due_date IS NULL OR due_date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
    ),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (
        status IN ('pending', 'in_progress', 'completed')
    ),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_meetings_project_date
    ON meetings(project_id, meeting_date DESC);
CREATE INDEX IF NOT EXISTS idx_participants_meeting
    ON participants(meeting_id);
CREATE INDEX IF NOT EXISTS idx_transcript_segments_meeting_time
    ON transcript_segments(meeting_id, start_time_seconds);
CREATE INDEX IF NOT EXISTS idx_visual_evidence_meeting_time
    ON visual_evidence(meeting_id, timestamp_seconds);
CREATE INDEX IF NOT EXISTS idx_decisions_project_field_canonical
    ON decisions(project_id, field_name, is_canonical);
CREATE INDEX IF NOT EXISTS idx_changes_project_field
    ON changes(project_id, field_name, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_action_items_meeting_status
    ON action_items(meeting_id, status);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path.resolve()

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(SCHEMA_SQL)
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(decisions)").fetchall()
            }
            if "normalized_value" not in columns:
                connection.execute(
                    "ALTER TABLE decisions ADD COLUMN normalized_value TEXT"
                )

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
