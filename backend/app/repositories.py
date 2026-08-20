import sqlite3
from datetime import datetime, timezone
from typing import Sequence
from uuid import uuid4

from app.database import Database
from app.schemas import MeetingCreate, ParticipantCreate, ProjectCreate


class ProjectNotFoundError(Exception):
    pass


class DuplicateMeetingNumberError(Exception):
    pass


class MeetingNotFoundError(Exception):
    pass


class RecordingAlreadyExistsError(Exception):
    pass


class MeetingNotProcessableError(Exception):
    def __init__(self, status: str):
        self.status = status
        super().__init__(status)


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class Repository:
    def __init__(self, database: Database):
        self.database = database

    def create_project(self, project: ProjectCreate) -> dict:
        project_id = str(uuid4())
        now = utc_now_text()
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO projects (
                    id, name, client_org, target_industry, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    project.name,
                    project.client_org,
                    project.target_industry,
                    now,
                    now,
                ),
            )
        result = self.get_project(project_id)
        if result is None:
            raise RuntimeError("Created project could not be retrieved")
        return result

    def get_project(self, project_id: str) -> dict | None:
        with self.database.connection() as connection:
            project = connection.execute(
                """
                SELECT id, name, client_org, target_industry, created_at, updated_at
                FROM projects
                WHERE id = ?
                """,
                (project_id,),
            ).fetchone()
            if project is None:
                return None

            stats = {
                "meetings_logged": connection.execute(
                    "SELECT COUNT(*) FROM meetings WHERE project_id = ?",
                    (project_id,),
                ).fetchone()[0],
                "decisions_changed": connection.execute(
                    "SELECT COUNT(*) FROM changes WHERE project_id = ?",
                    (project_id,),
                ).fetchone()[0],
                "unresolved_issues": connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM action_items AS action
                    JOIN meetings AS meeting ON meeting.id = action.meeting_id
                    WHERE meeting.project_id = ?
                      AND action.status IN ('pending', 'in_progress')
                    """,
                    (project_id,),
                ).fetchone()[0],
            }

            recent_rows = connection.execute(
                """
                SELECT id, title, meeting_date, meeting_number, status, created_at
                FROM meetings
                WHERE project_id = ?
                ORDER BY meeting_date DESC, created_at DESC
                LIMIT 10
                """,
                (project_id,),
            ).fetchall()

            recent_meetings = [
                self._recent_meeting(connection, project_id, row) for row in recent_rows
            ]
            current_memory = [
                memory
                for field_name in ("budget", "deadline", "owner")
                if (memory := self._current_memory(connection, project_id, field_name))
                is not None
            ]

            return {
                **dict(project),
                "stats": stats,
                "recent_meetings": recent_meetings,
                "current_memory": current_memory,
            }

    def _recent_meeting(
        self,
        connection: sqlite3.Connection,
        project_id: str,
        meeting: sqlite3.Row,
    ) -> dict:
        meeting_id = meeting["id"]
        participants = [
            row["name"]
            for row in connection.execute(
                "SELECT name FROM participants WHERE meeting_id = ? ORDER BY id",
                (meeting_id,),
            ).fetchall()
        ]
        decision_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM decisions
            WHERE meeting_id = ? AND field_name = 'decision_text'
            """,
            (meeting_id,),
        ).fetchone()[0]
        change_count = connection.execute(
            "SELECT COUNT(*) FROM changes WHERE to_meeting_id = ?",
            (meeting_id,),
        ).fetchone()[0]
        earlier_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM meetings
            WHERE project_id = ?
              AND (
                    meeting_date < ?
                    OR (meeting_date = ? AND created_at < ?)
              )
            """,
            (
                project_id,
                meeting["meeting_date"],
                meeting["meeting_date"],
                meeting["created_at"],
            ),
        ).fetchone()[0]

        state = "changed" if change_count else "baseline" if not earlier_count else "stable"
        return {
            "id": meeting_id,
            "title": meeting["title"],
            "meeting_date": meeting["meeting_date"],
            "meeting_number": meeting["meeting_number"],
            "status": meeting["status"],
            "participants": participants,
            "decision_count": decision_count,
            "state": state,
        }

    def _current_memory(
        self,
        connection: sqlite3.Connection,
        project_id: str,
        field_name: str,
    ) -> dict | None:
        row = connection.execute(
            """
            SELECT
                decision.field_value,
                decision.budget_amount_minor,
                decision.currency_code,
                decision.meeting_id,
                decision.source_type,
                meeting.meeting_date
            FROM decisions AS decision
            JOIN meetings AS meeting ON meeting.id = decision.meeting_id
            WHERE decision.project_id = ?
              AND decision.field_name = ?
              AND decision.is_canonical = 1
            ORDER BY
                meeting.meeting_date DESC,
                COALESCE(decision.timestamp_seconds, -1) DESC,
                decision.created_at DESC,
                decision.id DESC
            LIMIT 1
            """,
            (project_id, field_name),
        ).fetchone()
        if row is None:
            return None

        last_change = connection.execute(
            """
            SELECT meeting.meeting_date
            FROM changes AS change_record
            JOIN meetings AS meeting ON meeting.id = change_record.to_meeting_id
            WHERE change_record.project_id = ? AND change_record.field_name = ?
            ORDER BY change_record.detected_at DESC, change_record.id DESC
            LIMIT 1
            """,
            (project_id, field_name),
        ).fetchone()
        return {
            "field_name": field_name,
            "display_value": row["field_value"],
            "budget_amount_minor": row["budget_amount_minor"],
            "currency_code": row["currency_code"],
            "meeting_id": row["meeting_id"],
            "meeting_date": row["meeting_date"],
            "source_type": row["source_type"],
            "last_changed_at": last_change["meeting_date"] if last_change else None,
        }

    def create_meeting(
        self,
        meeting: MeetingCreate,
        participants: Sequence[ParticipantCreate] | None = None,
    ) -> dict:
        meeting_id = str(uuid4())
        now = utc_now_text()
        participant_list = meeting.participants if participants is None else participants

        try:
            with self.database.transaction() as connection:
                project_exists = connection.execute(
                    "SELECT 1 FROM projects WHERE id = ?",
                    (str(meeting.project_id),),
                ).fetchone()
                if project_exists is None:
                    raise ProjectNotFoundError

                connection.execute(
                    """
                    INSERT INTO meetings (
                        id, project_id, title, meeting_date, meeting_number,
                        status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'created', ?, ?)
                    """,
                    (
                        meeting_id,
                        str(meeting.project_id),
                        meeting.title,
                        meeting.meeting_date.isoformat(),
                        meeting.meeting_number,
                        now,
                        now,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO participants (meeting_id, name, role, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (meeting_id, participant.name, participant.role, now)
                        for participant in participant_list
                    ],
                )
        except sqlite3.IntegrityError as error:
            if "meetings.project_id, meetings.meeting_number" in str(error):
                raise DuplicateMeetingNumberError from error
            raise

        result = self.get_meeting(meeting_id)
        if result is None:
            raise RuntimeError("Created meeting could not be retrieved")
        return result

    def get_meeting(self, meeting_id: str) -> dict | None:
        with self.database.connection() as connection:
            meeting = connection.execute(
                """
                SELECT
                    id, project_id, title, meeting_date, meeting_number,
                    status, transcript, summary, processing_error, created_at, updated_at
                FROM meetings
                WHERE id = ?
                """,
                (meeting_id,),
            ).fetchone()
            if meeting is None:
                return None

            participants = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT id, name, role
                    FROM participants
                    WHERE meeting_id = ?
                    ORDER BY id
                    """,
                    (meeting_id,),
                ).fetchall()
            ]
            counts = {
                "decisions": connection.execute(
                    "SELECT COUNT(*) FROM decisions WHERE meeting_id = ?",
                    (meeting_id,),
                ).fetchone()[0],
                "action_items": connection.execute(
                    "SELECT COUNT(*) FROM action_items WHERE meeting_id = ?",
                    (meeting_id,),
                ).fetchone()[0],
                "visual_evidence": connection.execute(
                    "SELECT COUNT(*) FROM visual_evidence WHERE meeting_id = ?",
                    (meeting_id,),
                ).fetchone()[0],
                "changes": connection.execute(
                    "SELECT COUNT(*) FROM changes WHERE to_meeting_id = ?",
                    (meeting_id,),
                ).fetchone()[0],
                "unresolved_action_items": connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM action_items
                    WHERE meeting_id = ? AND status IN ('pending', 'in_progress')
                    """,
                    (meeting_id,),
                ).fetchone()[0],
            }
            transcript_segments = [
                dict(row) for row in connection.execute(
                    """
                    SELECT id, speaker, text, start_time_seconds
                    FROM transcript_segments WHERE meeting_id = ? ORDER BY id
                    """, (meeting_id,),
                ).fetchall()
            ]
            tracked_values = [
                {
                    "field_name": row["field_name"],
                    "raw_value": row["field_value"],
                    "normalized_value": row["normalized_value"],
                    "budget_amount_minor": row["budget_amount_minor"],
                    "currency_code": row["currency_code"],
                    "mentioned_by": row["decided_by"],
                    "timestamp_seconds": row["timestamp_seconds"],
                    "source_type": row["source_type"],
                    "is_canonical": bool(row["is_canonical"]),
                    "evidence_id": row["visual_evidence_id"],
                }
                for row in connection.execute(
                    """
                    SELECT field_name, field_value, normalized_value,
                           budget_amount_minor, currency_code, decided_by,
                           timestamp_seconds, source_type, is_canonical,
                           visual_evidence_id
                    FROM decisions
                    WHERE meeting_id = ? AND field_name IN ('budget','deadline','owner')
                    ORDER BY id
                    """, (meeting_id,),
                ).fetchall()
            ]
            decisions = [
                {
                    "id": row["id"], "text": row["field_value"],
                    "decided_by": row["decided_by"],
                    "timestamp_seconds": row["timestamp_seconds"],
                    "source_type": row["source_type"],
                    "evidence_id": row["visual_evidence_id"],
                }
                for row in connection.execute(
                    """
                    SELECT id, field_value, decided_by, timestamp_seconds,
                           source_type, visual_evidence_id
                    FROM decisions WHERE meeting_id = ? AND field_name = 'decision_text'
                    ORDER BY id
                    """, (meeting_id,),
                ).fetchall()
            ]
            action_items = [
                dict(row) for row in connection.execute(
                    """
                    SELECT id, description, owner, due_date, status
                    FROM action_items WHERE meeting_id = ? ORDER BY id
                    """, (meeting_id,),
                ).fetchall()
            ]
            visual_evidence = [
                {
                    "id": row["id"], "timestamp_seconds": row["timestamp_seconds"],
                    "evidence_type": row["evidence_type"], "text": row["raw_ocr_text"],
                    "confidence": row["confidence"],
                    "image_url": f"/api/evidence/{row['id']}/image",
                }
                for row in connection.execute(
                    """
                    SELECT id, timestamp_seconds, evidence_type, raw_ocr_text, confidence
                    FROM visual_evidence WHERE meeting_id = ? ORDER BY timestamp_seconds, id
                    """, (meeting_id,),
                ).fetchall()
            ]
            changes = [
                dict(row) for row in connection.execute(
                    """
                    SELECT id, field_name, old_value, new_value, reason, changed_by,
                           source_type, timestamp_seconds, from_meeting_id,
                           to_meeting_id, detected_at
                    FROM changes WHERE to_meeting_id = ? ORDER BY id
                    """, (meeting_id,),
                ).fetchall()
            ]
            return {
                **dict(meeting),
                "participants": participants,
                "counts": counts,
                "transcript_segments": transcript_segments,
                "tracked_values": tracked_values,
                "decisions": decisions,
                "action_items": action_items,
                "visual_evidence": visual_evidence,
                "changes": changes,
            }

    def get_recording_state(self, meeting_id: str) -> dict | None:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT status, recording_path
                FROM meetings
                WHERE id = ?
                """,
                (meeting_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def mark_recording_uploaded(self, meeting_id: str, recording_path: str) -> None:
        now = utc_now_text()
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE meetings
                SET recording_path = ?, status = 'uploaded', updated_at = ?
                WHERE id = ? AND status = 'created' AND recording_path IS NULL
                """,
                (recording_path, now, meeting_id),
            )
            if cursor.rowcount == 1:
                return

            meeting_exists = connection.execute(
                "SELECT 1 FROM meetings WHERE id = ?",
                (meeting_id,),
            ).fetchone()
            if meeting_exists is None:
                raise MeetingNotFoundError
            raise RecordingAlreadyExistsError

    def claim_processing(self, meeting_id: str) -> dict:
        now = utc_now_text()
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT id, project_id, meeting_date, status, recording_path FROM meetings WHERE id = ?",
                (meeting_id,),
            ).fetchone()
            if row is None:
                raise MeetingNotFoundError
            if row["status"] != "uploaded" or not row["recording_path"]:
                raise MeetingNotProcessableError(row["status"])
            cursor = connection.execute(
                """
                UPDATE meetings SET status = 'processing', processing_error = NULL,
                       updated_at = ? WHERE id = ? AND status = 'uploaded'
                """, (now, meeting_id),
            )
            if cursor.rowcount != 1:
                raise MeetingNotProcessableError("processing")
            return dict(row)

    def mark_processing_failed(self, meeting_id: str, safe_error: str) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """
                UPDATE meetings SET status = 'failed', processing_error = ?, updated_at = ?
                WHERE id = ? AND status = 'processing'
                """, (safe_error[:200], utc_now_text(), meeting_id),
            )

    def previous_canonical_values(self, project_id: str, meeting_id: str) -> dict[str, dict]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT d.*, m.meeting_date
                FROM decisions d
                JOIN meetings m ON m.id = d.meeting_id
                JOIN meetings current ON current.id = ?
                WHERE d.project_id = ? AND d.meeting_id <> ? AND d.is_canonical = 1
                  AND d.field_name IN ('budget','deadline','owner')
                  AND (m.meeting_date < current.meeting_date OR
                       (m.meeting_date = current.meeting_date AND m.created_at < current.created_at))
                ORDER BY m.meeting_date DESC, d.created_at DESC, d.id DESC
                """, (meeting_id, project_id, meeting_id),
            ).fetchall()
        result = {}
        for row in rows:
            result.setdefault(row["field_name"], dict(row))
        return result

    def persist_processing(self, meeting_id: str, payload: dict) -> None:
        now = utc_now_text()
        with self.database.transaction() as connection:
            state = connection.execute(
                "SELECT status FROM meetings WHERE id = ?", (meeting_id,),
            ).fetchone()
            if state is None:
                raise MeetingNotFoundError
            if state["status"] != "processing":
                raise MeetingNotProcessableError(state["status"])
            connection.executemany(
                """
                INSERT INTO transcript_segments
                    (meeting_id, speaker, text, start_time_seconds, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [(meeting_id, x["speaker"], x["text"], x.get("start_time_seconds"), now)
                 for x in payload["transcript_segments"]],
            )
            connection.executemany(
                """
                INSERT INTO visual_evidence
                    (id, meeting_id, timestamp_seconds, evidence_type, raw_ocr_text,
                     confidence, image_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [(x["id"], meeting_id, x["timestamp_seconds"], x["evidence_type"],
                  x["raw_ocr_text"], x["confidence"], x["image_path"], now)
                 for x in payload["visual_evidence"]],
            )
            connection.executemany(
                """
                INSERT INTO decisions
                    (project_id, meeting_id, field_name, field_value, normalized_value,
                     budget_amount_minor, currency_code, decided_by, timestamp_seconds,
                     source_type, reasoning_snippet, visual_evidence_id, is_canonical, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [(x["project_id"], meeting_id, x["field_name"], x["field_value"],
                  x.get("normalized_value"), x.get("budget_amount_minor"), x.get("currency_code"),
                  x.get("decided_by"), x.get("timestamp_seconds"), x["source_type"],
                  x.get("reasoning_snippet"), x.get("visual_evidence_id"),
                  int(x.get("is_canonical", False)), now) for x in payload["decisions"]],
            )
            connection.executemany(
                """
                INSERT INTO action_items
                    (meeting_id, description, owner, due_date, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'pending', ?, ?)
                """,
                [(meeting_id, x["description"], x["owner"], x.get("due_date"), now, now)
                 for x in payload["action_items"]],
            )
            connection.executemany(
                """
                INSERT INTO changes
                    (project_id, field_name, old_value, new_value,
                     old_budget_amount_minor, new_budget_amount_minor, currency_code,
                     from_meeting_id, to_meeting_id, reason, changed_by, source_type,
                     timestamp_seconds, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [(x["project_id"], x["field_name"], x["old_value"], x["new_value"],
                  x.get("old_budget_amount_minor"), x.get("new_budget_amount_minor"),
                  x.get("currency_code"), x["from_meeting_id"], meeting_id,
                  x.get("reason"), x.get("changed_by"), x["source_type"],
                  x.get("timestamp_seconds"), now) for x in payload["changes"]],
            )
            cursor = connection.execute(
                """
                UPDATE meetings SET transcript = ?, summary = ?, status = 'processed',
                       processing_error = NULL, updated_at = ?
                WHERE id = ? AND status = 'processing'
                """, (payload["transcript"], payload["summary"], now, meeting_id),
            )
            if cursor.rowcount != 1:
                raise MeetingNotProcessableError("unknown")

    def get_project_history(self, project_id: str) -> list[dict] | None:
        with self.database.connection() as connection:
            if connection.execute("SELECT 1 FROM projects WHERE id = ?", (project_id,)).fetchone() is None:
                return None
            rows = connection.execute(
                """
                SELECT d.field_name, d.meeting_id, m.title AS meeting_title,
                       m.meeting_date, d.field_value AS raw_value, d.normalized_value,
                       d.budget_amount_minor, d.currency_code, d.source_type,
                       d.decided_by AS speaker, d.timestamp_seconds,
                       d.visual_evidence_id AS evidence_id, d.is_canonical,
                       c.reason
                FROM decisions d JOIN meetings m ON m.id = d.meeting_id
                LEFT JOIN changes c ON c.to_meeting_id = d.meeting_id
                                   AND c.field_name = d.field_name
                WHERE d.project_id = ? AND d.field_name IN ('budget','deadline','owner')
                ORDER BY m.meeting_date, d.created_at, d.id
                """, (project_id,),
            ).fetchall()
        return [{**dict(row), "is_canonical": bool(row["is_canonical"]),
                 "image_url": f"/api/evidence/{row['evidence_id']}/image" if row["evidence_id"] else None}
                for row in rows]

    def get_evidence_path(self, evidence_id: str) -> str | None:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT image_path FROM visual_evidence WHERE id = ?", (evidence_id,),
            ).fetchone()
        return row["image_path"] if row else None

    def search_records(self, project_id: str) -> list[dict] | None:
        with self.database.connection() as connection:
            if connection.execute("SELECT 1 FROM projects WHERE id = ?", (project_id,)).fetchone() is None:
                return None
            rows = connection.execute(
                """
                SELECT m.id AS meeting_id, m.title AS meeting_title, m.meeting_date,
                       s.speaker, s.start_time_seconds AS timestamp_seconds,
                       'transcript' AS source_type, s.text, NULL AS evidence_id
                FROM transcript_segments s JOIN meetings m ON m.id = s.meeting_id
                WHERE m.project_id = ?
                UNION ALL
                SELECT m.id, m.title, m.meeting_date, NULL, v.timestamp_seconds,
                       'visual', v.raw_ocr_text, v.id
                FROM visual_evidence v JOIN meetings m ON m.id = v.meeting_id
                WHERE m.project_id = ?
                """, (project_id, project_id),
            ).fetchall()
        return [dict(row) for row in rows]
