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
                    status, summary, created_at, updated_at
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
            return {
                **dict(meeting),
                "participants": participants,
                "counts": counts,
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
