import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.config import BACKEND_DIR, Settings
from app.database import Database
from app.repositories import Repository
from app.schemas import MeetingCreate, ProjectCreate


class DatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "test.db"
        self.database = Database(self.database_path)
        self.database.initialize()
        self.repository = Repository(self.database)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_relative_paths_resolve_from_backend_directory(self) -> None:
        settings = Settings(
            database_path=Path("storage/custom.db"),
            storage_root=Path("runtime-storage"),
        )
        self.assertEqual(
            settings.resolved_database_path,
            (BACKEND_DIR / "storage/custom.db").resolve(),
        )
        self.assertEqual(
            settings.resolved_storage_root,
            (BACKEND_DIR / "runtime-storage").resolve(),
        )
        self.assertEqual(settings.max_upload_size_bytes, 524_288_000)

    def test_foreign_keys_are_enabled_for_every_connection(self) -> None:
        with self.database.connection() as connection:
            enabled = connection.execute("PRAGMA foreign_keys").fetchone()[0]
        with self.database.transaction() as connection:
            enabled_in_transaction = connection.execute("PRAGMA foreign_keys").fetchone()[0]
        self.assertEqual(enabled, 1)
        self.assertEqual(enabled_in_transaction, 1)

    def test_foreign_key_constraint_is_enforced(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            with self.database.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO meetings (
                        id, project_id, title, meeting_date, status,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "meeting-id",
                        "missing-project",
                        "Test meeting",
                        "2026-08-18",
                        "created",
                        "2026-08-18T00:00:00Z",
                        "2026-08-18T00:00:00Z",
                    ),
                )

    def test_participant_failure_rolls_back_meeting(self) -> None:
        project = self.repository.create_project(
            ProjectCreate(name="Project", client_org="Client")
        )
        meeting = MeetingCreate(
            project_id=project["id"],
            title="Rollback test",
            meeting_date="2026-08-18",
            meeting_number=1,
        )
        invalid_participant = SimpleNamespace(name=None, role=None)

        with self.assertRaises(sqlite3.IntegrityError):
            self.repository.create_meeting(meeting, participants=[invalid_participant])

        with self.database.connection() as connection:
            meeting_count = connection.execute(
                "SELECT COUNT(*) FROM meetings WHERE project_id = ?",
                (project["id"],),
            ).fetchone()[0]
            participant_count = connection.execute(
                "SELECT COUNT(*) FROM participants",
            ).fetchone()[0]
        self.assertEqual(meeting_count, 0)
        self.assertEqual(participant_count, 0)

    def test_schema_contains_all_phase_one_tables(self) -> None:
        expected = {
            "projects",
            "meetings",
            "participants",
            "transcript_segments",
            "visual_evidence",
            "decisions",
            "changes",
            "action_items",
        }
        with self.database.connection() as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        actual = {row["name"] for row in rows}
        self.assertTrue(expected.issubset(actual))

    def test_unresolved_issues_count_pending_and_in_progress_actions(self) -> None:
        project = self.repository.create_project(
            ProjectCreate(name="Project", client_org="Client")
        )
        meeting = self.repository.create_meeting(
            MeetingCreate(
                project_id=project["id"],
                title="Action item test",
                meeting_date="2026-08-18",
            )
        )
        now = "2026-08-18T00:00:00Z"
        with self.database.transaction() as connection:
            connection.executemany(
                """
                INSERT INTO action_items (
                    meeting_id, description, owner, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (meeting["id"], "Pending", "Sarah", "pending", now, now),
                    (meeting["id"], "Active", "Ahmad", "in_progress", now, now),
                    (meeting["id"], "Done", "John", "completed", now, now),
                ],
            )

        updated_project = self.repository.get_project(project["id"])
        self.assertIsNotNone(updated_project)
        self.assertEqual(updated_project["stats"]["unresolved_issues"], 2)
