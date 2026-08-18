import sqlite3
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from tests.base import ApiTestCase


class RecordingUploadTests(ApiTestCase):
    def upload(
        self,
        meeting_id: str,
        filename: str = "meeting.mp4",
        content: bytes = b"small mp4 recording",
        content_type: str = "video/mp4",
    ):
        return self.client.post(
            f"/api/meetings/{meeting_id}/recording",
            files={"file": (filename, content, content_type)},
        )

    def database_recording_state(self, meeting_id: str):
        database = self.client.app.state.database
        with database.connection() as connection:
            return connection.execute(
                "SELECT status, recording_path FROM meetings WHERE id = ?",
                (meeting_id,),
            ).fetchone()

    def assert_no_part_files(self) -> None:
        self.assertEqual(list(self.storage_root.rglob("*.part")), [])

    def test_successful_mp4_upload(self) -> None:
        meeting = self.create_meeting()
        content = b"mp4 recording"
        response = self.upload(meeting["id"], content=content)

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(
            response.json(),
            {
                "meeting_id": meeting["id"],
                "status": "uploaded",
                "original_filename": "meeting.mp4",
                "content_type": "video/mp4",
                "size_bytes": len(content),
            },
        )
        state = self.database_recording_state(meeting["id"])
        self.assertEqual(state["status"], "uploaded")
        self.assertFalse(Path(state["recording_path"]).is_absolute())
        self.assertNotIn("\\", state["recording_path"])
        stored_path = (Path(__file__).resolve().parent.parent / state["recording_path"]).resolve()
        self.assertEqual(stored_path.read_bytes(), content)
        self.assert_no_part_files()

        retrieved = self.client.get(f"/api/meetings/{meeting['id']}")
        self.assertEqual(retrieved.status_code, 200)
        self.assertEqual(retrieved.json()["status"], "uploaded")

    def test_successful_webm_upload(self) -> None:
        meeting = self.create_meeting()
        content = b"webm recording"
        response = self.upload(
            meeting["id"],
            filename="meeting.webm",
            content=content,
            content_type="video/webm",
        )

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["content_type"], "video/webm")
        state = self.database_recording_state(meeting["id"])
        stored_path = (Path(__file__).resolve().parent.parent / state["recording_path"]).resolve()
        self.assertEqual(stored_path.name, "recording.webm")
        self.assertEqual(stored_path.read_bytes(), content)
        self.assert_no_part_files()

    def test_unknown_meeting_returns_404(self) -> None:
        response = self.upload(str(uuid4()))
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Meeting not found"})
        self.assert_no_part_files()

    def test_unsupported_extension_returns_415(self) -> None:
        meeting = self.create_meeting()
        response = self.upload(meeting["id"], filename="meeting.avi")
        self.assertEqual(response.status_code, 415)
        self.assert_no_part_files()

    def test_unsupported_mime_type_returns_415(self) -> None:
        meeting = self.create_meeting()
        response = self.upload(
            meeting["id"],
            filename="meeting.mp4",
            content_type="application/octet-stream",
        )
        self.assertEqual(response.status_code, 415)
        self.assert_no_part_files()

    def test_mismatched_extension_and_mime_type_returns_415(self) -> None:
        meeting = self.create_meeting()
        response = self.upload(
            meeting["id"],
            filename="meeting.mp4",
            content_type="video/webm",
        )
        self.assertEqual(response.status_code, 415)
        self.assert_no_part_files()

    def test_empty_file_is_rejected(self) -> None:
        meeting = self.create_meeting()
        response = self.upload(meeting["id"], content=b"")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Recording file is empty"})
        state = self.database_recording_state(meeting["id"])
        self.assertEqual(state["status"], "created")
        self.assertIsNone(state["recording_path"])
        self.assert_no_part_files()

    def test_oversized_file_returns_413(self) -> None:
        meeting = self.create_meeting()
        response = self.upload(meeting["id"], content=b"x" * 33)
        self.assertEqual(response.status_code, 413)
        state = self.database_recording_state(meeting["id"])
        self.assertEqual(state["status"], "created")
        self.assertIsNone(state["recording_path"])
        self.assert_no_part_files()

    def test_duplicate_upload_preserves_original_recording(self) -> None:
        meeting = self.create_meeting()
        first_content = b"original recording"
        first = self.upload(meeting["id"], content=first_content)
        self.assertEqual(first.status_code, 201, first.text)
        original_state = self.database_recording_state(meeting["id"])
        original_path = (
            Path(__file__).resolve().parent.parent / original_state["recording_path"]
        ).resolve()

        duplicate = self.upload(
            meeting["id"],
            filename="replacement.webm",
            content=b"replacement",
            content_type="video/webm",
        )
        self.assertEqual(duplicate.status_code, 409)
        self.assertEqual(original_path.read_bytes(), first_content)
        current_state = self.database_recording_state(meeting["id"])
        self.assertEqual(current_state["recording_path"], original_state["recording_path"])
        self.assertEqual(current_state["status"], "uploaded")
        self.assert_no_part_files()

    def test_client_filename_cannot_control_stored_path(self) -> None:
        meeting = self.create_meeting()
        response = self.upload(
            meeting["id"],
            filename="../../outside/meeting.webm",
            content=b"safe webm",
            content_type="video/webm",
        )
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["original_filename"], "meeting.webm")
        state = self.database_recording_state(meeting["id"])
        self.assertTrue(state["recording_path"].endswith(f"/{meeting['id']}/recording.webm"))
        self.assertNotIn("..", state["recording_path"])
        self.assert_no_part_files()

    def test_partial_file_is_removed_when_write_fails(self) -> None:
        meeting = self.create_meeting()
        storage = self.client.app.state.recording_storage

        def fail_after_partial_write(destination, chunk):
            destination.write(chunk[:4])
            raise OSError("simulated storage failure")

        with patch.object(storage, "_write_chunk", side_effect=fail_after_partial_write):
            with self.assertLogs("app.api.meetings", level="ERROR"):
                response = self.upload(meeting["id"])

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Recording upload failed"})
        state = self.database_recording_state(meeting["id"])
        self.assertEqual(state["status"], "created")
        self.assertIsNone(state["recording_path"])
        self.assertEqual(list(self.storage_root.rglob("recording.*")), [])
        self.assert_no_part_files()

    def test_new_file_is_removed_when_database_update_fails(self) -> None:
        meeting = self.create_meeting()
        repository = self.client.app.state.repository

        with patch.object(
            repository,
            "mark_recording_uploaded",
            side_effect=sqlite3.OperationalError("simulated database failure"),
        ):
            with self.assertLogs("app.api.meetings", level="ERROR"):
                response = self.upload(meeting["id"])

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Recording upload failed"})
        state = self.database_recording_state(meeting["id"])
        self.assertEqual(state["status"], "created")
        self.assertIsNone(state["recording_path"])
        self.assertEqual(list(self.storage_root.rglob("recording.*")), [])
        self.assert_no_part_files()

    def test_preexisting_destination_is_not_deleted(self) -> None:
        meeting = self.create_meeting()
        destination = self.storage_root / "recordings" / meeting["id"] / "recording.mp4"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"preexisting file")

        response = self.upload(meeting["id"], content=b"new upload")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(destination.read_bytes(), b"preexisting file")
        state = self.database_recording_state(meeting["id"])
        self.assertEqual(state["status"], "created")
        self.assertIsNone(state["recording_path"])
        self.assert_no_part_files()
