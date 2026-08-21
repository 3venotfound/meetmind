import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import BACKEND_DIR, Settings
from app.main import create_app


class ApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix=".meetmind-test-",
            dir=BACKEND_DIR,
        )
        temporary_root = Path(self.temporary_directory.name)
        self.temporary_root = temporary_root
        self.storage_root = temporary_root / "storage"
        settings = Settings(
            database_path=temporary_root / "test.db",
            storage_root=self.storage_root,
            max_upload_size_bytes=32,
            cors_origins=(
                "http://testserver,https://meetmiind.netlify.app,"
                "http://127.0.0.1:5500,http://localhost:5500"
            ),
        )
        self.client_context = TestClient(create_app(settings))
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.temporary_directory.cleanup()

    def create_project(self, **overrides) -> dict:
        payload = {
            "name": "Digital Banking Revamp",
            "client_org": "FINOVA BANK",
            "target_industry": "Financial Services",
        }
        payload.update(overrides)
        response = self.client.post("/api/projects", json=payload)
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def create_meeting(self, **overrides) -> dict:
        project = self.create_project()
        payload = {
            "project_id": project["id"],
            "title": "Recording test meeting",
            "meeting_date": "2026-08-18",
            "meeting_number": 1,
            "participants": [],
        }
        payload.update(overrides)
        response = self.client.post("/api/meetings", json=payload)
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()
