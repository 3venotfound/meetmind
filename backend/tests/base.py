import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


class ApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        temporary_root = Path(self.temporary_directory.name)
        settings = Settings(
            database_path=temporary_root / "test.db",
            storage_root=temporary_root / "storage",
            cors_origins="http://testserver",
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
