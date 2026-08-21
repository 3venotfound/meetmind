import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.config import BACKEND_DIR, Settings
from app.main import create_app


class IntegrationStartupTests(unittest.TestCase):
    def test_startup_does_not_import_or_start_ai_or_cv_dependencies(self):
        dependency_modules = {"google.genai", "cv2", "easyocr", "pipeline"}
        self.assertTrue(dependency_modules.isdisjoint(sys.modules))
        with tempfile.TemporaryDirectory(
            prefix=".meetmind-test-",
            dir=BACKEND_DIR,
        ) as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                database_path=root / "test.db",
                storage_root=root / "storage",
                gemini_api_key="startup-test-secret",
                ai_python_executable="",
                cv_python_executable="",
            )
            create_process = AsyncMock()
            with patch("asyncio.create_subprocess_exec", create_process):
                with TestClient(create_app(settings)) as client:
                    self.assertIsNotNone(client.app.state.ai_adapter)
                    self.assertIsNotNone(client.app.state.cv_adapter)

            create_process.assert_not_awaited()
            self.assertTrue(dependency_modules.isdisjoint(sys.modules))
