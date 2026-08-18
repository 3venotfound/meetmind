import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from pydantic import SecretStr

from app.config import BACKEND_DIR
from app.integrations.ai_adapter import AIAdapter
from app.integrations.errors import IntegrationError, IntegrationErrorCode
from app.integrations.subprocess_runner import ProcessResult


VALID_AI_RESULT = {
    "transcript": "Sarah: The budget is now RM70,000.",
    "summary": "The budget was updated.",
    "decisions": [
        {
            "text": "Use a budget of RM70,000",
            "decided_by": "Sarah",
            "timestamp_seconds": 42,
        }
    ],
    "action_items": [
        {
            "description": "Update the plan",
            "owner": "Ahmad",
            "due_date": "20 August 2026",
        }
    ],
    "budget": {
        "value": "RM70,000",
        "mentioned_by": "Sarah",
        "timestamp_seconds": 42,
    },
    "deadline": {"value": None, "mentioned_by": None},
    "owner": {"value": "Ahmad", "mentioned_by": "Sarah"},
}


class FakeRunner:
    def __init__(self, payload=VALID_AI_RESULT, returncode=0):
        self.payload = payload
        self.returncode = returncode
        self.calls = []
        self.request_payload = None

    async def run(
        self,
        command,
        *,
        cwd,
        timeout_seconds,
        extra_environment=None,
    ):
        self.calls.append(
            {
                "command": command,
                "cwd": cwd,
                "timeout": timeout_seconds,
                "environment": extra_environment,
            }
        )
        self.request_payload = json.loads(Path(command[2]).read_text(encoding="utf-8"))
        if self.payload is not None:
            result_path = Path(command[3])
            if isinstance(self.payload, bytes):
                result_path.write_bytes(self.payload)
            else:
                result_path.write_text(json.dumps(self.payload), encoding="utf-8")
        return ProcessResult(self.returncode, b"", b"", False, False)


class AIAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix=".meetmind-test-",
            dir=BACKEND_DIR,
        )
        self.root = Path(self.temporary_directory.name)
        self.storage_root = self.root / "storage"
        self.meeting_id = uuid4()
        self.recording_path = (
            self.storage_root
            / "recordings"
            / str(self.meeting_id)
            / "recording.mp4"
        )
        self.recording_path.parent.mkdir(parents=True)
        self.recording_path.write_bytes(b"test video")
        self.secret = "test-secret-must-not-leak"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def adapter(self, runner, **overrides):
        arguments = {
            "storage_root": self.storage_root,
            "api_key": SecretStr(self.secret),
            "runner": runner,
            "max_result_bytes": 1024 * 1024,
        }
        arguments.update(overrides)
        return AIAdapter(**arguments)

    async def test_success_validates_result_and_cleans_worker_files(self):
        runner = FakeRunner()
        adapter = self.adapter(runner)
        result = await adapter.extract_recording(self.recording_path)

        self.assertEqual(result.budget.value, "RM70,000")
        self.assertEqual(result.decisions[0].timestamp_seconds, 42)
        call = runner.calls[0]
        self.assertIsInstance(call["command"], list)
        self.assertNotIn(self.secret, " ".join(call["command"]))
        self.assertNotIn(self.secret, json.dumps(runner.request_payload))
        self.assertEqual(call["environment"]["GEMINI_API_KEY"], self.secret)
        run_root = self.storage_root / "integration_runs" / "ai"
        self.assertEqual(list(run_root.iterdir()), [])

    async def test_configured_and_default_python_executables(self):
        configured_runner = FakeRunner()
        adapter = self.adapter(
            configured_runner,
            python_executable=sys.executable,
        )
        await adapter.extract_recording(self.recording_path)
        self.assertEqual(
            Path(configured_runner.calls[0]["command"][0]),
            Path(sys.executable).resolve(),
        )

        default_runner = FakeRunner()
        await self.adapter(default_runner).extract_recording(self.recording_path)
        self.assertEqual(
            Path(default_runner.calls[0]["command"][0]),
            Path(sys.executable).resolve(),
        )

    async def test_invalid_configured_executable_is_rejected_before_runner(self):
        runner = FakeRunner()
        adapter = self.adapter(
            runner,
            python_executable=str(self.root / "missing-python.exe"),
        )
        with self.assertRaises(IntegrationError) as context:
            await adapter.extract_recording(self.recording_path)
        self.assertEqual(context.exception.code, IntegrationErrorCode.UNAVAILABLE)
        self.assertEqual(runner.calls, [])

    async def test_missing_api_key_does_not_start_worker(self):
        runner = FakeRunner()
        adapter = AIAdapter(
            storage_root=self.storage_root,
            api_key=None,
            runner=runner,
        )
        with self.assertRaises(IntegrationError) as context:
            await adapter.extract_recording(self.recording_path)
        self.assertEqual(context.exception.code, IntegrationErrorCode.NOT_CONFIGURED)
        self.assertEqual(runner.calls, [])

    async def test_missing_dependency_is_sanitized(self):
        runner = FakeRunner(payload=None, returncode=20)
        with self.assertRaises(IntegrationError) as context:
            await self.adapter(runner).extract_recording(self.recording_path)
        self.assertEqual(context.exception.code, IntegrationErrorCode.UNAVAILABLE)
        self.assertNotIn(self.secret, str(context.exception))

    async def test_malformed_result_is_rejected(self):
        runner = FakeRunner(payload={"summary": "missing fields"})
        with self.assertRaises(IntegrationError) as context:
            await self.adapter(runner).extract_recording(self.recording_path)
        self.assertEqual(context.exception.code, IntegrationErrorCode.INVALID_RESULT)

    async def test_missing_and_oversized_results_are_rejected(self):
        missing_runner = FakeRunner(payload=None)
        with self.assertRaises(IntegrationError) as missing:
            await self.adapter(missing_runner).extract_recording(self.recording_path)
        self.assertEqual(missing.exception.code, IntegrationErrorCode.INVALID_RESULT)

        oversized_runner = FakeRunner(payload=b"x" * 65)
        with self.assertRaises(IntegrationError) as oversized:
            await self.adapter(
                oversized_runner,
                max_result_bytes=64,
            ).extract_recording(self.recording_path)
        self.assertEqual(oversized.exception.code, IntegrationErrorCode.INVALID_RESULT)

    async def test_failure_cleanup_does_not_delete_outside_owned_run(self):
        sentinel = self.storage_root / "integration_runs" / "keep.txt"
        sentinel.parent.mkdir(parents=True)
        sentinel.write_text("keep", encoding="utf-8")
        runner = FakeRunner(payload=None, returncode=21)

        with self.assertRaises(IntegrationError) as context:
            await self.adapter(runner).extract_recording(self.recording_path)
        self.assertEqual(context.exception.code, IntegrationErrorCode.EXECUTION_FAILED)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    async def test_cancellation_cleans_adapter_owned_run(self):
        class CancellingRunner:
            async def run(self, command, **kwargs):
                raise asyncio.CancelledError

        with self.assertRaises(asyncio.CancelledError):
            await self.adapter(CancellingRunner()).extract_recording(self.recording_path)
        run_root = self.storage_root / "integration_runs" / "ai"
        self.assertEqual(list(run_root.iterdir()), [])
