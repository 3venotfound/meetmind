import json
import sys
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from app.config import BACKEND_DIR
from app.integrations.cv_adapter import CVAdapter, normalize_timestamp
from app.integrations.errors import IntegrationError, IntegrationErrorCode
from app.integrations.subprocess_runner import (
    ProcessResult,
    WorkerLaunchError,
    WorkerTimeoutError,
)


AUTO_RESULT = object()


class FakeCVRunner:
    def __init__(
        self,
        payload=AUTO_RESULT,
        returncode=0,
        timeout=False,
        launch_error=False,
    ):
        self.payload = payload
        self.returncode = returncode
        self.timeout = timeout
        self.launch_error = launch_error
        self.calls = []

    async def run(
        self,
        command,
        *,
        cwd,
        timeout_seconds,
        extra_environment=None,
    ):
        self.calls.append((command, cwd, timeout_seconds, extra_environment))
        if self.timeout:
            raise WorkerTimeoutError
        if self.launch_error:
            raise WorkerLaunchError("test launch failure")
        payload = self.payload
        if payload is AUTO_RESULT and self.returncode == 0:
            evidence_directory = cwd / "evidence"
            evidence_directory.mkdir()
            (evidence_directory / "board.jpg").write_bytes(b"image")
            request = json.loads(Path(command[2]).read_text(encoding="utf-8"))
            payload = {
                "meeting_id": request["meeting_id"],
                "visual_evidence": [
                    {
                        "timestamp": "03:15",
                        "type": "whiteboard",
                        "text": "Budget: RM70,000",
                        "confidence": 0.86,
                        "image_path": "evidence/board.jpg",
                    }
                ],
            }
        if payload is not None:
            result_path = Path(command[3])
            if isinstance(payload, bytes):
                result_path.write_bytes(payload)
            else:
                result_path.write_text(json.dumps(payload), encoding="utf-8")
        return ProcessResult(self.returncode, b"", b"", False, False)


class CVAdapterTests(unittest.IsolatedAsyncioTestCase):
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
            / "recording.webm"
        )
        self.recording_path.parent.mkdir(parents=True)
        self.recording_path.write_bytes(b"test video")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def adapter(self, runner, **overrides):
        arguments = {
            "storage_root": self.storage_root,
            "runner": runner,
            "max_result_bytes": 1024 * 1024,
        }
        arguments.update(overrides)
        return CVAdapter(**arguments)

    async def test_success_normalizes_timestamp_and_validates_evidence(self):
        runner = FakeCVRunner()
        adapter = self.adapter(runner)
        result = await adapter.process_recording(self.recording_path, self.meeting_id)

        self.assertEqual(result.visual_evidence[0].timestamp_seconds, 195)
        self.assertEqual(result.visual_evidence[0].confidence, 0.86)
        self.assertFalse(Path(result.visual_evidence[0].image_path).is_absolute())
        run_directory = result._owned_run_directory.path
        self.assertTrue(run_directory.exists())
        adapter.cleanup_validated_run(result)
        self.assertFalse(run_directory.exists())

    async def test_configurable_python_executable(self):
        runner = FakeCVRunner(payload={"meeting_id": str(self.meeting_id), "visual_evidence": []})
        adapter = self.adapter(runner, python_executable=sys.executable)
        await adapter.process_recording(self.recording_path, self.meeting_id)
        self.assertEqual(Path(runner.calls[0][0][0]), Path(sys.executable).resolve())

    def test_timestamp_normalization(self):
        self.assertEqual(normalize_timestamp("00:42"), 42)
        self.assertEqual(normalize_timestamp("03:15"), 195)
        self.assertEqual(normalize_timestamp("100:05"), 6005)

    async def test_malformed_cv_results_are_rejected(self):
        cases = [
            {"meeting_id": str(self.meeting_id), "visual_evidence": [{"bad": "shape"}]},
            {
                "meeting_id": str(self.meeting_id),
                "visual_evidence": [{
                    "timestamp": "03:60", "type": "slide", "text": "x",
                    "confidence": 0.5, "image_path": "evidence/x.jpg",
                }],
            },
            {
                "meeting_id": str(self.meeting_id),
                "visual_evidence": [{
                    "timestamp": "00:01", "type": "slide", "text": "x",
                    "confidence": 1.5, "image_path": "evidence/x.jpg",
                }],
            },
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(IntegrationError) as context:
                    await self.adapter(FakeCVRunner(payload=payload)).process_recording(
                        self.recording_path,
                        self.meeting_id,
                    )
                self.assertEqual(context.exception.code, IntegrationErrorCode.INVALID_RESULT)

    async def test_unsafe_returned_paths_are_rejected(self):
        unsafe_paths = [
            "../outside.jpg",
            "C:\\outside.jpg",
            "/outside.jpg",
            "evidence/../../outside.jpg",
        ]
        for image_path in unsafe_paths:
            payload = {
                "meeting_id": str(self.meeting_id),
                "visual_evidence": [{
                    "timestamp": "00:01", "type": "slide", "text": "x",
                    "confidence": 0.5, "image_path": image_path,
                }],
            }
            with self.subTest(image_path=image_path):
                with self.assertRaises(IntegrationError) as context:
                    await self.adapter(FakeCVRunner(payload=payload)).process_recording(
                        self.recording_path,
                        self.meeting_id,
                    )
                self.assertEqual(context.exception.code, IntegrationErrorCode.UNSAFE_PATH)

    async def test_missing_and_oversized_result_files(self):
        with self.assertRaises(IntegrationError) as missing:
            await self.adapter(FakeCVRunner(payload=None, returncode=0)).process_recording(
                self.recording_path,
                self.meeting_id,
            )
        self.assertEqual(missing.exception.code, IntegrationErrorCode.INVALID_RESULT)

        with self.assertRaises(IntegrationError) as oversized:
            await self.adapter(
                FakeCVRunner(payload=b"x" * 65),
                max_result_bytes=64,
            ).process_recording(self.recording_path, self.meeting_id)
        self.assertEqual(oversized.exception.code, IntegrationErrorCode.INVALID_RESULT)

    async def test_subprocess_failure_and_timeout_are_sanitized(self):
        with self.assertRaises(IntegrationError) as launch_failure:
            await self.adapter(FakeCVRunner(launch_error=True)).process_recording(
                self.recording_path,
                self.meeting_id,
            )
        self.assertEqual(
            launch_failure.exception.code,
            IntegrationErrorCode.UNAVAILABLE,
        )
        self.assertEqual(
            launch_failure.exception.diagnostic.category,
            "worker_unavailable",
        )

        with self.assertRaises(IntegrationError) as unavailable:
            await self.adapter(FakeCVRunner(payload=None, returncode=20)).process_recording(
                self.recording_path,
                self.meeting_id,
            )
        self.assertEqual(unavailable.exception.code, IntegrationErrorCode.UNAVAILABLE)

        with self.assertRaises(IntegrationError) as failed:
            await self.adapter(FakeCVRunner(payload=None, returncode=21)).process_recording(
                self.recording_path,
                self.meeting_id,
            )
        self.assertEqual(failed.exception.code, IntegrationErrorCode.EXECUTION_FAILED)

        with self.assertRaises(IntegrationError) as timeout:
            await self.adapter(FakeCVRunner(timeout=True)).process_recording(
                self.recording_path,
                self.meeting_id,
            )
        self.assertEqual(timeout.exception.code, IntegrationErrorCode.TIMEOUT)

    async def test_worker_diagnostic_is_preserved_and_public_error_is_generic(self):
        payload = {
            "integration_diagnostic": {
                "component": "cv",
                "stage": "video_decode",
                "exception_class": "OSError",
                "category": "io_error",
                "message": "Video frames could not be decoded.",
            }
        }
        runner = FakeCVRunner(payload=payload, returncode=22)

        with self.assertLogs("app.integrations.cv_adapter", level="WARNING") as logs:
            with self.assertRaises(IntegrationError) as context:
                await self.adapter(runner).process_recording(
                    self.recording_path,
                    self.meeting_id,
                )

        error = context.exception
        self.assertEqual(error.code, IntegrationErrorCode.EXECUTION_FAILED)
        self.assertEqual(str(error), "CV processing failed")
        self.assertEqual(error.diagnostic.component, "cv")
        self.assertEqual(error.diagnostic.stage, "video_decode")
        self.assertEqual(error.diagnostic.category, "io_error")
        combined = "\n".join(logs.output)
        self.assertIn("component=cv", combined)
        self.assertIn("stage=video_decode", combined)
        self.assertNotIn(str(self.recording_path), combined)

    async def test_missing_or_oversized_diagnostic_uses_safe_bootstrap_fallback(self):
        oversized = {
            "integration_diagnostic": {
                "component": "cv",
                "stage": "pipeline_execution",
                "exception_class": "RuntimeError",
                "category": "execution_failed",
                "message": "x" * 5000,
            }
        }
        for payload in (None, oversized):
            with self.subTest(payload_present=payload is not None):
                with self.assertRaises(IntegrationError) as context:
                    await self.adapter(
                        FakeCVRunner(payload=payload, returncode=21)
                    ).process_recording(self.recording_path, self.meeting_id)
                diagnostic = context.exception.diagnostic
                self.assertEqual(diagnostic.component, "cv")
                self.assertEqual(diagnostic.stage, "cv_worker_bootstrap")
                self.assertEqual(diagnostic.category, "worker_exit_nonzero")

    async def test_cleanup_is_restricted_to_owned_run(self):
        sentinel = self.storage_root / "integration_runs" / "keep.txt"
        sentinel.parent.mkdir(parents=True)
        sentinel.write_text("keep", encoding="utf-8")
        adapter = self.adapter(FakeCVRunner())
        result = await adapter.process_recording(self.recording_path, self.meeting_id)
        adapter.cleanup_validated_run(result)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
