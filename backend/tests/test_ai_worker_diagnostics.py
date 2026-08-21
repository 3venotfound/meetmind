import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import BACKEND_DIR
from app.integrations.ai_diagnostics import (
    AI_DIAGNOSTIC_STAGES,
    build_ai_diagnostic,
    diagnostic_from_error,
)
from app.integrations.ai_worker import _write_failure_diagnostic
from app.integrations.schemas import AIWorkerDiagnostic


class AIWorkerDiagnosticTests(unittest.TestCase):
    def test_outer_boundary_writes_safe_envelope_for_every_worker_stage(self):
        unsafe = (
            "GEMINI_API_KEY=secret-value Authorization=Bearer-secret "
            "C:\\private\\recording.webm https://provider.invalid "
            "PROMPT TRANSCRIPT CONFIDENTIAL"
        )
        with tempfile.TemporaryDirectory(
            prefix=".meetmind-ai-diagnostic-",
            dir=BACKEND_DIR,
        ) as temporary:
            run_directory = Path(temporary)
            for index, stage in enumerate(AI_DIAGNOSTIC_STAGES):
                with self.subTest(stage=stage):
                    stage_run = run_directory / str(index)
                    stage_run.mkdir()
                    result_path = stage_run / "result.json"
                    written = _write_failure_diagnostic(
                        result_path,
                        stage_run,
                        RuntimeError(unsafe),
                        stage,
                    )
                    self.assertTrue(written)
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                    diagnostic = AIWorkerDiagnostic.model_validate(
                        payload["integration_diagnostic"]
                    )
                    self.assertEqual(diagnostic.component, "ai")
                    self.assertEqual(diagnostic.stage, stage)
                    self.assertEqual(diagnostic.deletion_state, "not_applicable")
                    serialized = json.dumps(payload)
                    for forbidden in (
                        "secret-value", "Bearer-secret", "private",
                        "recording.webm", "provider.invalid", "PROMPT",
                        "TRANSCRIPT", "CONFIDENTIAL",
                    ):
                        self.assertNotIn(forbidden, serialized)

    def test_attached_provider_fields_are_normalized_and_deletion_is_preserved(self):
        error = RuntimeError("secret transcript")
        error._meetmind_diagnostic = {
            "stage": "ACTIVE polling",
            "status_code": 429,
            "provider_category": "RESOURCE_EXHAUSTED",
            "message": "must never be copied",
        }
        error._meetmind_deletion_state = "failed"

        diagnostic = diagnostic_from_error(error, "upload")

        self.assertEqual(diagnostic["stage"], "active_polling")
        self.assertEqual(diagnostic["status_code"], 429)
        self.assertEqual(diagnostic["category"], "resource_exhausted")
        self.assertEqual(diagnostic["deletion_state"], "failed")
        self.assertNotIn("must never be copied", str(diagnostic))
        AIWorkerDiagnostic.model_validate(diagnostic)

    def test_untrusted_status_and_category_cannot_enter_envelope(self):
        diagnostic = build_ai_diagnostic(
            RuntimeError("secret"),
            "generation",
            status_code="secret-status",
            category="secret-category",
        )
        self.assertIsNone(diagnostic["status_code"])
        self.assertEqual(diagnostic["category"], "RuntimeError")
        self.assertNotIn("secret", str(diagnostic).lower())

    def test_diagnostic_write_failure_does_not_replace_original_error(self):
        with tempfile.TemporaryDirectory(
            prefix=".meetmind-ai-diagnostic-",
            dir=BACKEND_DIR,
        ) as temporary:
            run_directory = Path(temporary)
            original = RuntimeError("original processing failure")
            with patch(
                "app.integrations.ai_worker._write_result",
                side_effect=OSError("diagnostic write failed"),
            ):
                written = _write_failure_diagnostic(
                    run_directory / "result.json",
                    run_directory,
                    original,
                    "result_write",
                )
            self.assertFalse(written)
            self.assertEqual(str(original), "original processing failure")
