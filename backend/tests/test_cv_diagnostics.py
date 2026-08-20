import unittest

from pydantic import ValidationError

from app.integrations.cv_diagnostics import (
    CV_DIAGNOSTIC_CATEGORY_LIMIT,
    CV_DIAGNOSTIC_CLASS_LIMIT,
    CV_DIAGNOSTIC_MESSAGE_LIMIT,
    build_cv_diagnostic,
    map_cv_failure_stage,
)
from app.integrations.schemas import CVWorkerDiagnostic


class CVDiagnosticTests(unittest.TestCase):
    def test_diagnostic_redacts_exception_text(self):
        unsafe_text = (
            "GEMINI_API_KEY=secret-value recording.webm "
            "C:\\private\\meeting.webm https://provider.invalid OCR CONFIDENTIAL"
        )
        diagnostic = build_cv_diagnostic(
            RuntimeError(unsafe_text),
            "pipeline_execution",
            "execution_failed",
        )

        serialized = str(diagnostic)
        self.assertEqual(diagnostic["component"], "cv")
        self.assertEqual(diagnostic["stage"], "pipeline_execution")
        self.assertEqual(diagnostic["exception_class"], "RuntimeError")
        self.assertEqual(diagnostic["category"], "execution_failed")
        for forbidden in (
            "secret-value",
            "recording.webm",
            "private",
            "provider.invalid",
            "CONFIDENTIAL",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_diagnostic_fields_are_bounded_and_unknown_values_are_normalized(self):
        long_error_type = type("X" * 500, (Exception,), {})
        diagnostic = build_cv_diagnostic(
            long_error_type("Y" * 10_000),
            "untrusted-stage",
            "untrusted-category",
        )

        self.assertEqual(diagnostic["stage"], "cv_worker_bootstrap")
        self.assertEqual(diagnostic["category"], "execution_failed")
        self.assertLessEqual(
            len(diagnostic["exception_class"]), CV_DIAGNOSTIC_CLASS_LIMIT
        )
        self.assertLessEqual(
            len(diagnostic["category"]), CV_DIAGNOSTIC_CATEGORY_LIMIT
        )
        self.assertLessEqual(
            len(diagnostic["message"]), CV_DIAGNOSTIC_MESSAGE_LIMIT
        )

        with self.assertRaises(ValidationError):
            CVWorkerDiagnostic.model_validate(
                {
                    **diagnostic,
                    "message": "x" * (CV_DIAGNOSTIC_MESSAGE_LIMIT + 1),
                }
            )

    def test_known_video_failures_map_to_safe_stages(self):
        self.assertEqual(
            map_cv_failure_stage(
                "pipeline_execution",
                OSError("Could not open video file: C:\\private\\recording.webm"),
            ),
            "video_open",
        )
        self.assertEqual(
            map_cv_failure_stage(
                "pipeline_execution",
                OSError("Video opened but no frames could be decoded"),
            ),
            "video_decode",
        )
        self.assertEqual(
            map_cv_failure_stage(
                "pipeline_execution",
                ValueError("Video reported an invalid fps"),
            ),
            "video_decode",
        )
        self.assertEqual(
            map_cv_failure_stage("pipeline_execution", RuntimeError("unknown")),
            "pipeline_execution",
        )
