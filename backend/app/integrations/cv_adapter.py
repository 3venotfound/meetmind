import logging
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError

from app.config import BACKEND_DIR
from app.integrations.errors import (
    IntegrationError,
    execution_failed,
    invalid_result,
    timed_out,
    unavailable,
    unsafe_path,
)
from app.integrations.io_utils import read_result_json, write_request_json
from app.integrations.path_safety import (
    OwnedRunDirectory,
    UnsafePathError,
    validate_evidence_path,
    validate_recording_path,
)
from app.integrations.schemas import (
    CVProcessingResult,
    CVVisualEvidence,
    RawCVPipelineResult,
)
from app.integrations.subprocess_runner import (
    SubprocessRunner,
    WorkerLaunchError,
    WorkerRunner,
    WorkerTimeoutError,
    resolve_python_executable,
)


logger = logging.getLogger(__name__)
CV_WORKER_PATH = Path(__file__).with_name("cv_worker.py")
CV_RESULT_LIMIT_BYTES = 8 * 1024 * 1024
DEPENDENCY_MISSING_EXIT_CODE = 20


def normalize_timestamp(timestamp: str) -> int:
    minutes_text, seconds_text = timestamp.split(":", 1)
    return int(minutes_text) * 60 + int(seconds_text)


class CVAdapter:
    def __init__(
        self,
        *,
        storage_root: Path,
        python_executable: str = "",
        timeout_seconds: float = 1800,
        runner: WorkerRunner | None = None,
        max_result_bytes: int = CV_RESULT_LIMIT_BYTES,
    ):
        self.storage_root = storage_root.resolve()
        self.integration_root = self.storage_root / "integration_runs" / "cv"
        self.python_executable = python_executable
        self.timeout_seconds = timeout_seconds
        self.runner = runner or SubprocessRunner()
        self.max_result_bytes = max_result_bytes

    async def process_recording(
        self,
        recording_path: Path,
        meeting_id: UUID,
    ) -> CVProcessingResult:
        try:
            executable = resolve_python_executable(self.python_executable)
            safe_recording = validate_recording_path(recording_path, self.storage_root)
        except WorkerLaunchError:
            raise unavailable("cv") from None
        except UnsafePathError:
            raise unsafe_path("cv") from None

        try:
            run = OwnedRunDirectory(self.integration_root, "cv")
        except (OSError, UnsafePathError):
            raise execution_failed("cv") from None
        keep_run = False
        request_path = run.path / "request.json"
        result_path = run.path / "result.json"
        try:
            write_request_json(
                request_path,
                run.path,
                {
                    "recording_path": str(safe_recording),
                    "storage_root": str(self.storage_root),
                    "meeting_id": str(meeting_id),
                },
            )
            command = [
                str(executable),
                str(CV_WORKER_PATH),
                str(request_path),
                str(result_path),
            ]
            try:
                process_result = await self.runner.run(
                    command,
                    cwd=run.path,
                    timeout_seconds=self.timeout_seconds,
                )
            except WorkerTimeoutError:
                raise timed_out("cv") from None
            except WorkerLaunchError:
                raise unavailable("cv") from None
            except Exception:
                raise execution_failed("cv") from None

            if process_result.returncode == DEPENDENCY_MISSING_EXIT_CODE:
                raise unavailable("cv")
            if process_result.returncode != 0:
                raise execution_failed("cv")
            try:
                payload = read_result_json(
                    result_path,
                    run.path,
                    self.max_result_bytes,
                )
                raw_result = RawCVPipelineResult.model_validate(payload)
                if raw_result.meeting_id != str(meeting_id):
                    raise ValueError("Meeting ID mismatch")
                evidence = []
                for raw_evidence in raw_result.visual_evidence:
                    evidence_path = validate_evidence_path(
                        raw_evidence.image_path,
                        run.path,
                    )
                    evidence.append(
                        CVVisualEvidence(
                            timestamp_seconds=normalize_timestamp(raw_evidence.timestamp),
                            evidence_type=raw_evidence.type,
                            raw_ocr_text=raw_evidence.text,
                            confidence=raw_evidence.confidence,
                            image_path=evidence_path.relative_to(BACKEND_DIR).as_posix(),
                        )
                    )
                result = CVProcessingResult(
                    meeting_id=meeting_id,
                    visual_evidence=evidence,
                )
            except UnsafePathError:
                raise unsafe_path("cv") from None
            except (ValueError, ValidationError):
                raise invalid_result("cv") from None

            if evidence:
                result._owned_run_directory = run
                keep_run = True
            return result
        except IntegrationError as error:
            logger.warning(
                "CV integration ended with code=%s retryable=%s",
                error.code.value,
                error.retryable,
            )
            raise
        except (OSError, ValueError, UnsafePathError):
            error = execution_failed("cv")
            logger.warning(
                "CV integration ended with code=%s retryable=%s",
                error.code.value,
                error.retryable,
            )
            raise error from None
        finally:
            if not keep_run:
                try:
                    run.cleanup()
                except (OSError, UnsafePathError):
                    logger.error("CV run-directory cleanup was refused or failed")

    def cleanup_validated_run(self, result: CVProcessingResult) -> None:
        """Phase 3B must copy evidence to permanent storage before calling this."""
        owned_run = result._owned_run_directory
        if owned_run is None:
            return
        if not isinstance(owned_run, OwnedRunDirectory):
            raise UnsafePathError
        try:
            owned_run.cleanup()
        except (OSError, UnsafePathError):
            raise unsafe_path("cv") from None
        result._owned_run_directory = None
