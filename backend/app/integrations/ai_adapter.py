import logging
from pathlib import Path

from pydantic import SecretStr, ValidationError

from app.integrations.errors import (
    IntegrationError,
    execution_failed,
    invalid_result,
    not_configured,
    timed_out,
    unavailable,
    unsafe_path,
)
from app.integrations.io_utils import read_result_json, write_request_json
from app.integrations.path_safety import (
    OwnedRunDirectory,
    UnsafePathError,
    validate_recording_path,
)
from app.integrations.schemas import AIExtractionResult
from app.integrations.subprocess_runner import (
    SubprocessRunner,
    WorkerLaunchError,
    WorkerRunner,
    WorkerTimeoutError,
    resolve_python_executable,
)


logger = logging.getLogger(__name__)
AI_WORKER_PATH = Path(__file__).with_name("ai_worker.py")
AI_RESULT_LIMIT_BYTES = 16 * 1024 * 1024
DEPENDENCY_MISSING_EXIT_CODE = 20


class AIAdapter:
    def __init__(
        self,
        *,
        storage_root: Path,
        api_key: SecretStr | None,
        python_executable: str = "",
        timeout_seconds: float = 900,
        runner: WorkerRunner | None = None,
        max_result_bytes: int = AI_RESULT_LIMIT_BYTES,
    ):
        self.storage_root = storage_root.resolve()
        self.integration_root = self.storage_root / "integration_runs" / "ai"
        self.api_key = api_key
        self.python_executable = python_executable
        self.timeout_seconds = timeout_seconds
        self.runner = runner or SubprocessRunner()
        self.max_result_bytes = max_result_bytes

    async def extract_recording(self, recording_path: Path) -> AIExtractionResult:
        api_key = self.api_key.get_secret_value().strip() if self.api_key else ""
        if not api_key:
            raise not_configured("ai")
        try:
            executable = resolve_python_executable(self.python_executable)
            safe_recording = validate_recording_path(recording_path, self.storage_root)
        except WorkerLaunchError:
            raise unavailable("ai") from None
        except UnsafePathError:
            raise unsafe_path("ai") from None

        try:
            run = OwnedRunDirectory(self.integration_root, "ai")
        except (OSError, UnsafePathError):
            raise execution_failed("ai") from None
        request_path = run.path / "request.json"
        result_path = run.path / "result.json"
        try:
            write_request_json(
                request_path,
                run.path,
                {
                    "recording_path": str(safe_recording),
                    "storage_root": str(self.storage_root),
                },
            )
            command = [
                str(executable),
                str(AI_WORKER_PATH),
                str(request_path),
                str(result_path),
            ]
            try:
                process_result = await self.runner.run(
                    command,
                    cwd=run.path,
                    timeout_seconds=self.timeout_seconds,
                    extra_environment={"GEMINI_API_KEY": api_key},
                )
            except WorkerTimeoutError:
                raise timed_out("ai") from None
            except WorkerLaunchError:
                raise unavailable("ai") from None
            except Exception:
                raise execution_failed("ai") from None

            if process_result.returncode == DEPENDENCY_MISSING_EXIT_CODE:
                raise unavailable("ai")
            if process_result.returncode != 0:
                raise execution_failed("ai")
            try:
                payload = read_result_json(
                    result_path,
                    run.path,
                    self.max_result_bytes,
                )
                return AIExtractionResult.model_validate(payload)
            except (ValueError, ValidationError, UnsafePathError):
                raise invalid_result("ai") from None
        except IntegrationError as error:
            logger.warning(
                "AI integration ended with code=%s retryable=%s",
                error.code.value,
                error.retryable,
            )
            raise
        except (OSError, ValueError, UnsafePathError):
            error = execution_failed("ai")
            logger.warning(
                "AI integration ended with code=%s retryable=%s",
                error.code.value,
                error.retryable,
            )
            raise error from None
        finally:
            try:
                run.cleanup()
            except (OSError, UnsafePathError):
                logger.error("AI run-directory cleanup was refused or failed")
