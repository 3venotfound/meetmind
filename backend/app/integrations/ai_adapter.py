import logging
from pathlib import Path

from pydantic import SecretStr, ValidationError

from app.integrations.ai_diagnostics import build_ai_diagnostic
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
from app.integrations.schemas import (
    AIExtractionResult,
    AIReasonResult,
    AISearchResult,
    AITextExtractionResult,
    AIWorkerDiagnostic,
)
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
        model: str = "gemini-3-flash-preview",
        file_timeout_seconds: float = 300,
        file_poll_interval_seconds: float = 2,
        python_executable: str = "",
        timeout_seconds: float = 900,
        runner: WorkerRunner | None = None,
        max_result_bytes: int = AI_RESULT_LIMIT_BYTES,
    ):
        self.storage_root = storage_root.resolve()
        self.integration_root = self.storage_root / "integration_runs" / "ai"
        self.api_key = api_key
        self.model = model
        self.file_timeout_seconds = file_timeout_seconds
        self.file_poll_interval_seconds = file_poll_interval_seconds
        self.python_executable = python_executable
        self.timeout_seconds = timeout_seconds
        self.runner = runner or SubprocessRunner()
        self.max_result_bytes = max_result_bytes

    async def extract_recording(
        self,
        recording_path: Path,
        visual_context: str | None = None,
    ) -> AIExtractionResult:
        try:
            safe_recording = validate_recording_path(recording_path, self.storage_root)
        except UnsafePathError as cause:
            raise unsafe_path(
                "ai",
                diagnostic=self._adapter_diagnostic(cause, "recording_validation"),
            ) from None
        payload = {
            "operation": "extract_recording",
            "recording_path": str(safe_recording),
            "storage_root": str(self.storage_root),
        }
        if visual_context:
            payload["visual_context"] = visual_context[:50_000]
        return await self._execute(
            payload,
            AIExtractionResult,
        )

    async def extract_text(self, text: str) -> AITextExtractionResult:
        return await self._execute(
            {"operation": "extract_text", "text": text},
            AITextExtractionResult,
        )

    async def generate_change_reason(
        self,
        field_name: str,
        old_value: str,
        new_value: str,
        source_snippet: str,
    ) -> str:
        result = await self._execute(
            {
                "operation": "generate_change_reason",
                "field_name": field_name,
                "old_value": old_value,
                "new_value": new_value,
                "source_snippet": source_snippet,
            },
            AIReasonResult,
        )
        return result.reason

    async def search(self, question: str, records: list[dict]) -> AISearchResult:
        return await self._execute(
            {"operation": "search", "question": question, "records": records},
            AISearchResult,
        )

    async def _execute(self, payload: dict, result_model):
        is_recording_operation = payload.get("operation") == "extract_recording"
        api_key = self.api_key.get_secret_value().strip() if self.api_key else ""
        if not api_key:
            cause = ValueError()
            raise not_configured(
                "ai",
                diagnostic=self._adapter_diagnostic(
                    cause,
                    "client_initialization",
                    category="not_configured",
                ),
            )
        try:
            executable = resolve_python_executable(self.python_executable)
        except WorkerLaunchError as cause:
            raise unavailable(
                "ai",
                diagnostic=self._adapter_diagnostic(cause, "worker_bootstrap"),
            ) from None

        try:
            run = OwnedRunDirectory(self.integration_root, "ai")
        except (OSError, UnsafePathError) as cause:
            raise execution_failed(
                "ai",
                diagnostic=self._adapter_diagnostic(cause, "worker_bootstrap"),
            ) from None
        request_path = run.path / "request.json"
        result_path = run.path / "result.json"
        try:
            write_request_json(
                request_path,
                run.path,
                payload,
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
                    extra_environment={
                        "GEMINI_API_KEY": api_key,
                        "GEMINI_MODEL": self.model,
                        "GEMINI_FILE_TIMEOUT_SECONDS": str(self.file_timeout_seconds),
                        "GEMINI_FILE_POLL_INTERVAL_SECONDS": str(
                            self.file_poll_interval_seconds
                        ),
                    },
                )
            except WorkerTimeoutError as cause:
                raise timed_out(
                    "ai",
                    diagnostic=self._adapter_diagnostic(cause, "generation"),
                ) from None
            except WorkerLaunchError as cause:
                raise unavailable(
                    "ai",
                    diagnostic=self._adapter_diagnostic(cause, "worker_bootstrap"),
                ) from None
            except Exception as cause:
                raise execution_failed(
                    "ai",
                    diagnostic=self._adapter_diagnostic(cause, "worker_bootstrap"),
                ) from None

            worker_diagnostic = None
            if process_result.returncode == DEPENDENCY_MISSING_EXIT_CODE:
                worker_diagnostic = self._read_worker_diagnostic(result_path, run.path)
                if worker_diagnostic is None:
                    worker_diagnostic = self._adapter_diagnostic(
                        ModuleNotFoundError(),
                        "module_import",
                        category="dependency_missing",
                    )
                self._log_worker_diagnostic(worker_diagnostic)
                raise unavailable("ai", diagnostic=worker_diagnostic)
            if process_result.returncode != 0:
                worker_diagnostic = self._read_worker_diagnostic(result_path, run.path)
                if worker_diagnostic is None:
                    worker_diagnostic = self._adapter_diagnostic(
                        RuntimeError(),
                        "worker_bootstrap",
                        category="worker_failed",
                    )
                self._log_worker_diagnostic(worker_diagnostic)
                raise execution_failed("ai", diagnostic=worker_diagnostic)
            try:
                result_payload = read_result_json(
                    result_path,
                    run.path,
                    self.max_result_bytes,
                )
                return result_model.model_validate(result_payload)
            except (ValueError, ValidationError, UnsafePathError) as cause:
                raise invalid_result(
                    "ai",
                    diagnostic=self._adapter_diagnostic(
                        cause,
                        "result_validation",
                        deletion_state=(
                            "succeeded" if is_recording_operation else "not_applicable"
                        ),
                    ),
                ) from None
        except IntegrationError as error:
            logger.warning(
                "AI integration ended with code=%s retryable=%s",
                error.code.value,
                error.retryable,
            )
            raise
        except (OSError, ValueError, UnsafePathError) as cause:
            error = execution_failed(
                "ai",
                diagnostic=self._adapter_diagnostic(cause, "worker_bootstrap"),
            )
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

    def _adapter_diagnostic(
        self,
        error: BaseException,
        stage: str,
        *,
        category: object = None,
        deletion_state: str = "not_applicable",
    ) -> AIWorkerDiagnostic:
        return AIWorkerDiagnostic.model_validate(
            build_ai_diagnostic(
                error,
                stage,
                category=category,
                deletion_state=deletion_state,
            )
        )

    def _read_worker_diagnostic(
        self,
        result_path: Path,
        run_directory: Path,
    ) -> AIWorkerDiagnostic | None:
        try:
            payload = read_result_json(
                result_path,
                run_directory,
                min(self.max_result_bytes, 4096),
            )
            diagnostic = AIWorkerDiagnostic.model_validate(
                payload.get("integration_diagnostic")
            )
        except (AttributeError, OSError, ValueError, ValidationError, UnsafePathError):
            return None
        return diagnostic

    @staticmethod
    def _log_worker_diagnostic(diagnostic: AIWorkerDiagnostic) -> None:
        logger.warning(
            "AI worker diagnostic component=%s stage=%s exception=%s status=%s "
            "category=%s deletion=%s message=%s",
            diagnostic.component,
            diagnostic.stage,
            diagnostic.exception_class,
            diagnostic.status_code,
            diagnostic.category,
            diagnostic.deletion_state,
            diagnostic.message,
        )
