"""Child-process entry point. Do not import this module from FastAPI."""

import importlib.util
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any

try:
    from app.integrations.ai_diagnostics import diagnostic_from_error
except ModuleNotFoundError:  # Direct execution places this directory on sys.path.
    from ai_diagnostics import diagnostic_from_error


DEPENDENCY_MISSING_EXIT_CODE = 20
EXECUTION_FAILED_EXIT_CODE = 21
INVALID_REQUEST_EXIT_CODE = 22
MAX_REQUEST_BYTES = 65_536


def _controlled_files(arguments: list[str]) -> tuple[Path, Path, Path]:
    if len(arguments) != 3:
        raise ValueError
    request_path = Path(arguments[1]).resolve(strict=True)
    run_directory = request_path.parent
    result_path = Path(arguments[2]).resolve(strict=False)
    if (
        request_path.name != "request.json"
        or result_path.name != "result.json"
        or result_path.parent != run_directory
    ):
        raise ValueError
    return run_directory, request_path, result_path


def _read_request(path: Path) -> Any:
    if path.stat().st_size > MAX_REQUEST_BYTES:
        raise ValueError
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("operation"), str):
        raise ValueError
    return payload


def _serialize_result(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _write_encoded(path: Path, run_directory: Path, encoded: bytes) -> None:
    temporary_path = run_directory / ".result.json.part"
    with temporary_path.open("xb") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary_path, path)


def _write_result(path: Path, run_directory: Path, payload: Any) -> None:
    _write_encoded(path, run_directory, _serialize_result(payload))


def _write_failure_diagnostic(
    result_path: Path | None,
    run_directory: Path | None,
    error: BaseException,
    stage: str,
    *,
    module: object | None = None,
) -> bool:
    """Best effort only: diagnostic failure never replaces the original error."""
    if result_path is None or run_directory is None:
        return False
    try:
        resolved_run = run_directory.resolve(strict=True)
        resolved_result = result_path.resolve(strict=False)
        if resolved_result.parent != resolved_run or resolved_result.name != "result.json":
            return False
        diagnostic = diagnostic_from_error(error, stage, module=module)
        _write_result(
            resolved_result,
            resolved_run,
            {"integration_diagnostic": diagnostic},
        )
        return True
    except BaseException:
        return False


def _failure_exit_code(error: BaseException) -> int:
    if isinstance(error, ModuleNotFoundError):
        return DEPENDENCY_MISSING_EXIT_CODE
    if isinstance(
        error,
        (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError),
    ):
        return INVALID_REQUEST_EXIT_CODE
    return EXECUTION_FAILED_EXIT_CODE


def _load_ai_module(repository_directory: Path):
    module_path = repository_directory / "ai" / "ai_functions.py"
    if not module_path.is_file():
        raise ModuleNotFoundError
    spec = importlib.util.spec_from_file_location("meetmind_teammate_ai", module_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_recording_request(
    request: dict[str, Any],
    backend_directory: Path,
) -> tuple[Path, str | None]:
    allowed_keys = {"operation", "recording_path", "storage_root", "visual_context"}
    required_keys = {"operation", "recording_path", "storage_root"}
    if not set(request).issubset(allowed_keys) or not required_keys.issubset(request):
        raise ValueError
    visual_context = request.get("visual_context")
    if visual_context is not None and not isinstance(visual_context, str):
        raise ValueError
    storage_root = Path(request["storage_root"]).resolve(strict=True)
    storage_root.relative_to(backend_directory)
    recording_path = Path(request["recording_path"]).resolve(strict=True)
    recording_path.relative_to(storage_root / "recordings")
    if recording_path.name not in {"recording.mp4", "recording.webm"}:
        raise ValueError
    mimetypes.add_type("video/mp4", ".mp4", strict=True)
    mimetypes.add_type("video/webm", ".webm", strict=True)
    expected_mime = (
        "video/mp4" if recording_path.suffix.lower() == ".mp4" else "video/webm"
    )
    if mimetypes.guess_type(recording_path.name, strict=True)[0] != expected_mime:
        raise ValueError
    return recording_path, visual_context


def main(arguments: list[str]) -> int:
    stage = "worker_bootstrap"
    run_directory = None
    result_path = None
    module = None
    try:
        run_directory, request_path, result_path = _controlled_files(arguments)
        backend_directory = Path(__file__).resolve().parents[2]
        repository_directory = backend_directory.parent

        stage = "request_read"
        raw_request = _read_request(request_path)
        stage = "request_validation"
        request = _validate_request(raw_request)

        stage = "module_discovery"
        module_path = repository_directory / "ai" / "ai_functions.py"
        if not module_path.is_file():
            raise ModuleNotFoundError
        stage = "module_import"
        module = _load_ai_module(repository_directory)

        operation = request["operation"]
        if operation == "extract_recording":
            stage = "client_initialization"
            if not os.getenv("GEMINI_API_KEY", "").strip():
                raise ValueError
            stage = "recording_validation"
            recording_path, visual_context = _validate_recording_request(
                request,
                backend_directory,
            )
            stage = "upload"
            result = module.extract_from_audio(
                str(recording_path),
                visual_context=visual_context,
            )
        elif operation == "extract_text":
            stage = "request_validation"
            if set(request) != {"operation", "text"} or not isinstance(
                request["text"], str
            ):
                raise ValueError
            stage = "client_initialization"
            result = module.extract_meeting_data(request["text"])
        elif operation == "generate_change_reason":
            stage = "request_validation"
            if set(request) != {
                "operation", "field_name", "old_value", "new_value", "source_snippet"
            }:
                raise ValueError
            stage = "client_initialization"
            reason = module.generate_change_reason(
                request["field_name"], request["old_value"], request["new_value"],
                request["source_snippet"],
            )
            result = {"reason": reason}
        elif operation == "search":
            stage = "request_validation"
            if set(request) != {"operation", "question", "records"}:
                raise ValueError
            stage = "client_initialization"
            result = module.search_and_answer(request["question"], request["records"])
        else:
            raise ValueError

        stage = "result_validation"
        if not isinstance(result, dict):
            raise TypeError
        stage = "result_serialization"
        encoded = _serialize_result(result)
        stage = "result_write"
        _write_encoded(result_path, run_directory, encoded)
        return 0
    except BaseException as error:
        _write_failure_diagnostic(
            result_path,
            run_directory,
            error,
            stage,
            module=module,
        )
        return _failure_exit_code(error)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
