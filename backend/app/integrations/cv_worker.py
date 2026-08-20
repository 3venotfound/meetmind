"""Child-process entry point. Do not import this module from FastAPI."""

import json
import importlib
import os
import sys
from pathlib import Path
from typing import Any

try:
    from app.integrations.cv_diagnostics import build_cv_diagnostic
except ModuleNotFoundError:  # Direct worker execution uses this directory on sys.path.
    from cv_diagnostics import build_cv_diagnostic


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


def _load_request(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_REQUEST_BYTES:
        raise ValueError
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "recording_path",
        "storage_root",
        "meeting_id",
    }:
        raise ValueError
    return payload


def _write_result(path: Path, run_directory: Path, payload: Any) -> None:
    temporary_path = run_directory / ".result.json.part"
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    with temporary_path.open("xb") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary_path, path)


def _failure_code_and_category(error: BaseException) -> tuple[int, str]:
    if isinstance(error, ModuleNotFoundError):
        return DEPENDENCY_MISSING_EXIT_CODE, "dependency_missing"
    if isinstance(error, OSError):
        return INVALID_REQUEST_EXIT_CODE, "io_error"
    if isinstance(error, (KeyError, TypeError, ValueError, json.JSONDecodeError)):
        return INVALID_REQUEST_EXIT_CODE, "invalid_request"
    return EXECUTION_FAILED_EXIT_CODE, "execution_failed"


def _try_write_diagnostic(
    result_path: Path | None,
    run_directory: Path | None,
    error: BaseException,
    stage: str,
    category: str,
) -> None:
    if result_path is None or run_directory is None:
        return
    try:
        _write_result(
            result_path,
            run_directory,
            {"integration_diagnostic": build_cv_diagnostic(error, stage, category)},
        )
    except Exception:
        # The adapter supplies a fixed bootstrap diagnostic if the worker cannot
        # create its controlled result file. Never emit raw exception text.
        return


def main(arguments: list[str]) -> int:
    stage = "cv_worker_bootstrap"
    run_directory: Path | None = None
    result_path: Path | None = None
    try:
        run_directory, request_path, result_path = _controlled_files(arguments)
        request = _load_request(request_path)
        backend_directory = Path(__file__).resolve().parents[2]
        repository_directory = backend_directory.parent
        storage_root = Path(request["storage_root"]).resolve(strict=True)
        storage_root.relative_to(backend_directory)
        recording_path = Path(request["recording_path"]).resolve(strict=True)
        recording_path.relative_to(storage_root / "recordings")
        if recording_path.name not in {"recording.mp4", "recording.webm"}:
            raise ValueError

        cv_directory = repository_directory / "cv"
        sys.path.insert(0, str(cv_directory))

        stage = "ocr_initialization"
        importlib.import_module("ocr_processor")

        stage = "pipeline_import"
        pipeline = importlib.import_module("pipeline")

        os.chdir(run_directory)
        stage = "pipeline_execution"
        result = pipeline.process_meeting_video(
            str(recording_path),
            str(request["meeting_id"]),
        )
        stage = "result_write"
        _write_result(result_path, run_directory, result)
        return 0
    except Exception as error:
        exit_code, category = _failure_code_and_category(error)
        _try_write_diagnostic(
            result_path,
            run_directory,
            error,
            stage,
            category,
        )
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
