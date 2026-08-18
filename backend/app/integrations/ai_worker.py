"""Child-process entry point. Do not import this module from FastAPI."""

import importlib.util
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any


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
    if not isinstance(payload, dict) or set(payload) != {"recording_path", "storage_root"}:
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


def main(arguments: list[str]) -> int:
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
        if not os.getenv("GEMINI_API_KEY", "").strip():
            raise ValueError

        mimetypes.add_type("video/mp4", ".mp4", strict=True)
        mimetypes.add_type("video/webm", ".webm", strict=True)
        expected_mime = "video/mp4" if recording_path.suffix.lower() == ".mp4" else "video/webm"
        if mimetypes.guess_type(recording_path.name, strict=True)[0] != expected_mime:
            raise ValueError

        module_path = repository_directory / "ai" / "ai_functions.py"
        spec = importlib.util.spec_from_file_location("meetmind_teammate_ai", module_path)
        if spec is None or spec.loader is None:
            raise ModuleNotFoundError
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Phase 3B blocker: teammate code does not poll Gemini file state to ACTIVE
        # and does not delete the remote upload. This worker intentionally does not
        # duplicate or rewrite that logic in Phase 3A.
        result = module.extract_from_audio(str(recording_path))
        _write_result(result_path, run_directory, result)
        return 0
    except ModuleNotFoundError:
        return DEPENDENCY_MISSING_EXIT_CODE
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return INVALID_REQUEST_EXIT_CODE
    except Exception:
        return EXECUTION_FAILED_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
