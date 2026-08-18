import json
import os
from pathlib import Path
from typing import Any

from app.integrations.path_safety import UnsafePathError


MAX_REQUEST_BYTES = 65_536


def _require_direct_child(path: Path, run_directory: Path) -> Path:
    resolved_run = run_directory.resolve(strict=True)
    resolved_parent = path.parent.resolve(strict=True)
    if resolved_parent != resolved_run:
        raise UnsafePathError
    return path


def write_request_json(path: Path, run_directory: Path, payload: dict[str, Any]) -> None:
    controlled_path = _require_direct_child(path, run_directory)
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError("Worker request is too large")
    with controlled_path.open("xb") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())


def read_result_json(
    path: Path,
    run_directory: Path,
    max_result_bytes: int,
) -> Any:
    controlled_path = _require_direct_child(path, run_directory)
    try:
        stat_result = controlled_path.stat()
    except OSError as error:
        raise ValueError("Worker result is missing") from error
    if not controlled_path.is_file() or stat_result.st_size > max_result_bytes:
        raise ValueError("Worker result is invalid")
    try:
        raw_bytes = controlled_path.read_bytes()
        return json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Worker result is invalid") from error
