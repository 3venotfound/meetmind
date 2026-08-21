import os
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import UUID

from app.config import BACKEND_DIR


class UnsafePathError(ValueError):
    pass


def _require_inside(path: Path, parent: Path) -> None:
    try:
        path.relative_to(parent)
    except ValueError as error:
        raise UnsafePathError from error


def validate_recording_path(recording_path: Path, storage_root: Path) -> Path:
    try:
        resolved = recording_path.expanduser().resolve(strict=True)
        recordings_root = (storage_root / "recordings").resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise UnsafePathError from error
    _require_inside(resolved, recordings_root)
    if resolved.name not in {"recording.mp4", "recording.webm"} or not resolved.is_file():
        raise UnsafePathError
    return resolved


def resolve_stored_recording_path(
    relative_path: str,
    meeting_id: UUID,
    storage_root: Path,
) -> Path:
    if "\\" in relative_path:
        raise UnsafePathError
    pure_path = PurePosixPath(relative_path)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise UnsafePathError
    candidate = BACKEND_DIR.joinpath(*pure_path.parts)
    resolved = validate_recording_path(candidate, storage_root)
    expected_directory = (storage_root / "recordings" / str(meeting_id)).resolve()
    _require_inside(resolved, expected_directory)
    return resolved


def validate_evidence_path(raw_path: str, run_directory: Path) -> Path:
    windows_path = PureWindowsPath(raw_path)
    normalized = raw_path.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    if (
        windows_path.is_absolute()
        or bool(windows_path.drive)
        or posix_path.is_absolute()
        or ".." in posix_path.parts
        or not posix_path.parts
        or posix_path.parts[0] != "evidence"
    ):
        raise UnsafePathError
    try:
        run_root = run_directory.resolve(strict=True)
        candidate = run_root.joinpath(*posix_path.parts).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise UnsafePathError from error
    _require_inside(candidate, run_root / "evidence")
    if candidate.suffix.lower() not in {".jpg", ".jpeg"} or not candidate.is_file():
        raise UnsafePathError
    return candidate


class OwnedRunDirectory:
    def __init__(self, integration_root: Path, component: str):
        self.integration_root = integration_root.resolve()
        _require_inside(self.integration_root, BACKEND_DIR)
        self.integration_root.mkdir(parents=True, exist_ok=True)
        created = tempfile.mkdtemp(prefix=f"{component}-", dir=self.integration_root)
        self.path = Path(created).resolve(strict=True)
        _require_inside(self.path, self.integration_root)
        stat_result = os.lstat(self.path)
        self._identity = stat_result.st_dev, stat_result.st_ino
        self._cleaned = False

    def cleanup(self) -> None:
        if self._cleaned:
            return
        try:
            stat_result = os.lstat(self.path)
        except FileNotFoundError:
            self._cleaned = True
            return
        current_identity = stat_result.st_dev, stat_result.st_ino
        if (
            current_identity != self._identity
            or not stat.S_ISDIR(stat_result.st_mode)
            or stat.S_ISLNK(stat_result.st_mode)
        ):
            raise UnsafePathError
        _require_inside(self.path, self.integration_root)
        shutil.rmtree(self.path)
        self._cleaned = True
