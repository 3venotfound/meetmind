import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from fastapi import UploadFile

from app.config import BACKEND_DIR


logger = logging.getLogger(__name__)
UPLOAD_CHUNK_SIZE = 1024 * 1024


class EmptyRecordingError(Exception):
    pass


class RecordingTooLargeError(Exception):
    pass


class RecordingDestinationExistsError(Exception):
    pass


class StorageOwnershipError(Exception):
    pass


@dataclass(frozen=True)
class StoredRecording:
    relative_path: str
    size_bytes: int
    physical_path: Path
    file_identity: tuple[int, int]
    created_by_request: bool
    directory_created_by_request: bool


class RecordingStorage:
    def __init__(self, storage_root: Path, max_upload_size_bytes: int):
        self.storage_root = storage_root.resolve()
        try:
            self.storage_root.relative_to(BACKEND_DIR)
        except ValueError as error:
            raise ValueError("STORAGE_ROOT must be inside the backend directory") from error
        self.recordings_root = self.storage_root / "recordings"
        self.max_upload_size_bytes = max_upload_size_bytes

    async def store(self, meeting_id: str, extension: str, upload: UploadFile) -> StoredRecording:
        if extension not in {".mp4", ".webm"}:
            raise ValueError("Unsupported controlled recording extension")
        meeting_directory = self.recordings_root / meeting_id
        final_path = meeting_directory / f"recording{extension}"
        part_path = meeting_directory / f".{uuid4().hex}.part"
        directory_created = False
        part_created = False
        final_created = False
        part_identity: tuple[int, int] | None = None
        total_size = 0

        self.recordings_root.mkdir(parents=True, exist_ok=True)
        try:
            meeting_directory.mkdir()
            directory_created = True
        except FileExistsError:
            if not meeting_directory.is_dir():
                raise

        try:
            with part_path.open("xb") as destination:
                part_created = True
                part_identity = self._file_identity(destination)
                while True:
                    chunk = await upload.read(UPLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    next_size = total_size + len(chunk)
                    if next_size > self.max_upload_size_bytes:
                        raise RecordingTooLargeError
                    self._write_chunk(destination, chunk)
                    total_size = next_size
                if total_size == 0:
                    raise EmptyRecordingError
                destination.flush()
                os.fsync(destination.fileno())

            os.link(part_path, final_path)
            final_created = True
            final_identity = self._path_identity(final_path)
            if final_identity != part_identity:
                raise StorageOwnershipError

            self._unlink_owned(part_path, part_identity)
            part_created = False
            return StoredRecording(
                relative_path=final_path.relative_to(BACKEND_DIR).as_posix(),
                size_bytes=total_size,
                physical_path=final_path,
                file_identity=final_identity,
                created_by_request=True,
                directory_created_by_request=directory_created,
            )
        except FileExistsError as error:
            self._cleanup_owned_path(final_path, part_identity, final_created)
            self._cleanup_owned_path(part_path, part_identity, part_created)
            self._cleanup_owned_directory(meeting_directory, directory_created)
            raise RecordingDestinationExistsError from error
        except Exception:
            self._cleanup_owned_path(final_path, part_identity, final_created)
            self._cleanup_owned_path(part_path, part_identity, part_created)
            self._cleanup_owned_directory(meeting_directory, directory_created)
            raise

    def remove(self, recording: StoredRecording) -> None:
        if not recording.created_by_request:
            return
        self._unlink_owned(recording.physical_path, recording.file_identity)
        self._cleanup_owned_directory(
            recording.physical_path.parent,
            recording.directory_created_by_request,
        )

    @staticmethod
    def _write_chunk(destination: BinaryIO, chunk: bytes) -> None:
        bytes_written = destination.write(chunk)
        if bytes_written != len(chunk):
            raise OSError("Incomplete recording write")

    @staticmethod
    def _file_identity(file_object: BinaryIO) -> tuple[int, int]:
        stat_result = os.fstat(file_object.fileno())
        return stat_result.st_dev, stat_result.st_ino

    @staticmethod
    def _path_identity(path: Path) -> tuple[int, int]:
        stat_result = path.stat()
        return stat_result.st_dev, stat_result.st_ino

    def _unlink_owned(self, path: Path, identity: tuple[int, int] | None) -> None:
        if identity is None or not path.exists():
            return
        if self._path_identity(path) != identity:
            raise StorageOwnershipError
        path.unlink()

    def _cleanup_owned_path(
        self,
        path: Path,
        identity: tuple[int, int] | None,
        created_by_request: bool,
    ) -> None:
        if not created_by_request:
            return
        try:
            self._unlink_owned(path, identity)
        except Exception as error:
            logger.error(
                "Could not clean up request-owned recording file (%s)",
                type(error).__name__,
            )

    @staticmethod
    def _cleanup_owned_directory(path: Path, created_by_request: bool) -> None:
        if not created_by_request:
            return
        try:
            path.rmdir()
        except (FileNotFoundError, OSError):
            pass
