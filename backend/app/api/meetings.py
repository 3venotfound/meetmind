import logging
import re
from pathlib import PurePosixPath
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.repositories import (
    DuplicateMeetingNumberError,
    MeetingNotFoundError,
    ProjectNotFoundError,
    RecordingAlreadyExistsError,
    MeetingNotProcessableError,
    Repository,
)
from app.schemas import MeetingCreate, MeetingResponse, RecordingUploadResponse
from app.storage import (
    EmptyRecordingError,
    RecordingDestinationExistsError,
    RecordingStorage,
    RecordingTooLargeError,
    StoredRecording,
)


router = APIRouter(prefix="/meetings", tags=["meetings"])
logger = logging.getLogger(__name__)
ALLOWED_RECORDING_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}


def _repository(request: Request) -> Repository:
    return request.app.state.repository


def _recording_storage(request: Request) -> RecordingStorage:
    return request.app.state.recording_storage


def _safe_client_filename(filename: str | None) -> str:
    normalized = (filename or "").replace("\\", "/")
    basename = PurePosixPath(normalized).name
    return re.sub(r"[\x00-\x1f\x7f]", "", basename).strip()


def _normalized_content_type(content_type: str | None) -> str:
    return (content_type or "").split(";", 1)[0].strip().lower()


def _remove_after_database_failure(
    storage: RecordingStorage,
    recording: StoredRecording,
    meeting_id: str,
) -> bool:
    try:
        storage.remove(recording)
        return True
    except Exception as error:
        logger.error(
            "Recording cleanup failed after database error for meeting %s (%s)",
            meeting_id,
            type(error).__name__,
            exc_info=True,
        )
        return False


@router.post("", response_model=MeetingResponse, status_code=status.HTTP_201_CREATED)
def create_meeting(meeting: MeetingCreate, request: Request) -> dict:
    try:
        return _repository(request).create_meeting(meeting)
    except ProjectNotFoundError as error:
        raise HTTPException(status_code=404, detail="Project not found") from error
    except DuplicateMeetingNumberError as error:
        raise HTTPException(
            status_code=409,
            detail="Meeting number already exists for this project",
        ) from error


@router.get("/{meeting_id}", response_model=MeetingResponse)
def get_meeting(meeting_id: UUID, request: Request) -> dict:
    meeting = _repository(request).get_meeting(str(meeting_id))
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting


@router.post("/{meeting_id}/process", response_model=MeetingResponse)
async def process_meeting(meeting_id: UUID, request: Request) -> dict:
    try:
        return await request.app.state.processing_service.process(meeting_id)
    except MeetingNotFoundError as error:
        raise HTTPException(status_code=404, detail="Meeting not found") from error
    except MeetingNotProcessableError as error:
        raise HTTPException(
            status_code=409,
            detail=f"Meeting cannot be processed while status is {error.status}",
        ) from error
    except Exception as error:
        logger.error("Meeting processing request failed for %s", meeting_id)
        raise HTTPException(status_code=500, detail="Meeting processing failed") from error


@router.post(
    "/{meeting_id}/recording",
    response_model=RecordingUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_recording(
    meeting_id: UUID,
    request: Request,
    file: UploadFile = File(...),
) -> dict:
    meeting_id_text = str(meeting_id)
    repository = _repository(request)
    recording_state = repository.get_recording_state(meeting_id_text)
    if recording_state is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    if (
        recording_state["recording_path"] is not None
        or recording_state["status"] != "created"
    ):
        raise HTTPException(status_code=409, detail="Meeting already has a recording")

    original_filename = _safe_client_filename(file.filename)
    extension = PurePosixPath(original_filename).suffix.lower()
    content_type = _normalized_content_type(file.content_type)
    expected_content_type = ALLOWED_RECORDING_TYPES.get(extension)
    if expected_content_type is None or content_type != expected_content_type:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only MP4 (video/mp4) and WebM (video/webm) recordings are supported",
        )

    storage = _recording_storage(request)
    try:
        try:
            recording = await storage.store(meeting_id_text, extension, file)
        except EmptyRecordingError as error:
            raise HTTPException(status_code=400, detail="Recording file is empty") from error
        except RecordingTooLargeError as error:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="Recording exceeds the maximum upload size",
            ) from error
        except RecordingDestinationExistsError as error:
            raise HTTPException(
                status_code=409,
                detail="Meeting already has a recording",
            ) from error
        except Exception as error:
            logger.error(
                "Recording storage failed for meeting %s (%s)",
                meeting_id_text,
                type(error).__name__,
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail="Recording upload failed",
            ) from error

        try:
            repository.mark_recording_uploaded(
                meeting_id_text,
                recording.relative_path,
            )
        except MeetingNotFoundError as error:
            if not _remove_after_database_failure(storage, recording, meeting_id_text):
                raise HTTPException(status_code=500, detail="Recording upload failed") from error
            raise HTTPException(status_code=404, detail="Meeting not found") from error
        except RecordingAlreadyExistsError as error:
            if not _remove_after_database_failure(storage, recording, meeting_id_text):
                raise HTTPException(status_code=500, detail="Recording upload failed") from error
            raise HTTPException(
                status_code=409,
                detail="Meeting already has a recording",
            ) from error
        except Exception as error:
            cleanup_succeeded = _remove_after_database_failure(
                storage,
                recording,
                meeting_id_text,
            )
            logger.error(
                "Recording database update failed for meeting %s (%s); cleanup=%s",
                meeting_id_text,
                type(error).__name__,
                "complete" if cleanup_succeeded else "failed",
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail="Recording upload failed") from error

        return {
            "meeting_id": meeting_id_text,
            "status": "uploaded",
            "original_filename": original_filename,
            "content_type": content_type,
            "size_bytes": recording.size_bytes,
        }
    finally:
        await file.close()
