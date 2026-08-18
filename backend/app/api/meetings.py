from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status

from app.repositories import (
    DuplicateMeetingNumberError,
    ProjectNotFoundError,
    Repository,
)
from app.schemas import MeetingCreate, MeetingResponse


router = APIRouter(prefix="/meetings", tags=["meetings"])


def _repository(request: Request) -> Repository:
    return request.app.state.repository


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
