from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


MeetingStatus = Literal["created", "uploaded", "processing", "processed", "failed"]
MeetingState = Literal["baseline", "stable", "changed"]
MemoryFieldName = Literal["budget", "deadline", "owner"]
SourceType = Literal["transcript", "visual"]


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ParticipantCreate(RequestModel):
    name: str = Field(min_length=1, max_length=100)
    role: str | None = Field(default=None, min_length=1, max_length=100)


class ParticipantResponse(BaseModel):
    id: int
    name: str
    role: str | None


class ProjectCreate(RequestModel):
    name: str = Field(min_length=1, max_length=200)
    client_org: str = Field(min_length=1, max_length=200)
    target_industry: str | None = Field(default=None, min_length=1, max_length=200)


class ProjectStats(BaseModel):
    meetings_logged: int
    decisions_changed: int
    unresolved_issues: int


class CurrentMemoryValue(BaseModel):
    field_name: MemoryFieldName
    display_value: str
    budget_amount_minor: int | None
    currency_code: str | None
    meeting_id: UUID
    meeting_date: date
    source_type: SourceType
    last_changed_at: date | None


class RecentMeetingResponse(BaseModel):
    id: UUID
    title: str
    meeting_date: date
    meeting_number: int | None
    status: MeetingStatus
    participants: list[str]
    decision_count: int
    state: MeetingState


class ProjectResponse(BaseModel):
    id: UUID
    name: str
    client_org: str
    target_industry: str | None
    created_at: datetime
    updated_at: datetime
    stats: ProjectStats
    recent_meetings: list[RecentMeetingResponse]
    current_memory: list[CurrentMemoryValue]


class MeetingCreate(RequestModel):
    project_id: UUID
    title: str = Field(min_length=1, max_length=200)
    meeting_date: date
    meeting_number: int | None = Field(default=None, ge=1)
    participants: list[ParticipantCreate] = Field(default_factory=list, max_length=50)


class MeetingCounts(BaseModel):
    decisions: int
    action_items: int
    visual_evidence: int
    changes: int
    unresolved_action_items: int


class MeetingResponse(BaseModel):
    id: UUID
    project_id: UUID
    title: str
    meeting_date: date
    meeting_number: int | None
    status: MeetingStatus
    participants: list[ParticipantResponse]
    summary: str | None
    counts: MeetingCounts
    created_at: datetime
    updated_at: datetime


class RecordingUploadResponse(BaseModel):
    meeting_id: UUID
    status: Literal["uploaded"]
    original_filename: str
    content_type: Literal["video/mp4", "video/webm"]
    size_bytes: int
