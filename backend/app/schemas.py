from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


MeetingStatus = Literal["created", "uploaded", "processing", "processed", "failed"]
MeetingState = Literal["baseline", "stable", "changed"]
MemoryFieldName = Literal["budget", "deadline", "owner"]
HistoryFieldName = Literal["budget", "deadline", "owner", "decision_text"]
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


class ProjectListItem(BaseModel):
    id: UUID
    name: str
    client_org: str
    target_industry: str | None
    created_at: datetime
    meeting_count: int
    change_count: int
    unresolved_action_count: int


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


class TranscriptSegmentResponse(BaseModel):
    id: int
    speaker: str
    text: str
    start_time_seconds: int | None


class TrackedValueResponse(BaseModel):
    field_name: MemoryFieldName
    raw_value: str
    normalized_value: str | None
    budget_amount_minor: int | None
    currency_code: str | None
    mentioned_by: str | None
    timestamp_seconds: int | None
    source_type: SourceType
    is_canonical: bool
    evidence_id: UUID | None


class DecisionResponse(BaseModel):
    id: int
    text: str
    decided_by: str | None
    timestamp_seconds: int | None
    source_type: SourceType
    evidence_id: UUID | None


class ActionItemResponse(BaseModel):
    id: int
    description: str
    owner: str
    due_date: date | None
    status: Literal["pending", "in_progress", "completed"]


class VisualEvidenceResponse(BaseModel):
    id: UUID
    timestamp_seconds: int
    evidence_type: Literal["whiteboard", "slide", "unknown"]
    text: str
    confidence: float
    image_url: str


class ChangeResponse(BaseModel):
    id: int
    field_name: MemoryFieldName
    old_value: str
    new_value: str
    reason: str | None
    changed_by: str | None
    source_type: SourceType
    timestamp_seconds: int | None
    from_meeting_id: UUID
    to_meeting_id: UUID
    detected_at: datetime


class MeetingResponse(BaseModel):
    id: UUID
    project_id: UUID
    title: str
    meeting_date: date
    meeting_number: int | None
    status: MeetingStatus
    participants: list[ParticipantResponse]
    summary: str | None
    processing_error: str | None = None
    transcript: str | None = None
    transcript_segments: list[TranscriptSegmentResponse] = Field(default_factory=list)
    tracked_values: list[TrackedValueResponse] = Field(default_factory=list)
    decisions: list[DecisionResponse] = Field(default_factory=list)
    action_items: list[ActionItemResponse] = Field(default_factory=list)
    visual_evidence: list[VisualEvidenceResponse] = Field(default_factory=list)
    changes: list[ChangeResponse] = Field(default_factory=list)
    counts: MeetingCounts
    created_at: datetime
    updated_at: datetime


class RecordingUploadResponse(BaseModel):
    meeting_id: UUID
    status: Literal["uploaded"]
    original_filename: str
    content_type: Literal["video/mp4", "video/webm"]
    size_bytes: int


class HistoryEntryResponse(BaseModel):
    field_name: HistoryFieldName
    meeting_id: UUID
    meeting_title: str
    meeting_date: date
    raw_value: str
    normalized_value: str | None
    budget_amount_minor: int | None
    currency_code: str | None
    source_type: SourceType
    speaker: str | None
    timestamp_seconds: int | None
    reason: str | None
    evidence_id: UUID | None
    image_url: str | None
    is_canonical: bool


class ProjectHistoryResponse(BaseModel):
    project_id: UUID
    history: list[HistoryEntryResponse]


class SearchRequest(RequestModel):
    project_id: UUID
    question: str = Field(min_length=1, max_length=500)


class SearchEvidenceResponse(BaseModel):
    meeting_id: UUID
    meeting_title: str
    meeting_date: date
    speaker: str | None
    timestamp_seconds: int | None
    source_type: SourceType
    text: str
    image_url: str | None


class SearchResponse(BaseModel):
    answer: str
    evidence: list[SearchEvidenceResponse]
