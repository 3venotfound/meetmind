from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class StrictIntegrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AIDecision(StrictIntegrationModel):
    text: str = Field(min_length=1)
    decided_by: str = Field(min_length=1)
    timestamp_seconds: int | None = Field(default=None, ge=0)


class AIActionItem(StrictIntegrationModel):
    description: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    due_date: str | None = None


class AIExtractedValue(StrictIntegrationModel):
    value: str | None = None
    mentioned_by: str | None = None
    timestamp_seconds: int | None = Field(default=None, ge=0)


class AIBudget(AIExtractedValue):
    pass


class AIDeadline(AIExtractedValue):
    pass


class AIOwner(AIExtractedValue):
    pass


class AIExtractionResult(StrictIntegrationModel):
    transcript: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    decisions: list[AIDecision]
    action_items: list[AIActionItem]
    budget: AIBudget
    deadline: AIDeadline
    owner: AIOwner
    visual_extraction: "AITextExtractionResult | None" = None


class AITextExtractionResult(StrictIntegrationModel):
    summary: str
    decisions: list[AIDecision]
    action_items: list[AIActionItem]
    budget: AIBudget
    deadline: AIDeadline
    owner: AIOwner


AIExtractionResult.model_rebuild()


class AIReasonResult(StrictIntegrationModel):
    reason: str = Field(min_length=1)


class AISearchEvidence(StrictIntegrationModel):
    meeting_id: str
    speaker: str | None
    timestamp_seconds: int | None = Field(default=None, ge=0)
    source_type: Literal["transcript", "visual"]


class AISearchResult(StrictIntegrationModel):
    answer: str = Field(min_length=1)
    evidence: list[AISearchEvidence]


class AIWorkerDiagnostic(StrictIntegrationModel):
    component: Literal["ai"]
    stage: Literal[
        "worker_bootstrap",
        "request_read",
        "request_validation",
        "module_discovery",
        "module_import",
        "client_initialization",
        "recording_validation",
        "upload",
        "active_polling",
        "generation",
        "result_validation",
        "result_serialization",
        "result_write",
        "remote_deletion",
    ]
    exception_class: str = Field(min_length=1, max_length=100)
    status_code: int | str | None = None
    category: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=240)
    deletion_state: Literal["succeeded", "failed", "not_applicable"]


class CVWorkerDiagnostic(StrictIntegrationModel):
    component: Literal["cv"]
    stage: Literal[
        "cv_worker_bootstrap",
        "pipeline_import",
        "ocr_initialization",
        "video_open",
        "video_decode",
        "pipeline_execution",
        "result_validation",
        "result_write",
    ]
    exception_class: str = Field(min_length=1, max_length=100)
    category: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=240)


class RawCVVisualEvidence(StrictIntegrationModel):
    timestamp: str = Field(pattern=r"^\d{2,}:[0-5]\d$")
    type: Literal["whiteboard", "slide", "unknown"]
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    image_path: str = Field(min_length=1)


class RawCVPipelineResult(StrictIntegrationModel):
    meeting_id: str = Field(min_length=1)
    visual_evidence: list[RawCVVisualEvidence]


class CVVisualEvidence(StrictIntegrationModel):
    timestamp_seconds: int = Field(ge=0)
    evidence_type: Literal["whiteboard", "slide", "unknown"]
    raw_ocr_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    image_path: str = Field(min_length=1)


class CVProcessingResult(StrictIntegrationModel):
    meeting_id: UUID
    visual_evidence: list[CVVisualEvidence]
    _owned_run_directory: object | None = PrivateAttr(default=None)
