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
