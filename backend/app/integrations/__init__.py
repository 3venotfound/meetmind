"""Safe integration boundaries for teammate-owned AI and CV code."""

from app.integrations.ai_adapter import AIAdapter
from app.integrations.cv_adapter import CVAdapter

__all__ = ["AIAdapter", "CVAdapter"]
