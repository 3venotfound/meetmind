"""Safe, bounded diagnostics shared by the AI adapter and worker."""

import re
from typing import Any


AI_DIAGNOSTIC_STAGES = (
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
)
AI_DELETION_STATES = ("succeeded", "failed", "not_applicable")

_LEGACY_STAGE_MAP = {
    "configuration": "client_initialization",
    "ACTIVE polling": "active_polling",
    "deletion": "remote_deletion",
    "worker": "worker_bootstrap",
}
_FIXED_MESSAGES = {
    "worker_bootstrap": "AI worker could not start safely.",
    "request_read": "AI worker request could not be read.",
    "request_validation": "AI worker request was invalid.",
    "module_discovery": "AI module could not be located.",
    "module_import": "AI module could not be loaded.",
    "client_initialization": "AI client could not be initialized.",
    "recording_validation": "AI recording input was invalid.",
    "upload": "AI recording upload failed.",
    "active_polling": "AI recording did not become ready.",
    "generation": "AI generation failed.",
    "result_validation": "AI result was invalid.",
    "result_serialization": "AI result could not be serialized.",
    "result_write": "AI result could not be stored safely.",
    "remote_deletion": "AI remote-file cleanup failed.",
}
_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9_. -]")
_SAFE_STATUS_NAMES = {
    "ABORTED", "CANCELLED", "DEADLINE_EXCEEDED", "FAILED",
    "INTERNAL", "INVALID_ARGUMENT", "NOT_FOUND", "PERMISSION_DENIED",
    "RESOURCE_EXHAUSTED", "SERVICE_UNAVAILABLE", "UNAUTHENTICATED",
    "UNAVAILABLE", "UNKNOWN",
}
_SAFE_CATEGORIES = {
    "dependency_missing", "execution_failed", "invalid_request",
    "invalid_result", "not_configured", "provider_error", "timeout",
    "worker_failed", "worker_unavailable",
    *{value.lower() for value in _SAFE_STATUS_NAMES},
}


def normalize_ai_stage(value: object) -> str:
    candidate = _LEGACY_STAGE_MAP.get(str(value), str(value))
    return candidate if candidate in AI_DIAGNOSTIC_STAGES else "worker_bootstrap"


def normalize_deletion_state(value: object) -> str:
    candidate = str(value)
    return candidate if candidate in AI_DELETION_STATES else "not_applicable"


def _safe_token(value: object, fallback: str, limit: int) -> str:
    cleaned = _SAFE_TOKEN.sub("", str(value or "")).strip()[:limit]
    return cleaned or fallback


def _safe_status(value: object) -> int | str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if -999_999 <= value <= 999_999 else None
    if isinstance(value, str):
        cleaned = _safe_token(value, "", 40).upper()
        return cleaned if cleaned in _SAFE_STATUS_NAMES else None
    return None


def _safe_category(value: object, fallback: str) -> str:
    cleaned = _safe_token(value, "", 80).lower().replace(" ", "_")
    if cleaned in _SAFE_CATEGORIES:
        return cleaned
    fallback_cleaned = _safe_token(fallback, "Exception", 80)
    return fallback_cleaned


def build_ai_diagnostic(
    error: BaseException,
    stage: str,
    *,
    status_code: object = None,
    category: object = None,
    deletion_state: object = "not_applicable",
) -> dict[str, Any]:
    """Build a fixed-message envelope without using exception text."""
    safe_stage = normalize_ai_stage(stage)
    return {
        "component": "ai",
        "stage": safe_stage,
        "exception_class": _safe_token(
            type(error).__name__, "Exception", 100
        ),
        "status_code": _safe_status(status_code),
        "category": _safe_category(category, type(error).__name__),
        "message": _FIXED_MESSAGES[safe_stage],
        "deletion_state": normalize_deletion_state(deletion_state),
    }


def diagnostic_from_error(
    error: BaseException,
    stage: str,
    *,
    module: object | None = None,
) -> dict[str, Any]:
    """Normalize teammate diagnostics, accepting no free-form message text."""
    attached = getattr(error, "_meetmind_diagnostic", None)
    if not isinstance(attached, dict):
        attached = {}
    deletion_state = getattr(error, "_meetmind_deletion_state", None)
    if deletion_state is None and module is not None:
        deletion_state = getattr(module, "_last_remote_deletion_state", None)
    return build_ai_diagnostic(
        error,
        normalize_ai_stage(attached.get("stage", stage)),
        status_code=attached.get("status_code"),
        category=attached.get("provider_category", attached.get("category")),
        deletion_state=deletion_state,
    )
