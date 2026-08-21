import re
from typing import Any


CV_DIAGNOSTIC_MESSAGE_LIMIT = 240
CV_DIAGNOSTIC_CLASS_LIMIT = 100
CV_DIAGNOSTIC_CATEGORY_LIMIT = 80

CV_DIAGNOSTIC_STAGES = frozenset(
    {
        "cv_worker_bootstrap",
        "pipeline_import",
        "ocr_initialization",
        "video_open",
        "video_decode",
        "pipeline_execution",
        "result_validation",
        "result_write",
    }
)

CV_DIAGNOSTIC_CATEGORIES = frozenset(
    {
        "dependency_missing",
        "execution_failed",
        "invalid_request",
        "invalid_result",
        "io_error",
        "timeout",
        "unsafe_path",
        "worker_exit_nonzero",
        "worker_unavailable",
    }
)

_SAFE_MESSAGES = {
    "cv_worker_bootstrap": "CV worker startup failed.",
    "pipeline_import": "CV pipeline import failed.",
    "ocr_initialization": "OCR initialization failed.",
    "video_open": "Video could not be opened.",
    "video_decode": "Video frames could not be decoded.",
    "pipeline_execution": "CV pipeline execution failed.",
    "result_validation": "CV result validation failed.",
    "result_write": "CV result write failed.",
}


def normalize_cv_stage(stage: str) -> str:
    return stage if stage in CV_DIAGNOSTIC_STAGES else "cv_worker_bootstrap"


def normalize_cv_category(category: str) -> str:
    return category if category in CV_DIAGNOSTIC_CATEGORIES else "execution_failed"


def map_cv_failure_stage(stage: str, error: BaseException) -> str:
    safe_stage = normalize_cv_stage(stage)
    if safe_stage != "pipeline_execution":
        return safe_stage

    # Exception text is inspected only to classify known teammate pipeline
    # errors. It is never copied into the diagnostic envelope or logs.
    unsafe_message = str(error).casefold()
    if "could not open video" in unsafe_message:
        return "video_open"
    if (
        "no frames could be decoded" in unsafe_message
        or "invalid fps" in unsafe_message
    ):
        return "video_decode"
    return safe_stage


def _safe_exception_class(error: BaseException) -> str:
    raw_name = type(error).__name__
    safe_name = re.sub(r"[^A-Za-z0-9_.]", "", raw_name)
    return (safe_name or "Exception")[:CV_DIAGNOSTIC_CLASS_LIMIT]


def build_cv_diagnostic(
    error: BaseException,
    stage: str,
    category: str,
) -> dict[str, Any]:
    safe_stage = map_cv_failure_stage(stage, error)
    safe_category = normalize_cv_category(category)
    return {
        "component": "cv",
        "stage": safe_stage,
        "exception_class": _safe_exception_class(error),
        "category": safe_category[:CV_DIAGNOSTIC_CATEGORY_LIMIT],
        "message": _SAFE_MESSAGES[safe_stage][:CV_DIAGNOSTIC_MESSAGE_LIMIT],
    }
