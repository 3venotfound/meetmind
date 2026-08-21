import os
import re
import json
import mimetypes
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
DEFAULT_FILE_TIMEOUT_SECONDS = 300.0
DEFAULT_FILE_POLL_INTERVAL_SECONDS = 2.0
SUPPORTED_RECORDING_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}
DIAGNOSTIC_MESSAGE_LIMIT = 240
_last_remote_deletion_state = "not_applicable"
_DIAGNOSTIC_MESSAGES = {
    "recording_validation": "AI recording input was invalid.",
    "configuration": "AI client could not be initialized.",
    "upload": "AI recording upload failed.",
    "ACTIVE polling": "AI recording did not become ready.",
    "generation": "AI generation failed.",
    "deletion": "AI remote-file cleanup failed.",
}
_SAFE_PROVIDER_CATEGORIES = {
    "ABORTED", "CANCELLED", "DEADLINE_EXCEEDED", "FAILED", "INTERNAL",
    "INVALID_ARGUMENT", "NOT_FOUND", "PERMISSION_DENIED",
    "RESOURCE_EXHAUSTED", "SERVICE_UNAVAILABLE", "UNAUTHENTICATED",
    "UNAVAILABLE", "UNKNOWN",
}


def _safe_diagnostic(error: BaseException, stage: str) -> dict:
    """Create a fixed-message diagnostic without reading exception text."""
    status_code = getattr(error, "code", None)
    response = getattr(error, "response", None)
    if not isinstance(status_code, (int, str)):
        status_code = getattr(response, "status_code", None)
    if isinstance(status_code, str):
        normalized_status = status_code.upper()
        status_code = (
            normalized_status
            if normalized_status in _SAFE_PROVIDER_CATEGORIES
            else None
        )
    elif not isinstance(status_code, int) or isinstance(status_code, bool):
        status_code = None
    raw_category = (
        getattr(error, "status", None)
        or getattr(error, "reason", None)
        or type(error).__name__
    )
    normalized_category = str(raw_category).upper()
    category = (
        normalized_category
        if normalized_category in _SAFE_PROVIDER_CATEGORIES
        else type(error).__name__
    )
    return {
        "stage": stage,
        "exception_class": type(error).__name__[:100],
        "status_code": status_code,
        "provider_category": category[:80],
        "message": _DIAGNOSTIC_MESSAGES.get(
            stage,
            "AI processing failed.",
        )[:DIAGNOSTIC_MESSAGE_LIMIT],
    }


def _positive_float_setting(name: str, default: float) -> float:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    value = float(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


def _get_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured")
    return genai.Client(api_key=api_key)


def _file_state_name(file_object) -> str:
    state = getattr(file_object, "state", None)
    if state is None:
        return ""
    name = getattr(state, "name", None)
    if isinstance(name, str):
        return name.upper()
    return str(state).rsplit(".", 1)[-1].upper()

_FIELDS = {
    "summary": {"type": "STRING"},
    "decisions": {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "text": {"type": "STRING"},
                "decided_by": {"type": "STRING"},
                "timestamp_seconds": {"type": "INTEGER"},
            },
            "required": ["text", "decided_by"],
        },
    },
    "action_items": {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "description": {"type": "STRING"},
                "owner": {"type": "STRING"},
                "due_date": {"type": "STRING", "nullable": True},
            },
            "required": ["description", "owner"],
        },
    },
    "budget": {
        "type": "OBJECT",
        "properties": {
            "value": {"type": "STRING", "nullable": True},
            "mentioned_by": {"type": "STRING", "nullable": True},
            "timestamp_seconds": {"type": "INTEGER"},
        },
    },
    "deadline": {
        "type": "OBJECT",
        "properties": {
            "value": {"type": "STRING", "nullable": True},
            "mentioned_by": {"type": "STRING", "nullable": True},
            "timestamp_seconds": {"type": "INTEGER"},
        },
    },
    "owner": {
        "type": "OBJECT",
        "properties": {
            "value": {"type": "STRING", "nullable": True},
            "mentioned_by": {"type": "STRING", "nullable": True},
            "timestamp_seconds": {"type": "INTEGER"},
        },
    },
}

TEXT_SCHEMA = {
    "type": "OBJECT",
    "properties": dict(_FIELDS),
    "required": ["summary", "decisions", "action_items", "budget", "deadline", "owner"],
}

# Audio-input schema: model has to produce the transcript itself, so it's included + required
AUDIO_SCHEMA = {
    "type": "OBJECT",
    "properties": {"transcript": {"type": "STRING"}, **_FIELDS},
    "required": ["transcript", "summary", "decisions", "action_items", "budget", "deadline", "owner"],
}

AUDIO_WITH_VISUAL_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "transcript": {"type": "STRING"},
        **_FIELDS,
        "visual_extraction": TEXT_SCHEMA,
    },
    "required": [
        "transcript", "summary", "decisions", "action_items",
        "budget", "deadline", "owner", "visual_extraction",
    ],
}

_EXTRACTION_RULES = """
Rules:
- "decided_by" = the person who made or proposed the decision, not who merely confirmed/approved it.
- "budget" value = the actual budget figure mentioned (not the delta/increase amount alone). If only a cost increase is mentioned without a base figure, state that in the summary and put the delta in "value" with a note that it's a delta, not a base figure.
- Every "decided_by" and "mentioned_by" MUST be a speaker name that literally appears in the transcript — never invent, generalize ("the team"), or guess a name.
- An action-item "owner" may be a named assignee who did not speak. Preserve the explicitly stated spelling exactly, without inventing or replacing a name, and return it as a non-empty trimmed string.
- If a field isn't mentioned at all, use null for its value — do not guess a plausible-sounding number or date.
- decisions and action_items can be empty arrays if none exist.
"""

AUDIO_PROMPT = f"""
Listen to this meeting audio and do two things:

1. Produce a plain-text transcript with speaker labels, in the format "Name: text",
   one line per utterance. Use the speaker names as they introduce themselves or are
   addressed by others in the audio. If a speaker's name is genuinely never stated,
   label them Speaker1, Speaker2, etc. consistently.

2. Extract structured meeting data from that same audio.
{_EXTRACTION_RULES}
- timestamp_seconds should be your best estimate of when in the audio that fact was stated.
"""


def _get_speakers(transcript_text: str) -> set[str]:
    """Pulls speaker names from 'Name:' style lines so we can catch the
    model inventing an attribution that isn't actually in the transcript."""
    return set(re.findall(r"^([A-Za-z][\w .]{0,30}):", transcript_text, flags=re.MULTILINE))


def _validate(data: dict, speakers: set[str]) -> list[str]:
    """Returns a list of problems found — empty list means clean."""
    issues = []

    def check(name, field_label):
        if name is not None and name not in speakers:
            issues.append(f"{field_label} is not a speaker in this transcript")

    for d in data.get("decisions", []):
        check(d.get("decided_by"), "decision.decided_by")
    for a in data.get("action_items", []):
        if "owner" not in a or a["owner"] is None:
            continue
        action_owner = a["owner"]
        if (
            not isinstance(action_owner, str)
            or not action_owner.strip()
            or action_owner != action_owner.strip()
        ):
            issues.append("action_item.owner must be a non-empty trimmed string")
    for field in ("budget", "deadline", "owner"):
        check(data.get(field, {}).get("mentioned_by"), f"{field}.mentioned_by")

    return issues


def _generate_json(contents, schema, speakers_source: str, max_retries: int, client=None) -> dict:
    """Shared call-and-validate loop used by both entry points below."""
    active_client = client or _get_client()
    last_error = None
    for attempt in range(max_retries + 1):
        response = active_client.models.generate_content(
            model=_gemini_model(),
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.1,  # low = consistent extraction across meetings, matters for diffing
            ),
        )

        try:
            data = json.loads(response.text)
        except json.JSONDecodeError as e:
            last_error = f"invalid JSON: {e}"
            continue

        # audio path validates against the transcript Gemini just produced;
        # text path validates against the transcript we were given
        speakers = _get_speakers(data.get("transcript", "") if speakers_source == "audio" else speakers_source)
        issues = _validate(data, speakers)
        if issues:
            last_error = "; ".join(issues)
            continue

        return data

    raise ValueError(f"Extraction failed after {max_retries + 1} attempt(s): {last_error}")


def extract_meeting_data(transcript_text: str, max_retries: int = 1) -> dict:
    """
    Takes raw transcript text (e.g. from a separate STT step, or from Groq
    Whisper as backup) and returns structured JSON matching the MeetMind schema.
    """
    prompt = f"""
You are extracting structured meeting data from this transcript:

{transcript_text}
{_EXTRACTION_RULES}
"""
    return _generate_json(
        prompt,
        TEXT_SCHEMA,
        speakers_source=transcript_text,
        max_retries=max_retries,
    )


def extract_from_audio(
    audio_path: str,
    max_retries: int = 1,
    file_timeout_seconds: float | None = None,
    poll_interval_seconds: float | None = None,
    visual_context: str | None = None,
) -> dict:
    """
    Uploads an audio file and returns structured JSON (transcript + extraction)
    in a single Gemini call — Gemini does STT and extraction together.
    """
    global _last_remote_deletion_state
    _last_remote_deletion_state = "not_applicable"
    stage = "recording_validation"
    active_client = None
    uploaded_file = None
    processing_error = None
    try:
        recording_path = Path(audio_path).expanduser().resolve(strict=True)
        mime_type = SUPPORTED_RECORDING_MIME_TYPES.get(recording_path.suffix.lower())
        if mime_type is None:
            raise ValueError("Recording must be MP4 or WebM")
        mimetypes.add_type(mime_type, recording_path.suffix.lower(), strict=True)
        timeout_seconds = (
            file_timeout_seconds
            if file_timeout_seconds is not None
            else _positive_float_setting(
                "GEMINI_FILE_TIMEOUT_SECONDS",
                DEFAULT_FILE_TIMEOUT_SECONDS,
            )
        )
        interval_seconds = (
            poll_interval_seconds
            if poll_interval_seconds is not None
            else _positive_float_setting(
                "GEMINI_FILE_POLL_INTERVAL_SECONDS",
                DEFAULT_FILE_POLL_INTERVAL_SECONDS,
            )
        )
        if timeout_seconds <= 0 or interval_seconds <= 0:
            raise ValueError("Gemini file timing settings must be positive")

        stage = "configuration"
        active_client = _get_client()
        stage = "upload"
        uploaded_file = active_client.files.upload(
            file=str(recording_path),
            config=types.UploadFileConfig(mime_type=mime_type),
        )
        deadline = time.monotonic() + timeout_seconds
        stage = "ACTIVE polling"
        while True:
            state_name = _file_state_name(uploaded_file)
            if state_name == "ACTIVE":
                break
            if state_name == "FAILED":
                raise RuntimeError("Gemini file processing failed")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Gemini file processing timed out")
            time.sleep(min(interval_seconds, remaining))
            uploaded_file = active_client.files.get(name=uploaded_file.name)

        stage = "generation"
        prompt = AUDIO_PROMPT
        schema = AUDIO_SCHEMA
        if visual_context:
            prompt += f"""

The CV pipeline separately extracted this OCR text from slides or whiteboards:
{visual_context[:50_000]}

Also return `visual_extraction`, based ONLY on that OCR text. It must contain
summary, decisions, action_items, budget, deadline, and owner. Keep visual
decisions and action_items empty when the OCR does not identify a responsible
speaker. Set mentioned_by to null because OCR does not establish who spoke.
Do not mix visual-only facts into the spoken budget/deadline/owner fields.
"""
            schema = AUDIO_WITH_VISUAL_SCHEMA
        return _generate_json(
            [prompt, uploaded_file],
            schema,
            speakers_source="audio",
            max_retries=max_retries,
            client=active_client,
        )
    except BaseException as error:
        processing_error = error
        try:
            error._meetmind_deletion_state = "not_applicable"
        except Exception:
            pass
        try:
            error._meetmind_diagnostic = _safe_diagnostic(error, stage)
        except Exception:
            pass
        raise
    finally:
        remote_name = getattr(uploaded_file, "name", None)
        if remote_name and active_client is not None:
            try:
                stage = "deletion"
                active_client.files.delete(name=remote_name)
                _last_remote_deletion_state = "succeeded"
                if processing_error is not None:
                    try:
                        processing_error._meetmind_deletion_state = "succeeded"
                    except Exception:
                        pass
            except Exception as deletion_error:
                _last_remote_deletion_state = "failed"
                if processing_error is not None:
                    try:
                        processing_error._meetmind_deletion_state = "failed"
                    except Exception:
                        pass
                if processing_error is None:
                    try:
                        deletion_error._meetmind_diagnostic = _safe_diagnostic(
                            deletion_error,
                            stage,
                        )
                        deletion_error._meetmind_deletion_state = "failed"
                    except Exception:
                        pass
                    raise

def generate_change_reason(field_name: str, old_value: str, new_value: str, transcript_snippet: str) -> str:
    """
    Takes a detected change (old vs new value) + the relevant transcript snippet,
    returns a short one-sentence reason.

    NOTE: whether something "changed" is decided by backend's rule-based diff,
    NOT by this function — this only explains a change already confirmed.
    """
    prompt = f"""
A meeting tracking system detected that the "{field_name}" field changed.

Old value: {old_value}
New value: {new_value}

Relevant transcript segment:
{transcript_snippet}

Task: Write ONE short sentence (max 20 words) explaining WHY this change happened, based only on the transcript segment above. Do not add information not present in the transcript. Return only the sentence, no extra text.
"""

    response = _get_client().models.generate_content(
        model=_gemini_model(),
        contents=prompt
    )

    return response.text.strip()

def search_and_answer(question: str, records: list[dict]) -> dict:
    """
    records: list of dicts, each like:
        {
            "meeting_id": 2,
            "speaker": "Sarah",
            "timestamp_seconds": 45,
            "text": "The vendor requested five additional days...",
            "source_type": "transcript"  # or "visual"
        }

    Returns: {"answer": str, "evidence": [{"meeting_id", "speaker", "timestamp_seconds", "source_type"}]}
    """

    # Format records as numbered context for the prompt
    context_lines = []
    for i, r in enumerate(records):
        context_lines.append(
            f"[{i}] Meeting {r['meeting_id']}, {r['speaker']}, "
            f"{r['timestamp_seconds']}s, source: {r['source_type']}: \"{r['text']}\""
        )
    context_text = "\n".join(context_lines)

    prompt = f"""
You are answering a question using ONLY the meeting records provided below. Do not use outside knowledge or guess.

Records:
{context_text}

Question: {question}

Instructions:
- Answer in 1-3 sentences, based only on the records above.
- If the records don't contain enough information to answer, say so plainly.
- Then list which record numbers (the [N] labels) support your answer.
"""

    schema = {
        "type": "OBJECT",
        "properties": {
            "answer": {"type": "STRING"},
            "supporting_record_indices": {
                "type": "ARRAY",
                "items": {"type": "INTEGER"}
            }
        },
        "required": ["answer", "supporting_record_indices"]
    }

    response = _get_client().models.generate_content(
        model=_gemini_model(),
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.1,
        ),
    )

    data = json.loads(response.text)

    # Map the record indices back to actual evidence objects (meeting/speaker/timestamp/source)
    evidence = []
    for idx in data.get("supporting_record_indices", []):
        if 0 <= idx < len(records):
            r = records[idx]
            evidence.append({
                "meeting_id": r["meeting_id"],
                "speaker": r["speaker"],
                "timestamp_seconds": r["timestamp_seconds"],
                "source_type": r["source_type"]
            })

    return {"answer": data["answer"], "evidence": evidence}


if __name__ == "__main__":
    dummy_transcript = """
Sarah: The vendor requested five additional days, so I propose moving the launch from 10 August to 15 August.
Sarah: That gives Ahmad and the team enough buffer to test the new integration properly.
Ahmad: This will increase the integration cost by RM20,000.
John: Compliance has reviewed the change. We can proceed.
Sarah: Sarah will take ownership of the launch workstream going forward.
"""

    print("--- Text extraction ---")
    result = extract_meeting_data(dummy_transcript)
    print(json.dumps(result, indent=2))

    print("\n--- Change reason (mock: comparing meeting 1 vs meeting 2) ---")

    test_cases = [
        {
            "field_name": "deadline",
            "old_value": "10 August",
            "new_value": "15 August",
            "transcript_snippet": "Sarah: The vendor requested five additional days, so I propose moving the launch from 10 August to 15 August."
        },
        {
            "field_name": "budget",
            "old_value": "RM50,000",
            "new_value": "RM70,000",
            "transcript_snippet": "Ahmad: This will increase the integration cost by RM20,000."
        },
        {
            "field_name": "owner",
            "old_value": "Ahmad",
            "new_value": "Sarah",
            "transcript_snippet": "Sarah: Sarah will take ownership of the launch workstream going forward."
        }
    ]

    for case in test_cases:
        reason = generate_change_reason(**case)
        print(f"[{case['field_name']}] {case['old_value']} → {case['new_value']}")
        print(f"Reason: {reason}\n")


    # Uncomment once you have a real sample audio file to test with:
    # print("\n--- Audio extraction ---")
    # audio_result = extract_from_audio("sample_meeting.mp3")
    # print(json.dumps(audio_result, indent=2))

    print("\n--- AI Search test ---")

    dummy_records = [
        {"meeting_id": 1, "speaker": "Sarah", "timestamp_seconds": 12, "text": "Launch date is 10 August, budget RM50,000.", "source_type": "transcript"},
        {"meeting_id": 2, "speaker": "Sarah", "timestamp_seconds": 45, "text": "The vendor requested five additional days, so I propose moving the launch from 10 August to 15 August.", "source_type": "transcript"},
        {"meeting_id": 2, "speaker": "Ahmad", "timestamp_seconds": 60, "text": "This will increase the integration cost by RM20,000.", "source_type": "transcript"},
    ]

    result = search_and_answer("Why was the deadline changed?", dummy_records)
    print(json.dumps(result, indent=2))
