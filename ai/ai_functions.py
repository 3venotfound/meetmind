import os
import re
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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

_EXTRACTION_RULES = """
Rules:
- "decided_by" = the person who made or proposed the decision, not who merely confirmed/approved it.
- "budget" value = the actual budget figure mentioned (not the delta/increase amount alone). If only a cost increase is mentioned without a base figure, state that in the summary and put the delta in "value" with a note that it's a delta, not a base figure.
- Every "decided_by", "owner", and "mentioned_by" MUST be a speaker name that literally appears in the transcript — never invent, generalize ("the team"), or guess a name.
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
    if not speakers:
        return issues  # nothing to validate against; skip silently

    def check(name, field_label):
        if name and name not in speakers:
            issues.append(f"{field_label} = '{name}' is not a speaker in this transcript")

    for d in data.get("decisions", []):
        check(d.get("decided_by"), "decision.decided_by")
    for a in data.get("action_items", []):
        check(a.get("owner"), "action_item.owner")
    for field in ("budget", "deadline", "owner"):
        check(data.get(field, {}).get("mentioned_by"), f"{field}.mentioned_by")

    return issues


def _generate_json(contents, schema, speakers_source: str, max_retries: int) -> dict:
    """Shared call-and-validate loop used by both entry points below."""
    last_error = None
    for attempt in range(max_retries + 1):
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
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
    return _generate_json(prompt, TEXT_SCHEMA, speakers_source=transcript_text, max_retries=max_retries)


def extract_from_audio(audio_path: str, max_retries: int = 1) -> dict:
    """
    Uploads an audio file and returns structured JSON (transcript + extraction)
    in a single Gemini call — Gemini does STT and extraction together.
    """
    audio_file = client.files.upload(file=audio_path)
    return _generate_json([AUDIO_PROMPT, audio_file], AUDIO_SCHEMA, speakers_source="audio", max_retries=max_retries)

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

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
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

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
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