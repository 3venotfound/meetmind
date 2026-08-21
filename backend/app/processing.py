import logging
import os
import re
import shutil
from calendar import month_name
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

from app.config import BACKEND_DIR
from app.integrations import AIAdapter, CVAdapter
from app.integrations.errors import IntegrationError
from app.integrations.path_safety import UnsafePathError, resolve_stored_recording_path
from app.integrations.schemas import AIExtractionResult, AITextExtractionResult, CVProcessingResult
from app.repositories import Repository


logger = logging.getLogger(__name__)
NOT_FOUND_ANSWER = "Not found in project evidence."


def parse_transcript(transcript: str) -> list[dict]:
    segments = []
    for raw_line in transcript.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^([^:\n]{1,100}):\s*(.+)$", line)
        speaker, text = (match.group(1).strip(), match.group(2).strip()) if match else ("Unknown", line)
        if text:
            segments.append({"speaker": speaker or "Unknown", "text": text, "start_time_seconds": None})
    return segments


def normalize_budget(raw_value: str | None) -> dict:
    raw = (raw_value or "").strip()
    result = {"raw": raw, "normalized": None, "amount_minor": None, "currency": None}
    if not raw or re.search(r"\b(increase|decrease|delta|additional|extra|reduction)\b", raw, re.I):
        return result
    match = re.search(r"(?i)(RM|MYR|USD|\$|EUR|€|GBP|£)\s*([0-9][0-9,]*(?:\.\d{1,2})?)", raw)
    if not match:
        return result
    currency = {"RM": "MYR", "MYR": "MYR", "USD": "USD", "$": "USD", "EUR": "EUR", "€": "EUR", "GBP": "GBP", "£": "GBP"}[match.group(1).upper()]
    try:
        amount_minor = int(round(float(match.group(2).replace(",", "")) * 100))
    except ValueError:
        return result
    result.update(normalized=f"{currency}:{amount_minor}", amount_minor=amount_minor, currency=currency)
    return result


MONTHS = {name.lower(): number for number, name in enumerate(month_name) if name}
MONTHS.update({name[:3].lower(): number for name, number in MONTHS.items()})


def normalize_deadline(raw_value: str | None, meeting_year: int) -> dict:
    raw = (raw_value or "").strip()
    result = {"raw": raw, "normalized": None}
    if not raw:
        return result
    try:
        result["normalized"] = date.fromisoformat(raw).isoformat()
        return result
    except ValueError:
        pass
    match = re.fullmatch(r"\s*(\d{1,2})\s+([A-Za-z]+)(?:\s+(\d{4}))?\s*", raw)
    if not match:
        return result
    month = MONTHS.get(match.group(2).lower())
    if not month:
        return result
    try:
        result["normalized"] = date(int(match.group(3) or meeting_year), month, int(match.group(1))).isoformat()
    except ValueError:
        pass
    return result


def normalize_owner(raw_value: str | None) -> dict:
    raw = (raw_value or "").strip()
    return {"raw": raw, "normalized": raw.casefold() if raw else None}


def _normalize_field(field_name: str, raw_value: str | None, meeting_year: int) -> dict:
    if field_name == "budget":
        return normalize_budget(raw_value)
    if field_name == "deadline":
        return normalize_deadline(raw_value, meeting_year)
    return normalize_owner(raw_value)


def _values_equal(field_name: str, current: dict, previous: dict) -> bool:
    if field_name == "budget" and current.get("budget_amount_minor") is not None and previous.get("budget_amount_minor") is not None:
        return (current["budget_amount_minor"], current.get("currency_code")) == (previous["budget_amount_minor"], previous.get("currency_code"))
    current_value = current.get("normalized_value") or current["field_value"].strip().casefold()
    previous_value = previous.get("normalized_value") or previous["field_value"].strip().casefold()
    return current_value == previous_value


class EvidenceStore:
    def __init__(self, storage_root: Path):
        self.storage_root = storage_root.resolve()
        self.evidence_root = self.storage_root / "evidence"

    def promote(self, meeting_id: UUID, cv_result: CVProcessingResult) -> tuple[list[dict], list[Path]]:
        records, created = [], []
        meeting_directory = self.evidence_root / str(meeting_id)
        meeting_directory.mkdir(parents=True, exist_ok=True)
        try:
            for evidence in cv_result.visual_evidence:
                source = (BACKEND_DIR / PurePosixPath(evidence.image_path)).resolve(strict=True)
                run = cv_result._owned_run_directory
                if run is None:
                    raise UnsafePathError
                source.relative_to(run.path / "evidence")
                evidence_id = str(uuid4())
                final_path = meeting_directory / f"{evidence_id}.jpg"
                part_path = meeting_directory / f".{uuid4().hex}.part"
                try:
                    with source.open("rb") as input_file, part_path.open("xb") as output_file:
                        shutil.copyfileobj(input_file, output_file, 1024 * 1024)
                        output_file.flush()
                        os.fsync(output_file.fileno())
                    os.link(part_path, final_path)
                    created.append(final_path)
                finally:
                    if part_path.exists():
                        part_path.unlink()
                records.append({
                    "id": evidence_id,
                    "timestamp_seconds": evidence.timestamp_seconds,
                    "evidence_type": evidence.evidence_type,
                    "raw_ocr_text": evidence.raw_ocr_text,
                    "confidence": evidence.confidence,
                    "image_path": final_path.relative_to(BACKEND_DIR).as_posix(),
                })
            return records, created
        except Exception:
            self.cleanup(created)
            raise

    @staticmethod
    def cleanup(paths: list[Path]) -> None:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.error("Could not remove processing-attempt evidence file")

    def resolve(self, relative_path: str) -> Path:
        if "\\" in relative_path:
            raise UnsafePathError
        pure = PurePosixPath(relative_path)
        if pure.is_absolute() or ".." in pure.parts:
            raise UnsafePathError
        path = BACKEND_DIR.joinpath(*pure.parts).resolve(strict=True)
        path.relative_to(self.evidence_root.resolve(strict=True))
        if path.suffix.lower() != ".jpg" or not path.is_file():
            raise UnsafePathError
        return path


class ProcessingService:
    def __init__(self, repository: Repository, ai: AIAdapter, cv: CVAdapter, storage_root: Path):
        self.repository = repository
        self.ai = ai
        self.cv = cv
        self.storage_root = storage_root.resolve()
        self.evidence_store = EvidenceStore(self.storage_root)

    async def process(self, meeting_id: UUID) -> dict:
        claimed = self.repository.claim_processing(str(meeting_id))
        cv_result = None
        created_files: list[Path] = []
        persisted = False
        try:
            recording_path = resolve_stored_recording_path(claimed["recording_path"], meeting_id, self.storage_root)
            cv_result = await self.cv.process_recording(recording_path, meeting_id)
            visual_context = "\n\n".join(
                f"Visual evidence at {item.timestamp_seconds} seconds:\n{item.raw_ocr_text.strip()}"
                for item in cv_result.visual_evidence
                if item.raw_ocr_text.strip()
            )[:50_000]
            ai_result = await self.ai.extract_recording(
                recording_path,
                visual_context=visual_context or None,
            )
            visual_result = ai_result.visual_extraction
            evidence_records, created_files = self.evidence_store.promote(meeting_id, cv_result)
            payload = await self._build_payload(claimed, ai_result, visual_result, evidence_records)
            self.repository.persist_processing(str(meeting_id), payload)
            persisted = True
            meeting = self.repository.get_meeting(str(meeting_id))
            if meeting is None:
                raise RuntimeError
            return meeting
        except Exception as error:
            if not persisted:
                self.evidence_store.cleanup(created_files)
                code = error.code.value if isinstance(error, IntegrationError) else "processing_failed"
                diagnostic = (
                    error.diagnostic if isinstance(error, IntegrationError) else None
                )
                if diagnostic is not None:
                    logger.error(
                        "Meeting processing failed for %s component=%s stage=%s category=%s code=%s",
                        meeting_id,
                        error.component,
                        getattr(diagnostic, "stage", "unknown"),
                        getattr(diagnostic, "category", "unknown"),
                        code,
                    )
                else:
                    logger.error("Meeting processing failed for %s with code=%s", meeting_id, code)
                self.repository.mark_processing_failed(str(meeting_id), code)
            else:
                logger.error("Processed meeting could not be read back: %s", meeting_id)
            raise
        finally:
            if cv_result is not None:
                try:
                    self.cv.cleanup_validated_run(cv_result)
                except Exception:
                    logger.error("CV temporary-run cleanup failed for meeting %s", meeting_id)

    async def _build_payload(self, claimed: dict, spoken: AIExtractionResult, visual: AITextExtractionResult | None, evidence: list[dict]) -> dict:
        project_id = claimed["project_id"]
        meeting_year = date.fromisoformat(claimed["meeting_date"]).year
        previous = self.repository.previous_canonical_values(project_id, claimed["id"])
        decisions = []
        for item in spoken.decisions:
            decisions.append(self._decision(project_id, "decision_text", item.text, None, item.decided_by, item.timestamp_seconds, "transcript", True, spoken.transcript, None))
        if visual:
            evidence_id = evidence[0]["id"] if evidence else None
            for item in visual.decisions:
                decisions.append(self._decision(project_id, "decision_text", item.text, None, item.decided_by, item.timestamp_seconds, "visual", False, "", evidence_id))

        canonical = {}
        for field_name in ("budget", "deadline", "owner"):
            spoken_value = getattr(spoken, field_name)
            visual_value = getattr(visual, field_name) if visual else None
            spoken_raw = (spoken_value.value or "").strip()
            visual_raw = (visual_value.value or "").strip() if visual_value else ""
            if spoken_raw:
                row = self._tracked(project_id, field_name, spoken_value, "transcript", True, meeting_year, spoken.transcript, None)
                decisions.append(row)
                canonical[field_name] = row
            if visual_raw:
                evidence_id = self._matching_evidence(visual_raw, evidence)
                row = self._tracked(project_id, field_name, visual_value, "visual", not spoken_raw, meeting_year, visual_raw, evidence_id)
                decisions.append(row)
                if not spoken_raw:
                    canonical[field_name] = row

        changes = []
        for field_name, current in canonical.items():
            old = previous.get(field_name)
            if old and not _values_equal(field_name, current, old):
                changes.append({
                    "project_id": project_id, "field_name": field_name,
                    "old_value": old["field_value"], "new_value": current["field_value"],
                    "old_budget_amount_minor": old.get("budget_amount_minor"),
                    "new_budget_amount_minor": current.get("budget_amount_minor"),
                    "currency_code": current.get("currency_code") or old.get("currency_code"),
                    "from_meeting_id": old["meeting_id"], "reason": None,
                    "changed_by": current.get("decided_by"), "source_type": current["source_type"],
                    "timestamp_seconds": current.get("timestamp_seconds"),
                })

        actions = []
        for item in spoken.action_items:
            due = normalize_deadline(item.due_date, meeting_year)["normalized"] if item.due_date else None
            actions.append({"description": item.description.strip(), "owner": item.owner.strip(), "due_date": due})
        return {
            "transcript": spoken.transcript,
            "summary": spoken.summary,
            "transcript_segments": parse_transcript(spoken.transcript),
            "visual_evidence": evidence,
            "decisions": decisions,
            "action_items": actions,
            "changes": changes,
        }

    @staticmethod
    def _matching_evidence(value: str, evidence: list[dict]) -> str | None:
        lowered = value.casefold()
        for item in evidence:
            if lowered in item["raw_ocr_text"].casefold() or item["raw_ocr_text"].casefold() in lowered:
                return item["id"]
        return evidence[0]["id"] if evidence else None

    @staticmethod
    def _decision(project_id, field_name, raw, normalized, by, timestamp, source, canonical, snippet, evidence_id):
        return {"project_id": project_id, "field_name": field_name, "field_value": raw.strip(), "normalized_value": normalized, "decided_by": (by or "").strip() or None, "timestamp_seconds": timestamp, "source_type": source, "is_canonical": canonical, "reasoning_snippet": snippet[:1000] if snippet else None, "visual_evidence_id": evidence_id}

    def _tracked(self, project_id, field_name, value, source, canonical, year, snippet, evidence_id):
        normalized = _normalize_field(field_name, value.value, year)
        row = self._decision(project_id, field_name, normalized["raw"], normalized["normalized"], value.mentioned_by, value.timestamp_seconds, source, canonical, snippet, evidence_id)
        row["budget_amount_minor"] = normalized.get("amount_minor")
        row["currency_code"] = normalized.get("currency")
        return row

    async def search(self, project_id: UUID, question: str) -> dict | None:
        records = self.repository.search_records(str(project_id))
        if records is None:
            return None
        terms = {word for word in re.findall(r"[a-z0-9]+", question.casefold()) if len(word) > 2}
        ranked = sorted(records, key=lambda item: sum(term in item["text"].casefold() for term in terms), reverse=True)
        selected = [item for item in ranked if any(term in item["text"].casefold() for term in terms)][:12]
        if not selected:
            return {"answer": NOT_FOUND_ANSWER, "evidence": []}
        ai_records = [{"meeting_id": x["meeting_id"], "speaker": x["speaker"], "timestamp_seconds": x["timestamp_seconds"], "source_type": x["source_type"], "text": x["text"][:2_000]} for x in selected]
        result = await self.ai.search(question, ai_records)
        enriched = []
        for item in result.evidence:
            match = next((x for x in selected if x["meeting_id"] == item.meeting_id and x["speaker"] == item.speaker and x["timestamp_seconds"] == item.timestamp_seconds and x["source_type"] == item.source_type), None)
            if match:
                enriched.append({**match, "image_url": f"/api/evidence/{match['evidence_id']}/image" if match["evidence_id"] else None})
        unsupported = any(
            phrase in result.answer.casefold()
            for phrase in ("not enough", "cannot answer", "can't answer", "do not contain", "does not contain")
        )
        if not enriched or unsupported:
            return {"answer": NOT_FOUND_ANSWER, "evidence": []}
        return {"answer": result.answer, "evidence": enriched}
