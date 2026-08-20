import unittest
from pathlib import Path
from uuid import UUID

from app.integrations.errors import execution_failed
from app.integrations.path_safety import OwnedRunDirectory
from app.integrations.schemas import (
    AIActionItem,
    AIBudget,
    AIDeadline,
    AIDecision,
    AIExtractionResult,
    AIOwner,
    AISearchEvidence,
    AISearchResult,
    AITextExtractionResult,
    CVProcessingResult,
    CVVisualEvidence,
)
from app.processing import normalize_budget, normalize_deadline, parse_transcript
from tests.base import ApiTestCase


class FakeAI:
    def __init__(self, *, spoken_budget="RM70,000", visual_budget="RM50,000", fail=False):
        self.spoken_budget = spoken_budget
        self.visual_budget = visual_budget
        self.fail = fail
        self.search_records = []
        self.recording_calls = 0
        self.text_calls = 0
        self.reason_calls = 0

    async def extract_recording(self, path, visual_context=None):
        self.recording_calls += 1
        if self.fail:
            raise execution_failed("ai")
        visual = AITextExtractionResult(
            summary="Board details.", decisions=[], action_items=[],
            budget=AIBudget(value=self.visual_budget, mentioned_by=None, timestamp_seconds=2),
            deadline=AIDeadline(value=None, mentioned_by=None),
            owner=AIOwner(value=None, mentioned_by=None),
        ) if visual_context else None
        return AIExtractionResult(
            transcript="Sarah: Budget is RM70,000.\nUnstructured closing note",
            summary="The team confirmed delivery details.",
            decisions=[AIDecision(text="Proceed with launch", decided_by="Sarah", timestamp_seconds=5)],
            action_items=[AIActionItem(description="Update the plan", owner="Ahmad", due_date="20 August")],
            budget=AIBudget(value=self.spoken_budget, mentioned_by="Sarah", timestamp_seconds=4),
            deadline=AIDeadline(value="15 August", mentioned_by="Sarah", timestamp_seconds=6),
            owner=AIOwner(value=" Ahmad ", mentioned_by="Sarah", timestamp_seconds=7),
            visual_extraction=visual,
        )

    async def extract_text(self, text):
        self.text_calls += 1
        return AITextExtractionResult(
            summary="Board details.", decisions=[], action_items=[],
            budget=AIBudget(value=self.visual_budget, mentioned_by=None, timestamp_seconds=2),
            deadline=AIDeadline(value=None, mentioned_by=None),
            owner=AIOwner(value=None, mentioned_by=None),
        )

    async def generate_change_reason(self, field_name, old_value, new_value, source_snippet):
        self.reason_calls += 1
        raise AssertionError("generate_change_reason must not run during processing")

    async def search(self, question, records):
        self.search_records = records
        first = records[0]
        return AISearchResult(
            answer="The budget was confirmed in the meeting.",
            evidence=[AISearchEvidence(
                meeting_id=first["meeting_id"], speaker=first["speaker"],
                timestamp_seconds=first["timestamp_seconds"], source_type=first["source_type"],
            )],
        )


class FakeCV:
    def __init__(self, storage_root: Path, with_evidence=True):
        self.storage_root = storage_root
        self.with_evidence = with_evidence
        self.cleaned = False

    async def process_recording(self, recording_path, meeting_id):
        result = CVProcessingResult(meeting_id=meeting_id, visual_evidence=[])
        if self.with_evidence:
            run = OwnedRunDirectory(self.storage_root / "integration_runs" / "cv", "cv")
            evidence = run.path / "evidence"
            evidence.mkdir()
            image = evidence / "board.jpg"
            image.write_bytes(b"jpeg-test")
            result.visual_evidence.append(CVVisualEvidence(
                timestamp_seconds=2, evidence_type="whiteboard",
                raw_ocr_text="Budget: RM50,000", confidence=.91,
                image_path=image.relative_to(Path(__file__).resolve().parents[1]).as_posix(),
            ))
            result._owned_run_directory = run
        return result

    def cleanup_validated_run(self, result):
        if result._owned_run_directory:
            result._owned_run_directory.cleanup()
            result._owned_run_directory = None
        self.cleaned = True


class ProcessingRouteTests(ApiTestCase):
    def uploaded_meeting(self):
        meeting = self.create_meeting()
        response = self.client.post(
            f"/api/meetings/{meeting['id']}/recording",
            files={"file": ("meeting.webm", b"real-video", "video/webm")},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return meeting

    def install_fakes(self, ai=None, cv=None):
        ai = ai or FakeAI()
        cv = cv or FakeCV(self.storage_root)
        self.client.app.state.ai_adapter = ai
        self.client.app.state.cv_adapter = cv
        service = self.client.app.state.processing_service
        service.ai = ai
        service.cv = cv
        return ai, cv

    def create_uploaded_for_project(self, project_id, number, meeting_date):
        response = self.client.post("/api/meetings", json={
            "project_id": project_id, "title": f"Meeting {number}",
            "meeting_date": meeting_date, "meeting_number": number,
            "participants": [{"name": "Sarah", "role": "Lead"}],
        })
        self.assertEqual(response.status_code, 201, response.text)
        meeting = response.json()
        uploaded = self.client.post(
            f"/api/meetings/{meeting['id']}/recording",
            files={"file": ("meeting.webm", b"real-video", "video/webm")},
        )
        self.assertEqual(uploaded.status_code, 201, uploaded.text)
        return meeting

    def test_missing_recording_and_atomic_claim_conflicts(self):
        meeting = self.create_meeting()
        response = self.client.post(f"/api/meetings/{meeting['id']}/process")
        self.assertEqual(response.status_code, 409)
        uploaded = self.uploaded_meeting()
        self.client.app.state.repository.claim_processing(uploaded["id"])
        response = self.client.post(f"/api/meetings/{uploaded['id']}/process")
        self.assertEqual(response.status_code, 409)

    def test_success_persists_all_results_and_spoken_precedence(self):
        meeting = self.uploaded_meeting()
        ai, cv = self.install_fakes()
        response = self.client.post(f"/api/meetings/{meeting['id']}/process")
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["status"], "processed")
        self.assertEqual(result["summary"], "The team confirmed delivery details.")
        self.assertEqual(result["counts"], {
            "decisions": 5, "action_items": 1, "visual_evidence": 1,
            "changes": 0, "unresolved_action_items": 1,
        })
        budgets = [item for item in result["tracked_values"] if item["field_name"] == "budget"]
        self.assertEqual(len(budgets), 2)
        self.assertTrue(next(item for item in budgets if item["source_type"] == "transcript")["is_canonical"])
        self.assertFalse(next(item for item in budgets if item["source_type"] == "visual")["is_canonical"])
        self.assertEqual(next(item for item in budgets if item["source_type"] == "transcript")["budget_amount_minor"], 7_000_000)
        self.assertEqual(
            next(item for item in result["tracked_values"] if item["field_name"] == "deadline")["normalized_value"],
            "2026-08-15",
        )
        self.assertEqual(result["transcript_segments"][1]["speaker"], "Unknown")
        self.assertTrue(cv.cleaned)
        self.assertEqual(ai.recording_calls, 1)
        self.assertEqual(ai.text_calls, 0)
        evidence_id = result["visual_evidence"][0]["id"]
        image = self.client.get(f"/api/evidence/{evidence_id}/image")
        self.assertEqual(image.status_code, 200)
        with self.client.app.state.database.connection() as connection:
            stored = connection.execute("SELECT image_path FROM visual_evidence").fetchone()[0]
            self.assertFalse(Path(stored).is_absolute())
            self.assertNotIn("\\", stored)

    def test_duplicate_processing_is_409(self):
        meeting = self.uploaded_meeting()
        self.install_fakes()
        self.assertEqual(self.client.post(f"/api/meetings/{meeting['id']}/process").status_code, 200)
        self.assertEqual(self.client.post(f"/api/meetings/{meeting['id']}/process").status_code, 409)

    def test_failure_rolls_back_and_marks_failed(self):
        meeting = self.uploaded_meeting()
        self.install_fakes(ai=FakeAI(fail=True), cv=FakeCV(self.storage_root))
        response = self.client.post(f"/api/meetings/{meeting['id']}/process")
        self.assertEqual(response.status_code, 500)
        state = self.client.get(f"/api/meetings/{meeting['id']}").json()
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["processing_error"], "execution_failed")
        self.assertEqual(state["counts"]["decisions"], 0)

    def test_visual_only_value_is_canonical(self):
        meeting = self.uploaded_meeting()
        self.install_fakes(ai=FakeAI(spoken_budget=None))
        result = self.client.post(f"/api/meetings/{meeting['id']}/process").json()
        budget = next(item for item in result["tracked_values"] if item["field_name"] == "budget")
        self.assertEqual(budget["source_type"], "visual")
        self.assertTrue(budget["is_canonical"])

    def test_deterministic_change_detection_uses_canonical_values(self):
        project = self.create_project()
        first = self.create_uploaded_for_project(project["id"], 1, "2026-08-01")
        self.install_fakes(ai=FakeAI(spoken_budget="RM50,000", visual_budget=None), cv=FakeCV(self.storage_root, with_evidence=False))
        self.assertEqual(self.client.post(f"/api/meetings/{first['id']}/process").status_code, 200)
        second = self.create_uploaded_for_project(project["id"], 2, "2026-08-15")
        second_ai = FakeAI(spoken_budget="RM70,000", visual_budget=None)
        self.install_fakes(ai=second_ai, cv=FakeCV(self.storage_root, with_evidence=False))
        result = self.client.post(f"/api/meetings/{second['id']}/process").json()
        self.assertEqual(result["status"], "processed")
        budget_change = next(item for item in result["changes"] if item["field_name"] == "budget")
        self.assertEqual((budget_change["old_value"], budget_change["new_value"]), ("RM50,000", "RM70,000"))
        self.assertIsNone(budget_change["reason"])
        self.assertEqual(second_ai.recording_calls, 1)
        self.assertEqual(second_ai.reason_calls, 0)
        history = self.client.get(f"/api/projects/{project['id']}/history")
        self.assertEqual(history.status_code, 200, history.text)
        changed_budget_history = [
            item for item in history.json()["history"]
            if item["meeting_id"] == second["id"] and item["field_name"] == "budget"
        ]
        self.assertEqual(len(changed_budget_history), 1)
        self.assertIsNone(changed_budget_history[0]["reason"])

    def test_history_and_search_are_project_scoped(self):
        meeting = self.uploaded_meeting()
        ai, _ = self.install_fakes()
        self.client.post(f"/api/meetings/{meeting['id']}/process")
        project_id = self.client.get(f"/api/meetings/{meeting['id']}").json()["project_id"]
        history = self.client.get(f"/api/projects/{project_id}/history")
        self.assertEqual(history.status_code, 200)
        self.assertTrue(all(item["meeting_id"] == meeting["id"] for item in history.json()["history"]))
        search = self.client.post("/api/search", json={"project_id": project_id, "question": "What budget was confirmed?"})
        self.assertEqual(search.status_code, 200, search.text)
        self.assertTrue(search.json()["evidence"])
        self.assertTrue(all(item["meeting_id"] == meeting["id"] for item in search.json()["evidence"]))
        self.assertTrue(ai.search_records)

    def test_search_without_support_returns_required_message(self):
        project = self.create_project()
        response = self.client.post("/api/search", json={"project_id": project["id"], "question": "unknown topic"})
        self.assertEqual(response.json(), {"answer": "Not found in project evidence.", "evidence": []})

    def test_unknown_evidence_and_project_history_return_404(self):
        self.assertEqual(self.client.get("/api/evidence/00000000-0000-0000-0000-000000000000/image").status_code, 404)
        self.assertEqual(self.client.get("/api/projects/00000000-0000-0000-0000-000000000000/history").status_code, 404)

    def test_unsafe_evidence_database_path_is_not_served(self):
        meeting = self.create_meeting()
        evidence_id = "11111111-1111-1111-1111-111111111111"
        with self.client.app.state.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO visual_evidence
                    (id, meeting_id, timestamp_seconds, evidence_type, raw_ocr_text,
                     confidence, image_path, created_at)
                VALUES (?, ?, 0, 'unknown', '', 0, '../secret.jpg',
                        '2026-08-19T00:00:00Z')
                """,
                (evidence_id, meeting["id"]),
            )
        self.assertEqual(self.client.get(f"/api/evidence/{evidence_id}/image").status_code, 404)


class NormalizationTests(unittest.TestCase):
    def test_budget_deadline_and_transcript_normalization(self):
        self.assertEqual(normalize_budget("RM70,000")["amount_minor"], 7_000_000)
        self.assertIsNone(normalize_budget("increase by RM20,000")["amount_minor"])
        self.assertEqual(normalize_deadline("15 August", 2026)["normalized"], "2026-08-15")
        self.assertIsNone(normalize_deadline("next month", 2026)["normalized"])
        self.assertIsNone(parse_transcript("No label")[0]["start_time_seconds"])
