import unittest
from pathlib import Path


FRONTEND = (Path(__file__).resolve().parents[2] / "frontend" / "index.html").read_text(encoding="utf-8")


class FrontendStaticTests(unittest.TestCase):
    def test_real_recording_and_required_routes_are_present(self):
        self.assertIn('const API_BASE_URL = "http://127.0.0.1:8000"', FRONTEND)
        self.assertIn("navigator.mediaDevices.getUserMedia", FRONTEND)
        self.assertIn("new MediaRecorder", FRONTEND)
        self.assertIn('form.append("file"', FRONTEND)
        for route in (
            "/api/projects", "/api/meetings", "/recording", "/process",
            "/history", "/api/search",
        ):
            self.assertIn(route, FRONTEND)

    def test_obsolete_simulation_is_absent(self):
        self.assertNotIn("fake processing", FRONTEND.lower())
        self.assertNotIn("setTimeout(() =>", FRONTEND)
        self.assertNotIn("const answers", FRONTEND)
        self.assertNotIn("detectFrame", FRONTEND)

    def test_camera_is_not_started_during_page_load(self):
        load_call = FRONTEND.rfind("loadProjects().catch")
        self.assertGreater(load_call, 0)
        self.assertNotIn("startRecording()", FRONTEND[load_call:])

    def test_multiple_project_dashboard_uses_real_api_data(self):
        self.assertIn('state.projects=await api("/api/projects")', FRONTEND)
        self.assertIn('id="createProjectForm"', FRONTEND)
        self.assertIn('async function selectProject', FRONTEND)
        self.assertIn("No projects yet", FRONTEND)
        self.assertIn('method:"POST"', FRONTEND)
        self.assertNotIn("Digital Banking Revamp", FRONTEND)
        self.assertNotIn("FINOVA BANK", FRONTEND)

    def test_only_selected_project_is_stored_in_local_storage(self):
        self.assertIn('const PROJECT_KEY = "meetmind_project_id"', FRONTEND)
        self.assertNotIn("LAST_MEETING_KEY", FRONTEND)
        self.assertIn('localStorage.removeItem("meetmind_last_meeting_id")', FRONTEND)
        storage_writes = [
            line.strip() for line in FRONTEND.splitlines()
            if "localStorage.setItem" in line
        ]
        self.assertTrue(storage_writes)
        self.assertTrue(all("PROJECT_KEY" in line for line in storage_writes))

    def test_selected_project_scopes_meetings_history_and_search(self):
        self.assertIn("project_id:state.project.id", FRONTEND)
        self.assertIn("/history`", FRONTEND)
        self.assertIn("meeting.project_id!==state.project.id", FRONTEND)

    def test_processed_meeting_can_be_selected_without_a_hard_coded_uuid(self):
        self.assertIn('startupParams.get("project_id")', FRONTEND)
        self.assertIn('startupParams.get("meeting_id")', FRONTEND)
        self.assertNotRegex(
            FRONTEND,
            r"9fe67b71-fa6e-46bf-8945-df7c78b41392",
        )

    def test_null_change_reason_is_rendered_safely(self):
        self.assertIn('x.reason==null?"Reason not generated":x.reason', FRONTEND)
        self.assertNotIn('escapeHtml(x.reason||"")', FRONTEND)
