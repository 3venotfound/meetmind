from uuid import uuid4

from tests.base import ApiTestCase


class MeetingRouteTests(ApiTestCase):
    def meeting_payload(self, project_id: str, **overrides) -> dict:
        payload = {
            "project_id": project_id,
            "title": "Weekly Project Steering Committee",
            "meeting_date": "2026-08-15",
            "meeting_number": 12,
            "participants": [
                {"name": "Sarah", "role": "PM"},
                {"name": "Ahmad", "role": "Engineering Lead"},
                {"name": "John", "role": "Compliance"},
            ],
        }
        payload.update(overrides)
        return payload

    def test_create_and_retrieve_meeting(self) -> None:
        project = self.create_project()
        response = self.client.post(
            "/api/meetings",
            json=self.meeting_payload(project["id"]),
        )
        self.assertEqual(response.status_code, 201, response.text)
        meeting = response.json()
        self.assertEqual(meeting["project_id"], project["id"])
        self.assertEqual(meeting["status"], "created")
        self.assertEqual(meeting["meeting_date"], "2026-08-15")
        self.assertEqual(
            [participant["name"] for participant in meeting["participants"]],
            ["Sarah", "Ahmad", "John"],
        )
        self.assertEqual(
            meeting["counts"],
            {
                "decisions": 0,
                "action_items": 0,
                "visual_evidence": 0,
                "changes": 0,
                "unresolved_action_items": 0,
            },
        )

        retrieved = self.client.get(f"/api/meetings/{meeting['id']}")
        self.assertEqual(retrieved.status_code, 200, retrieved.text)
        self.assertEqual(retrieved.json(), meeting)

        project_response = self.client.get(f"/api/projects/{project['id']}")
        self.assertEqual(project_response.status_code, 200)
        project_data = project_response.json()
        self.assertEqual(project_data["stats"]["meetings_logged"], 1)
        self.assertEqual(project_data["recent_meetings"][0]["state"], "baseline")

    def test_unknown_project_returns_404_when_creating_meeting(self) -> None:
        response = self.client.post(
            "/api/meetings",
            json=self.meeting_payload(str(uuid4())),
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Project not found"})

    def test_unknown_meeting_returns_404(self) -> None:
        response = self.client.get(f"/api/meetings/{uuid4()}")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Meeting not found"})

    def test_duplicate_meeting_number_returns_409(self) -> None:
        project = self.create_project()
        first = self.client.post(
            "/api/meetings",
            json=self.meeting_payload(project["id"]),
        )
        self.assertEqual(first.status_code, 201, first.text)

        duplicate = self.client.post(
            "/api/meetings",
            json=self.meeting_payload(
                project["id"],
                title="Another meeting with the same number",
                meeting_date="2026-08-18",
            ),
        )
        self.assertEqual(duplicate.status_code, 409, duplicate.text)
        self.assertEqual(
            duplicate.json(),
            {"detail": "Meeting number already exists for this project"},
        )

        project_response = self.client.get(f"/api/projects/{project['id']}")
        self.assertEqual(project_response.json()["stats"]["meetings_logged"], 1)

    def test_same_meeting_number_is_allowed_in_different_projects(self) -> None:
        first_project = self.create_project(name="First project")
        second_project = self.create_project(name="Second project")
        first = self.client.post(
            "/api/meetings",
            json=self.meeting_payload(first_project["id"]),
        )
        second = self.client.post(
            "/api/meetings",
            json=self.meeting_payload(second_project["id"]),
        )
        self.assertEqual(first.status_code, 201, first.text)
        self.assertEqual(second.status_code, 201, second.text)

    def test_whitespace_only_meeting_title_is_rejected(self) -> None:
        project = self.create_project()
        response = self.client.post(
            "/api/meetings",
            json=self.meeting_payload(project["id"], title="   "),
        )
        self.assertEqual(response.status_code, 422)

    def test_whitespace_only_participant_name_is_rejected(self) -> None:
        project = self.create_project()
        response = self.client.post(
            "/api/meetings",
            json=self.meeting_payload(
                project["id"],
                participants=[{"name": "\t", "role": "PM"}],
            ),
        )
        self.assertEqual(response.status_code, 422)
