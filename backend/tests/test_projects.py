from uuid import uuid4

from tests.base import ApiTestCase


class ProjectRouteTests(ApiTestCase):
    def test_empty_project_list(self) -> None:
        response = self.client.get("/api/projects")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), [])

    def test_list_projects_with_isolated_counts_and_recent_order(self) -> None:
        older = self.create_project(name="Older project", client_org="Client A")
        newer = self.create_project(name="Newer project", client_org="Client B")
        meeting_response = self.client.post(
            "/api/meetings",
            json={
                "project_id": older["id"],
                "meeting_number": 1,
                "title": "Recent work on older project",
                "meeting_date": "2026-08-18",
                "participants": [],
            },
        )
        self.assertEqual(meeting_response.status_code, 201, meeting_response.text)
        meeting = meeting_response.json()

        with self.client.app.state.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO action_items (
                    meeting_id, description, owner, due_date, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    meeting["id"], "Open action", "Alex", None, "in_progress",
                    "2026-08-20T00:00:00Z", "2026-08-20T00:00:00Z",
                ),
            )
            connection.execute(
                """
                INSERT INTO changes (
                    project_id, field_name, old_value, new_value,
                    from_meeting_id, to_meeting_id, reason, changed_by,
                    source_type, timestamp_seconds, detected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    older["id"], "owner", "A", "B", meeting["id"],
                    meeting["id"], None, "Alex", "transcript", None,
                    "2026-08-20T00:00:00Z",
                ),
            )
            connection.execute(
                "UPDATE meetings SET updated_at = ? WHERE id = ?",
                ("2099-01-01T00:00:00Z", meeting["id"]),
            )

        response = self.client.get("/api/projects")
        self.assertEqual(response.status_code, 200, response.text)
        projects = response.json()
        self.assertEqual([project["id"] for project in projects], [older["id"], newer["id"]])
        self.assertEqual(
            set(projects[0]),
            {
                "id", "name", "client_org", "target_industry", "created_at",
                "meeting_count", "change_count", "unresolved_action_count",
            },
        )
        self.assertEqual(projects[0]["meeting_count"], 1)
        self.assertEqual(projects[0]["change_count"], 1)
        self.assertEqual(projects[0]["unresolved_action_count"], 1)
        self.assertEqual(projects[1]["meeting_count"], 0)
        self.assertEqual(projects[1]["change_count"], 0)
        self.assertEqual(projects[1]["unresolved_action_count"], 0)

    def test_project_detail_and_history_remain_isolated(self) -> None:
        project_a = self.create_project(name="Project A", client_org="Client A")
        project_b = self.create_project(name="Project B", client_org="Client B")
        meeting_a_response = self.client.post(
            "/api/meetings",
            json={
                "project_id": project_a["id"], "meeting_number": 1,
                "title": "A meeting", "meeting_date": "2026-08-18", "participants": [],
            },
        )
        meeting_b_response = self.client.post(
            "/api/meetings",
            json={
                "project_id": project_b["id"], "meeting_number": 1,
                "title": "B meeting", "meeting_date": "2026-08-18", "participants": [],
            },
        )
        self.assertEqual(meeting_a_response.status_code, 201, meeting_a_response.text)
        self.assertEqual(meeting_b_response.status_code, 201, meeting_b_response.text)
        meeting_a = meeting_a_response.json()
        meeting_b = meeting_b_response.json()

        with self.client.app.state.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO decisions (
                    project_id, meeting_id, field_name, field_value,
                    normalized_value, source_type, is_canonical, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_a["id"], meeting_a["id"], "deadline", "2026-09-01",
                    "2026-09-01", "transcript", 1, "2026-08-20T00:00:00Z",
                ),
            )

        detail_a = self.client.get(f"/api/projects/{project_a['id']}").json()
        detail_b = self.client.get(f"/api/projects/{project_b['id']}").json()
        self.assertEqual([item["id"] for item in detail_a["recent_meetings"]], [meeting_a["id"]])
        self.assertEqual([item["id"] for item in detail_b["recent_meetings"]], [meeting_b["id"]])
        self.assertEqual(
            self.client.get(f"/api/projects/{project_b['id']}/history").json()["history"],
            [],
        )
        history_a = self.client.get(f"/api/projects/{project_a['id']}/history").json()["history"]
        self.assertEqual(len(history_a), 1)
        self.assertEqual(history_a[0]["meeting_id"], meeting_a["id"])

    def test_diagnostic_routes_are_available(self) -> None:
        self.assertEqual(
            self.client.get("/").json(),
            {"message": "MeetMind API is running"},
        )
        self.assertEqual(
            self.client.get("/health").json(),
            {"status": "healthy"},
        )

    def test_create_and_retrieve_project(self) -> None:
        created = self.create_project()
        self.assertEqual(created["name"], "Digital Banking Revamp")
        self.assertEqual(created["client_org"], "FINOVA BANK")
        self.assertEqual(
            created["stats"],
            {
                "meetings_logged": 0,
                "decisions_changed": 0,
                "unresolved_issues": 0,
            },
        )
        self.assertEqual(created["recent_meetings"], [])
        self.assertEqual(created["current_memory"], [])

        retrieved = self.client.get(f"/api/projects/{created['id']}")
        self.assertEqual(retrieved.status_code, 200, retrieved.text)
        self.assertEqual(retrieved.json(), created)

    def test_unknown_project_returns_404(self) -> None:
        response = self.client.get(f"/api/projects/{uuid4()}")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Project not found"})

    def test_whitespace_only_project_name_is_rejected(self) -> None:
        response = self.client.post(
            "/api/projects",
            json={"name": "   ", "client_org": "FINOVA BANK"},
        )
        self.assertEqual(response.status_code, 422)

    def test_whitespace_only_client_name_is_rejected(self) -> None:
        response = self.client.post(
            "/api/projects",
            json={"name": "Project", "client_org": "\t  "},
        )
        self.assertEqual(response.status_code, 422)

    def test_sql_like_project_name_is_stored_as_plain_data(self) -> None:
        suspicious_name = "Project'); DROP TABLE projects; --"
        created = self.create_project(name=suspicious_name)
        retrieved = self.client.get(f"/api/projects/{created['id']}")
        self.assertEqual(retrieved.status_code, 200)
        self.assertEqual(retrieved.json()["name"], suspicious_name)

    def test_configured_cors_origin_is_allowed(self) -> None:
        response = self.client.options(
            "/api/projects",
            headers={
                "Origin": "http://testserver",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            "http://testserver",
        )
