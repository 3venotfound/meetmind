from uuid import uuid4

from tests.base import ApiTestCase


class ProjectRouteTests(ApiTestCase):
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
