import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.api.health import ComponentHealth
from app.main import app


class AppTests(unittest.TestCase):
    def test_root_returns_service_identity(self) -> None:
        response = TestClient(app).get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"service": "online-study-platform-ai", "status": "UP"},
        )

    def test_health_returns_up_when_datastores_are_reachable(self) -> None:
        with patch("app.api.health.check_postgresql") as check_postgresql:
            check_postgresql.return_value = ComponentHealth(status="UP")
            with patch("app.api.health.check_mongodb", new_callable=AsyncMock) as check_mongodb:
                check_mongodb.return_value = ComponentHealth(status="UP")

                response = TestClient(app).get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "UP",
                "components": {
                    "postgresql": {"status": "UP"},
                    "mongodb": {"status": "UP"},
                },
            },
        )

    def test_notifications_endpoint_is_removed(self) -> None:
        response = TestClient(app).get("/notifications")

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
