import unittest

from fastapi.testclient import TestClient

from app.core.config import get_ai_settings
from app.main import app


def recommendation_payload() -> dict:
    return {
        "learnerId": "learner-1",
        "limit": 8,
        "signals": {
            "categories": {"Backend": 2.0},
            "completedCourseIds": ["course-done"],
            "engagementByCourseId": {"course-done": 4.0},
            "quizPerformanceByCourseId": {"course-done": 0.8},
        },
        "candidates": [
            {
                "id": "course-design",
                "title": "Design Systems",
                "category": "Design",
                "description": "Build consistent product UI.",
                "popularity": 999,
            },
            {
                "id": "course-backend",
                "title": "Backend APIs",
                "category": "Backend",
                "description": "Build API services.",
                "popularity": 1,
            },
            {
                "id": "course-node",
                "title": "Node Services",
                "category": "Backend",
                "description": "Build Node.js services.",
                "popularity": 3,
            },
        ],
    }


class RecommendationTests(unittest.TestCase):
    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_recommend_returns_deterministic_ordered_course_ids(self) -> None:
        client = TestClient(app)
        payload = recommendation_payload()

        first_response = client.post("/recommend", json=payload)
        second_response = client.post("/recommend", json=payload)

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        self.assertEqual(first_response.json(), second_response.json())
        self.assertEqual(
            first_response.json(),
            {"courseIds": ["course-node", "course-backend", "course-design"]},
        )

    def test_category_affinity_ranks_above_popularity(self) -> None:
        payload = recommendation_payload()

        response = TestClient(app).post("/recommend", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["courseIds"][:2],
            ["course-node", "course-backend"],
        )

    def test_title_and_id_break_ties_stably(self) -> None:
        payload = {
            "learnerId": "learner-1",
            "limit": 5,
            "signals": {
                "categories": {},
                "completedCourseIds": [],
                "engagementByCourseId": {},
                "quizPerformanceByCourseId": {},
            },
            "candidates": [
                {
                    "id": "course-z",
                    "title": "Same Title",
                    "category": "Backend",
                    "description": "Z",
                    "popularity": 1,
                },
                {
                    "id": "course-a",
                    "title": "Same Title",
                    "category": "Backend",
                    "description": "A",
                    "popularity": 1,
                },
                {
                    "id": "course-b",
                    "title": "Alpha Title",
                    "category": "Backend",
                    "description": "B",
                    "popularity": 1,
                },
            ],
        }

        response = TestClient(app).post("/recommend", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["courseIds"],
            ["course-b", "course-a", "course-z"],
        )

    def test_title_sort_uses_raw_title_before_id_tiebreaker(self) -> None:
        payload = {
            "learnerId": "learner-1",
            "limit": 5,
            "signals": {
                "categories": {},
                "completedCourseIds": [],
                "engagementByCourseId": {},
                "quizPerformanceByCourseId": {},
            },
            "candidates": [
                {
                    "id": "course-a",
                    "title": "alpha",
                    "category": "Backend",
                    "description": "Lowercase title.",
                    "popularity": 1,
                },
                {
                    "id": "course-z",
                    "title": "Alpha",
                    "category": "Backend",
                    "description": "Uppercase title.",
                    "popularity": 1,
                },
            ],
        }

        response = TestClient(app).post("/recommend", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["courseIds"], ["course-z", "course-a"])

    def test_limit_is_respected(self) -> None:
        payload = recommendation_payload()
        payload["limit"] = 2

        response = TestClient(app).post("/recommend", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["courseIds"],
            ["course-node", "course-backend"],
        )

    def test_duplicate_candidate_ids_keep_first_valid_occurrence_before_ranking(
        self,
    ) -> None:
        payload = {
            "learnerId": "learner-1",
            "limit": 5,
            "signals": {
                "categories": {"Backend": 1.0},
                "completedCourseIds": [],
                "engagementByCourseId": {},
                "quizPerformanceByCourseId": {},
            },
            "candidates": [
                {
                    "id": "course-1",
                    "title": "First Copy",
                    "category": "Design",
                    "description": "First candidate wins during dedupe.",
                    "popularity": 1,
                },
                {
                    "id": "course-1",
                    "title": "Second Copy",
                    "category": "Backend",
                    "description": "This duplicate should be ignored.",
                    "popularity": 999,
                },
                {
                    "id": "course-2",
                    "title": "Backend Match",
                    "category": "Backend",
                    "description": "Matches affinity.",
                    "popularity": 0,
                },
            ],
        }

        response = TestClient(app).post("/recommend", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["courseIds"], ["course-2", "course-1"])

    def test_empty_candidates_returns_empty_course_ids(self) -> None:
        payload = recommendation_payload()
        payload["candidates"] = []

        response = TestClient(app).post("/recommend", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"courseIds": []})

    def test_invalid_token_is_rejected_when_ai_token_is_configured(self) -> None:
        class Settings:
            ai_token = "secret"

        app.dependency_overrides[get_ai_settings] = lambda: Settings()

        response = TestClient(app).post(
            "/recommend",
            json=recommendation_payload(),
            headers={"Authorization": "Bearer wrong"},
        )

        self.assertEqual(response.status_code, 401)

    def test_valid_token_is_accepted_when_ai_token_is_configured(self) -> None:
        class Settings:
            ai_token = "secret"

        app.dependency_overrides[get_ai_settings] = lambda: Settings()

        response = TestClient(app).post(
            "/recommend",
            json=recommendation_payload(),
            headers={"Authorization": "Bearer secret"},
        )

        self.assertEqual(response.status_code, 200)

    def test_request_without_token_is_allowed_when_ai_token_is_not_configured(
        self,
    ) -> None:
        response = TestClient(app).post("/recommend", json=recommendation_payload())

        self.assertEqual(response.status_code, 200)

    def test_malformed_candidate_is_rejected_by_validation(self) -> None:
        payload = recommendation_payload()
        payload["candidates"] = [
            {
                "id": "course-invalid",
                "title": "Invalid",
                "category": "Backend",
                "description": "Missing popularity.",
            }
        ]

        response = TestClient(app).post("/recommend", json=payload)

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
