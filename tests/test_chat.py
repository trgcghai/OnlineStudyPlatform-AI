import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.core.config import AiSettings, get_ai_settings
from app.main import app
from app.services.vector_store import VectorSearchResult


def match(
    score: float = 0.82,
    text: str = "Dependency injection giúp quản lý provider và service trong module.",
) -> VectorSearchResult:
    return VectorSearchResult(
        score=score,
        lesson_id="lesson-1",
        lesson_title="Dependency Injection",
        source_type="material",
        locator="Trang 3",
        text=text,
    )


class ChatTests(unittest.TestCase):
    def setUp(self) -> None:
        app.dependency_overrides[get_ai_settings] = lambda: AiSettings(
            AI_TOKEN=None,
            LLM_PROVIDER="local",
            GEMINI_API_KEY=None,
        )

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_chat_searches_only_the_requested_course_and_returns_citation(self) -> None:
        with patch(
            "app.api.chat.search_course",
            new_callable=AsyncMock,
            return_value=[match()],
        ) as search:
            response = TestClient(app).post(
                "/chat",
                json={
                    "learnerId": "learner-1",
                    "courseId": "course-1",
                    "question": "Dependency injection là gì?",
                    "similarityThreshold": 0.2,
                    "history": [],
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["inScope"])
        self.assertEqual(body["citations"][0]["lessonId"], "lesson-1")
        self.assertEqual(body["citations"][0]["sourceType"], "material")
        self.assertEqual(body["citations"][0]["locator"], "Trang 3")
        self.assertEqual(search.await_args.args[1], "course-1")

    def test_chat_uses_recent_user_history_in_vector_query(self) -> None:
        with patch(
            "app.api.chat.search_course",
            new_callable=AsyncMock,
            return_value=[match()],
        ) as search:
            response = TestClient(app).post(
                "/chat",
                json={
                    "learnerId": "learner-1",
                    "courseId": "course-1",
                    "question": "Nó dùng để làm gì?",
                    "similarityThreshold": 0.2,
                    "history": [
                        {"role": "user", "content": "Dependency injection là gì?"},
                        {"role": "assistant", "content": "Đây là một cơ chế."},
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Dependency injection", search.await_args.args[2])
        self.assertIn("Nó dùng để làm gì", search.await_args.args[2])

    def test_chat_keeps_distinct_chunks_with_the_same_source_locator(self) -> None:
        with patch(
            "app.api.chat.search_course",
            new_callable=AsyncMock,
            return_value=[
                match(text="Đoạn transcript nói về provider."),
                match(text="Đoạn transcript nói về service."),
            ],
        ):
            response = TestClient(app).post(
                "/chat",
                json={
                    "learnerId": "learner-1",
                    "courseId": "course-1",
                    "question": "Provider và service liên quan thế nào?",
                    "similarityThreshold": 0.2,
                    "history": [],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["citations"]), 2)

    def test_chat_refuses_without_relevant_vectors_and_skips_gemini(self) -> None:
        with (
            patch(
                "app.api.chat.search_course",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.api.chat.generate_gemini_answer",
                new_callable=AsyncMock,
            ) as generate,
        ):
            response = TestClient(app).post(
                "/chat",
                json={
                    "learnerId": "learner-1",
                    "courseId": "course-1",
                    "question": "Thời tiết hôm nay thế nào?",
                    "similarityThreshold": 0.5,
                    "history": [],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["inScope"])
        self.assertIn("ngoài nội dung", response.json()["answer"])
        generate.assert_not_awaited()

    def test_chat_uses_gemini_for_a_grounded_answer(self) -> None:
        with (
            patch(
                "app.api.chat.search_course",
                new_callable=AsyncMock,
                return_value=[match()],
            ),
            patch(
                "app.api.chat.generate_gemini_answer",
                new_callable=AsyncMock,
                return_value="Câu trả lời tự nhiên từ Gemini.",
            ) as generate,
        ):
            response = TestClient(app).post(
                "/chat",
                json={
                    "learnerId": "learner-1",
                    "courseId": "course-1",
                    "question": "Dependency injection là gì?",
                    "similarityThreshold": 0.2,
                    "history": [],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Câu trả lời tự nhiên từ Gemini.")
        generate.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
