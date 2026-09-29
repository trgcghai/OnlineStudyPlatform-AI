import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.core.config import AiSettings, get_ai_settings
from app.main import app


class ChatTests(unittest.TestCase):
    def setUp(self) -> None:
        app.dependency_overrides[get_ai_settings] = lambda: AiSettings(
            AI_TOKEN=None,
            LLM_PROVIDER="local",
            GEMINI_API_KEY=None,
        )

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_chat_returns_grounded_answer_with_citation(self) -> None:
        response = TestClient(app).post(
            "/chat",
            json={
                "learnerId": "learner-1",
                "courseId": "course-1",
                "question": "NestJS dependency injection la gi?",
                "similarityThreshold": 0.1,
                "history": [],
                "sources": [
                    {
                        "lessonId": "lesson-1",
                        "lessonTitle": "Dependency Injection",
                        "content": "NestJS dependency injection giup quan ly provider va service trong module.",
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["inScope"])
        self.assertEqual(body["citations"][0]["lessonId"], "lesson-1")

    def test_chat_refuses_low_similarity_question(self) -> None:
        response = TestClient(app).post(
            "/chat",
            json={
                "learnerId": "learner-1",
                "courseId": "course-1",
                "question": "Thoi tiet hom nay the nao?",
                "similarityThreshold": 0.8,
                "history": [],
                "sources": [
                    {
                        "lessonId": "lesson-1",
                        "lessonTitle": "Dependency Injection",
                        "content": "NestJS dependency injection giup quan ly provider va service trong module.",
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["inScope"])
        self.assertIn("ngoai noi dung kien thuc cua khoa hoc", body["answer"])

    def test_chat_retrieves_matching_chunk_from_long_lesson(self) -> None:
        unrelated_intro = " ".join(["mo dau tong quan khoa hoc"] * 120)
        target_content = (
            "Guard trong NestJS kiem tra request truoc khi vao controller "
            "va thuong dung de xac thuc phan quyen."
        )
        response = TestClient(app).post(
            "/chat",
            json={
                "learnerId": "learner-1",
                "courseId": "course-1",
                "question": "Guard trong NestJS dung de lam gi?",
                "similarityThreshold": 0.05,
                "history": [],
                "sources": [
                    {
                        "lessonId": "lesson-guard",
                        "lessonTitle": "Guard",
                        "content": f"{unrelated_intro} {target_content}",
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["inScope"])
        self.assertEqual(body["citations"][0]["lessonId"], "lesson-guard")
        self.assertIn("Guard trong NestJS", body["citations"][0]["excerpt"])

    def test_chat_returns_multiple_ranked_citations(self) -> None:
        response = TestClient(app).post(
            "/chat",
            json={
                "learnerId": "learner-1",
                "courseId": "course-1",
                "question": "module provider service trong NestJS lien quan the nao?",
                "similarityThreshold": 0.05,
                "history": [],
                "sources": [
                    {
                        "lessonId": "lesson-module",
                        "lessonTitle": "Module",
                        "content": "NestJS module gom provider va controller de to chuc service.",
                    },
                    {
                        "lessonId": "lesson-provider",
                        "lessonTitle": "Provider",
                        "content": "Provider trong NestJS thuong la service duoc inject vao module.",
                    },
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["inScope"])
        self.assertGreaterEqual(len(body["citations"]), 2)

    def test_chat_uses_history_for_follow_up_question(self) -> None:
        response = TestClient(app).post(
            "/chat",
            json={
                "learnerId": "learner-1",
                "courseId": "course-1",
                "question": "no dung de lam gi?",
                "similarityThreshold": 0.05,
                "history": [
                    {
                        "role": "user",
                        "content": "Hay giai thich dependency injection trong NestJS",
                    },
                    {
                        "role": "assistant",
                        "content": "Dependency injection giup quan ly provider.",
                    },
                ],
                "sources": [
                    {
                        "lessonId": "lesson-di",
                        "lessonTitle": "Dependency Injection",
                        "content": "Dependency injection trong NestJS giup quan ly provider va service.",
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["inScope"])
        self.assertIn("ngu canh trao doi truoc", body["answer"])

    def test_chat_refuses_when_sources_are_empty(self) -> None:
        response = TestClient(app).post(
            "/chat",
            json={
                "learnerId": "learner-1",
                "courseId": "course-1",
                "question": "Dependency injection la gi?",
                "similarityThreshold": 0.05,
                "history": [],
                "sources": [],
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["inScope"])
        self.assertEqual(body["citations"], [])

    def test_chat_uses_gemini_answer_when_available(self) -> None:
        with patch(
            "app.api.chat.generate_gemini_answer",
            new_callable=AsyncMock,
        ) as generate_answer:
            generate_answer.return_value = "Gemini: Dependency injection giup inject service vao provider."
            response = TestClient(app).post(
                "/chat",
                json={
                    "learnerId": "learner-1",
                    "courseId": "course-1",
                    "question": "Dependency injection la gi?",
                    "similarityThreshold": 0.05,
                    "history": [],
                    "sources": [
                        {
                            "lessonId": "lesson-di",
                            "lessonTitle": "Dependency Injection",
                            "content": "Dependency injection trong NestJS giup quan ly provider va service.",
                        }
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            body["answer"],
            "Gemini: Dependency injection giup inject service vao provider.",
        )
        generate_answer.assert_awaited_once()

    def test_chat_does_not_call_gemini_when_question_is_out_of_scope(self) -> None:
        with patch(
            "app.api.chat.generate_gemini_answer",
            new_callable=AsyncMock,
        ) as generate_answer:
            response = TestClient(app).post(
                "/chat",
                json={
                    "learnerId": "learner-1",
                    "courseId": "course-1",
                    "question": "Thoi tiet hom nay the nao?",
                    "similarityThreshold": 0.8,
                    "history": [],
                    "sources": [
                        {
                            "lessonId": "lesson-di",
                            "lessonTitle": "Dependency Injection",
                            "content": "Dependency injection trong NestJS giup quan ly provider va service.",
                        }
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["inScope"])
        generate_answer.assert_not_awaited()

    def test_chat_falls_back_to_rag_answer_when_gemini_returns_none(self) -> None:
        with patch(
            "app.api.chat.generate_gemini_answer",
            new_callable=AsyncMock,
        ) as generate_answer:
            generate_answer.return_value = None
            response = TestClient(app).post(
                "/chat",
                json={
                    "learnerId": "learner-1",
                    "courseId": "course-1",
                    "question": "Dependency injection la gi?",
                    "similarityThreshold": 0.05,
                    "history": [],
                    "sources": [
                        {
                            "lessonId": "lesson-di",
                            "lessonTitle": "Dependency Injection",
                            "content": "Dependency injection trong NestJS giup quan ly provider va service.",
                        }
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["inScope"])
        self.assertIn("Dua tren tai lieu khoa hoc", body["answer"])
        generate_answer.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
