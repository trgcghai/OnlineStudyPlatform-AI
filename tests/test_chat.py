import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.core.config import AiSettings, get_ai_settings
from app.main import app
from app.services.llm import build_chat_prompt
from app.services.rag import build_citations
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
        )

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_chat_searches_only_the_requested_course_and_returns_citation(self) -> None:
        with (
            patch(
                "app.api.chat.search_course",
                new_callable=AsyncMock,
                return_value=[match()],
            ) as search,
            patch(
                "app.api.chat.generate_ninerouter_chat_answer",
                new_callable=AsyncMock,
                return_value=None,
            ),
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
        body = response.json()
        self.assertTrue(body["inScope"])
        self.assertEqual(body["citations"][0]["lessonId"], "lesson-1")
        self.assertEqual(body["citations"][0]["sourceType"], "material")
        self.assertEqual(body["citations"][0]["locator"], "Trang 3")
        self.assertEqual(search.await_args.args[1], "course-1")

    def test_topic_change_does_not_reuse_old_history(self) -> None:
        with (
            patch(
                "app.api.chat.search_course",
                new_callable=AsyncMock,
                return_value=[match()],
            ) as search,
            patch(
                "app.api.chat.generate_ninerouter_chat_answer",
                new_callable=AsyncMock,
                return_value="Hướng dẫn tách nền.",
            ) as generate,
        ):
            response = TestClient(app).post(
                "/chat",
                json={
                    "learnerId": "learner-1",
                    "courseId": "course-1",
                    "question": "cách tách phông nền bằng capcut",
                    "similarityThreshold": 0.2,
                    "history": [
                        {"role": "user", "content": "kết hợp như nào"},
                        {"role": "assistant", "content": "Dùng ChatGPT và Canva."},
                        {
                            "role": "user",
                            "content": "sử dụng ChatGPT kết hợp với Canva để tạo subtitle và nội dung video.",
                        },
                        {"role": "assistant", "content": "Training ChatGPT trước."},
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(search.await_args.args[2], "cách tách phông nền bằng capcut")
        self.assertEqual(generate.await_args.kwargs["history"], [])

    def test_follow_up_uses_only_the_latest_topic(self) -> None:
        with (
            patch(
                "app.api.chat.search_course",
                new_callable=AsyncMock,
                return_value=[match()],
            ) as search,
            patch(
                "app.api.chat.generate_ninerouter_chat_answer",
                new_callable=AsyncMock,
                return_value="Chọn màu cần loại bỏ.",
            ) as generate,
        ):
            response = TestClient(app).post(
                "/chat",
                json={
                    "learnerId": "learner-1",
                    "courseId": "course-1",
                    "question": "Nó dùng để làm gì?",
                    "similarityThreshold": 0.2,
                    "history": [
                        {"role": "user", "content": "Canva là gì?"},
                        {"role": "assistant", "content": "Công cụ thiết kế."},
                        {"role": "user", "content": "Chroma key trong CapCut là gì?"},
                        {"role": "assistant", "content": "Dùng để loại bỏ màu nền."},
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        query = search.await_args.args[2]
        self.assertIn("Chroma key trong CapCut", query)
        self.assertIn("Nó dùng để làm gì?", query)
        self.assertNotIn("Canva", query)
        history = generate.await_args.kwargs["history"]
        self.assertEqual(len(history), 2)
        self.assertIn("Chroma key", history[0].content)

    def test_follow_up_keeps_two_recent_turns_and_supports_unaccented_text(self) -> None:
        with (
            patch(
                "app.api.chat.search_course",
                new_callable=AsyncMock,
                return_value=[match()],
            ) as search,
            patch(
                "app.api.chat.generate_ninerouter_chat_answer",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            response = TestClient(app).post(
                "/chat",
                json={
                    "learnerId": "learner-1",
                    "courseId": "course-1",
                    "question": "giai thich them",
                    "similarityThreshold": 0.2,
                    "history": [
                        {"role": "user", "content": "Canva là gì?"},
                        {"role": "assistant", "content": "Công cụ thiết kế."},
                        {"role": "user", "content": "Chroma key trong CapCut là gì?"},
                        {"role": "assistant", "content": "Dùng để loại bỏ màu nền."},
                        {"role": "user", "content": "Nó dùng thế nào?"},
                        {"role": "assistant", "content": "Chọn màu nền."},
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        query = search.await_args.args[2]
        self.assertIn("Chroma key trong CapCut", query)
        self.assertIn("Nó dùng thế nào?", query)
        self.assertIn("giai thich them", query)
        self.assertNotIn("Canva", query)

    def test_ninerouter_uses_full_chunk_while_citation_keeps_short_excerpt(self) -> None:
        answer_at_end = "Đáp án chỉ xuất hiện ở cuối chunk."
        chunk_text = f"{'Nội dung mở đầu. ' * 30}{answer_at_end}"

        with (
            patch(
                "app.api.chat.search_course",
                new_callable=AsyncMock,
                return_value=[match(text=chunk_text)],
            ),
            patch(
                "app.api.chat.generate_ninerouter_chat_answer",
                new_callable=AsyncMock,
                return_value="Đã tìm thấy đáp án.",
            ) as generate,
        ):
            response = TestClient(app).post(
                "/chat",
                json={
                    "learnerId": "learner-1",
                    "courseId": "course-1",
                    "question": "Đáp án ở đâu?",
                    "similarityThreshold": 0.2,
                    "history": [],
                },
            )

        citation = generate.await_args.kwargs["citations"][0]
        prompt = build_chat_prompt("Đáp án ở đâu?", [], [citation])
        self.assertIn(answer_at_end, prompt)
        self.assertNotIn(answer_at_end, response.json()["citations"][0]["excerpt"])
        self.assertLessEqual(len(response.json()["citations"][0]["excerpt"]), 360)
        self.assertNotIn("text", response.json()["citations"][0])

    def test_full_chunks_with_identical_excerpt_are_not_dropped(self) -> None:
        common_start = "Thông tin chung. " * 30
        citations = build_citations(
            [
                match(text=common_start + "Bước 4: Điều chỉnh Intensity."),
                match(text=common_start + "Bước 5: Điều chỉnh Shadow."),
                match(text=common_start + "Bước 5: Điều chỉnh Shadow."),
            ]
        )

        self.assertEqual(len(citations), 2)
        prompt = build_chat_prompt("Các bước tách nền?", [], citations)
        self.assertIn("Bước 4: Điều chỉnh Intensity.", prompt)
        self.assertIn("Bước 5: Điều chỉnh Shadow.", prompt)

    def test_context_budget_keeps_complete_chunks_in_score_order(self) -> None:
        chunks = [f"Chunk {index} " + "x" * 3992 for index in range(4)]
        citations = build_citations([match(text=text) for text in chunks])

        self.assertEqual([citation.text for citation in citations], chunks[:3])
        self.assertLessEqual(sum(len(citation.text) for citation in citations), 12_000)

    def test_chat_keeps_distinct_chunks_with_the_same_source_locator(self) -> None:
        with (
            patch(
                "app.api.chat.search_course",
                new_callable=AsyncMock,
                return_value=[
                    match(text="Đoạn transcript nói về provider."),
                    match(text="Đoạn transcript nói về service."),
                ],
            ),
            patch(
                "app.api.chat.generate_ninerouter_chat_answer",
                new_callable=AsyncMock,
                return_value=None,
            ),
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

    def test_chat_refuses_without_relevant_vectors_and_skips_ninerouter(self) -> None:
        with (
            patch(
                "app.api.chat.search_course",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.api.chat.generate_ninerouter_chat_answer",
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

    def test_chat_uses_ninerouter_for_a_grounded_answer(self) -> None:
        with (
            patch(
                "app.api.chat.search_course",
                new_callable=AsyncMock,
                return_value=[match()],
            ),
            patch(
                "app.api.chat.generate_ninerouter_chat_answer",
                new_callable=AsyncMock,
                return_value="Câu trả lời tự nhiên từ 9Router.",
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
        self.assertEqual(response.json()["answer"], "Câu trả lời tự nhiên từ 9Router.")
        generate.assert_awaited_once()

    def test_chat_returns_server_error_when_9router_generation_fails(self) -> None:
        with (
            patch(
                "app.api.chat.search_course",
                new_callable=AsyncMock,
                return_value=[match()],
            ),
            patch(
                "app.api.chat.generate_ninerouter_chat_answer",
                new_callable=AsyncMock,
                side_effect=RuntimeError("9Router unavailable"),
            ) as generate,
        ):
            response = TestClient(app, raise_server_exceptions=False).post(
                "/chat",
                json={
                    "learnerId": "learner-1",
                    "courseId": "course-1",
                    "question": "Dependency injection là gì?",
                    "similarityThreshold": 0.2,
                    "history": [],
                },
            )

        self.assertEqual(response.status_code, 500)
        generate.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
