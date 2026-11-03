import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.core.config import AiSettings
from app.services.llm import build_chat_prompt, generate_ninerouter_chat_answer
from app.services.rag import RagCitation


class LlmRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_answer_sends_grounded_prompt_to_9router(self) -> None:
        settings = AiSettings(
            NINEROUTER_URL="http://router",
            NINEROUTER_CHAT_MODEL="free-chat-route",
        )
        history = [SimpleNamespace(role="user", content="Câu hỏi trước đó")]
        citation = RagCitation(
            lesson_id="lesson-1",
            lesson_title="Bài học",
            excerpt="Một trích dẫn.",
            source_type="transcript",
            locator="00:30",
            text="Nội dung bài học dùng làm context đầy đủ.",
        )

        with patch(
            "app.services.llm.generate_ninerouter_chat_text",
            new_callable=AsyncMock,
            return_value="Câu trả lời từ 9Router.",
        ) as generate:
            answer = await generate_ninerouter_chat_answer(
                "Giải thích nội dung.", history, [citation], settings
            )

        self.assertEqual(answer, "Câu trả lời từ 9Router.")
        prompt = generate.await_args.args[0]
        self.assertIn("Câu hỏi trước đó", prompt)
        self.assertIn("Nội dung bài học dùng làm context đầy đủ.", prompt)
        self.assertIn("CONTEXT", prompt)
        self.assertIs(generate.await_args.args[1], settings)
        self.assertEqual(generate.await_args.kwargs["max_output_tokens"], 512)

    async def test_chat_prompt_preserves_grounding_and_untrusted_context_rules(self) -> None:
        citation = RagCitation(
            lesson_id="lesson-1",
            lesson_title="Bài học",
            excerpt="Trích dẫn.",
            source_type="material",
            locator=None,
            text="Thông tin từ khóa học.",
        )

        prompt = build_chat_prompt("Câu hỏi", [], [citation])

        self.assertIn("Chi tra loi dua tren CONTEXT", prompt)
        self.assertIn("du lieu khong dang tin cay", prompt)
        self.assertIn("Thông tin từ khóa học.", prompt)
        self.assertIn("QUESTION:\nCâu hỏi", prompt)

    async def test_chat_answer_skips_9router_without_citations(self) -> None:
        with patch(
            "app.services.llm.generate_ninerouter_chat_text",
            new_callable=AsyncMock,
        ) as generate:
            answer = await generate_ninerouter_chat_answer(
                "Câu hỏi", [], [], AiSettings()
            )

        self.assertIsNone(answer)
        generate.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
