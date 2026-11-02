import unittest
from unittest.mock import AsyncMock, patch

from app.core.config import AiSettings
from app.services.llm import generate_gemini_answer
from app.services.rag import RagCitation


class LlmRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_rag_answer_still_uses_gemini_with_summary_router_configured(self) -> None:
        settings = AiSettings(
            LLM_PROVIDER="gemini",
            GEMINI_API_KEY="test-gemini-key",
            NINEROUTER_SUMMARY_MODEL="free-summary-route",
        )
        citation = RagCitation(
            lesson_id="lesson-1",
            lesson_title="Bài học",
            excerpt="Một trích dẫn.",
            source_type="transcript",
            locator=None,
            text="Nội dung bài học dùng làm context.",
        )

        with patch(
            "app.services.llm.generate_gemini_text",
            new_callable=AsyncMock,
            return_value="Câu trả lời từ Gemini.",
        ) as generate_gemini:
            answer = await generate_gemini_answer(
                "Giải thích nội dung.", [], [citation], settings
            )

        self.assertEqual(answer, "Câu trả lời từ Gemini.")
        generate_gemini.assert_awaited_once()
        self.assertIn("Nội dung bài học dùng làm context.", generate_gemini.await_args.args[0])


if __name__ == "__main__":
    unittest.main()
