import unittest
from unittest.mock import AsyncMock, patch

import httpx

from app.core.config import AiSettings
from app.services.ninerouter import NineRouterSummaryError, generate_ninerouter_text


class _AsyncClientStub:
    def __init__(self, *, result=None, error=None) -> None:
        self.post = AsyncMock(return_value=result, side_effect=error)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


def _response(status_code: int = 200, payload=None, content: bytes | None = None):
    request = httpx.Request("POST", "http://router/v1/chat/completions")
    if content is not None:
        return httpx.Response(status_code, content=content, request=request)
    return httpx.Response(status_code, json=payload, request=request)


class NineRouterSummaryAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.settings = AiSettings(
            NINEROUTER_URL="http://router:20128",
            NINEROUTER_API_KEY="router-key",
            NINEROUTER_STT_MODEL="stt-route",
            NINEROUTER_SUMMARY_MODEL="free-chat-route",
            NINEROUTER_SUMMARY_TIMEOUT_SECONDS=23,
        )

    async def test_posts_openai_chat_request_and_parses_message_content(self) -> None:
        client = _AsyncClientStub(
            result=_response(
                payload={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "## Tổng quan\n\nTóm tắt thử nghiệm.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
        )

        with patch(
            "app.services.ninerouter.httpx.AsyncClient", return_value=client
        ) as client_factory:
            result = await generate_ninerouter_text(
                "Summarize the lesson", self.settings, max_output_tokens=2048
            )

        self.assertEqual(result, "## Tổng quan\n\nTóm tắt thử nghiệm.")
        client_factory.assert_called_once_with(timeout=23)
        client.post.assert_awaited_once_with(
            "http://router:20128/v1/chat/completions",
            headers={"Authorization": "Bearer router-key"},
            json={
                "model": "free-chat-route",
                "messages": [{"role": "user", "content": "Summarize the lesson"}],
                "stream": False,
                "temperature": 0.2,
                "max_tokens": 2048,
            },
        )

    async def test_omits_authorization_when_router_key_is_not_configured(self) -> None:
        client = _AsyncClientStub(
            result=_response(payload={"choices": [{"message": {"content": "A summary."}}]})
        )
        settings = AiSettings(
            NINEROUTER_URL="http://router",
            NINEROUTER_SUMMARY_MODEL="free-chat-route",
        )
        with patch("app.services.ninerouter.httpx.AsyncClient", return_value=client):
            result = await generate_ninerouter_text(
                "prompt", settings, max_output_tokens=512
            )

        self.assertEqual(result, "A summary.")
        self.assertEqual(
            client.post.await_args.kwargs["headers"],
            {},
        )

    async def test_rejects_missing_url_or_summary_model_before_request(self) -> None:
        settings_cases = (
            AiSettings(NINEROUTER_URL="", NINEROUTER_SUMMARY_MODEL="free-chat-route"),
            AiSettings(NINEROUTER_URL="http://router", NINEROUTER_SUMMARY_MODEL=""),
        )
        for index, settings in enumerate(settings_cases):
            with self.subTest(case=index), patch(
                "app.services.ninerouter.httpx.AsyncClient"
            ) as client_factory:
                with self.assertRaises(NineRouterSummaryError):
                    await generate_ninerouter_text(
                        "prompt", settings, max_output_tokens=512
                    )
            client_factory.assert_not_called()

    async def test_translates_http_and_timeout_failures(self) -> None:
        failures = (
            _response(status_code=503, payload={"error": {"message": "unavailable"}}),
            httpx.TimeoutException("router timed out"),
        )
        for failure in failures:
            with self.subTest(failure=failure):
                options = (
                    {"result": failure}
                    if isinstance(failure, httpx.Response)
                    else {"error": failure}
                )
                client = _AsyncClientStub(**options)
                with patch(
                    "app.services.ninerouter.httpx.AsyncClient", return_value=client
                ):
                    with self.assertRaises(NineRouterSummaryError):
                        await generate_ninerouter_text(
                            "prompt", self.settings, max_output_tokens=512
                        )

    async def test_rejects_malformed_or_empty_chat_completion_content(self) -> None:
        invalid_responses = (
            _response(content=b"not-json"),
            _response(payload={}),
            _response(payload={"choices": [{}]}),
            _response(payload={"choices": [{"message": {}}]}),
            _response(payload={"choices": [{"message": {"content": "  "}}]}),
        )
        for response in invalid_responses:
            with self.subTest(response=response):
                client = _AsyncClientStub(result=response)
                with patch(
                    "app.services.ninerouter.httpx.AsyncClient", return_value=client
                ):
                    with self.assertRaises(NineRouterSummaryError):
                        await generate_ninerouter_text(
                            "prompt", self.settings, max_output_tokens=512
                        )


if __name__ == "__main__":
    unittest.main()
