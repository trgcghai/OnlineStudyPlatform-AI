import unittest
from unittest.mock import AsyncMock, call, patch

import httpx

from app.core.config import AiSettings
from app.services.ninerouter import (
    NineRouterChatError,
    NineRouterSummaryError,
    generate_ninerouter_chat_text,
    generate_ninerouter_text,
)


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

    async def test_retries_429_and_server_errors_with_next_summary_model(self) -> None:
        for status_code in (429, 503):
            with self.subTest(status_code=status_code):
                settings = AiSettings(
                    NINEROUTER_URL="http://router",
                    NINEROUTER_SUMMARY_MODEL="summary-primary, summary-fallback",
                    NINEROUTER_SUMMARY_TIMEOUT_SECONDS=24,
                )
                first = _AsyncClientStub(
                    result=_response(
                        status_code=status_code,
                        payload={"error": {"message": "upstream detail"}},
                    )
                )
                second = _AsyncClientStub(
                    result=_response(
                        payload={"choices": [{"message": {"content": "Summary."}}]}
                    )
                )

                with patch(
                    "app.services.ninerouter.httpx.AsyncClient",
                    side_effect=[first, second],
                ) as client_factory:
                    result = await generate_ninerouter_text(
                        "prompt", settings, max_output_tokens=512
                    )

                self.assertEqual(result, "Summary.")
                self.assertEqual(
                    client_factory.call_args_list,
                    [call(timeout=12), call(timeout=12)],
                )
                self.assertEqual(
                    [
                        first.post.await_args.kwargs["json"]["model"],
                        second.post.await_args.kwargs["json"]["model"],
                    ],
                    ["summary-primary", "summary-fallback"],
                )

    async def test_retries_network_and_malformed_summary_responses(self) -> None:
        failures = (
            _AsyncClientStub(error=httpx.TimeoutException("router timed out")),
            _AsyncClientStub(error=httpx.ConnectError("router unavailable")),
            _AsyncClientStub(result=_response(content=b"not-json")),
            _AsyncClientStub(result=_response(payload={})),
            _AsyncClientStub(
                result=_response(
                    payload={"choices": [{"message": {"content": "  "}}]}
                )
            ),
        )
        for first in failures:
            with self.subTest(first=first):
                settings = AiSettings(
                    NINEROUTER_URL="http://router",
                    NINEROUTER_SUMMARY_MODEL="summary-primary,summary-fallback",
                )
                second = _AsyncClientStub(
                    result=_response(
                        payload={"choices": [{"message": {"content": "Summary."}}]}
                    )
                )

                with patch(
                    "app.services.ninerouter.httpx.AsyncClient",
                    side_effect=[first, second],
                ):
                    result = await generate_ninerouter_text(
                        "prompt", settings, max_output_tokens=512
                    )

                self.assertEqual(result, "Summary.")
                self.assertEqual(first.post.await_count, 1)
                self.assertEqual(second.post.await_count, 1)

    async def test_fails_fast_on_nonretryable_summary_http_status(self) -> None:
        for status_code in (302, 400):
            with self.subTest(status_code=status_code):
                settings = AiSettings(
                    NINEROUTER_URL="http://router",
                    NINEROUTER_SUMMARY_MODEL="summary-primary,summary-fallback",
                )
                first = _AsyncClientStub(
                    result=_response(
                        status_code=status_code,
                        payload={"error": {"message": "invalid request"}},
                    )
                )
                second = _AsyncClientStub(
                    result=_response(
                        payload={"choices": [{"message": {"content": "Summary."}}]}
                    )
                )

                with patch(
                    "app.services.ninerouter.httpx.AsyncClient",
                    side_effect=[first, second],
                ) as client_factory:
                    with self.assertRaises(NineRouterSummaryError) as raised:
                        await generate_ninerouter_text(
                            "prompt", settings, max_output_tokens=512
                        )

                self.assertEqual(client_factory.call_count, 1)
                self.assertNotIn("invalid request", str(raised.exception))

    async def test_summary_exhaustion_hides_upstream_error_body(self) -> None:
        settings = AiSettings(
            NINEROUTER_URL="http://router",
            NINEROUTER_SUMMARY_MODEL="summary-primary,summary-fallback",
        )
        clients = [
            _AsyncClientStub(
                result=_response(
                    status_code=503,
                    payload={"error": {"message": "sensitive upstream detail"}},
                )
            )
            for _ in range(2)
        ]

        with patch(
            "app.services.ninerouter.httpx.AsyncClient", side_effect=clients
        ):
            with self.assertRaises(NineRouterSummaryError) as raised:
                await generate_ninerouter_text(
                    "prompt", settings, max_output_tokens=512
                )

        self.assertNotIn("sensitive upstream detail", str(raised.exception))

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


class NineRouterChatAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.settings = AiSettings(
            NINEROUTER_URL="http://router:20128/",
            NINEROUTER_API_KEY="router-key",
            NINEROUTER_CHAT_MODEL="free-chat-route",
            NINEROUTER_CHAT_TIMEOUT_SECONDS=23,
        )

    async def test_posts_chat_request_and_parses_assistant_content(self) -> None:
        client = _AsyncClientStub(
            result=_response(payload={"choices": [{"message": {"content": "  Answer.  "}}]})
        )
        with patch(
            "app.services.ninerouter.httpx.AsyncClient", return_value=client
        ) as client_factory:
            result = await generate_ninerouter_chat_text("prompt", self.settings)

        self.assertEqual(result, "Answer.")
        client_factory.assert_called_once_with(timeout=23)
        client.post.assert_awaited_once_with(
            "http://router:20128/v1/chat/completions",
            headers={"Authorization": "Bearer router-key"},
            json={
                "model": "free-chat-route",
                "messages": [{"role": "user", "content": "prompt"}],
                "stream": False,
                "temperature": 0.2,
                "max_tokens": 512,
            },
        )

    async def test_omits_authorization_without_optional_key(self) -> None:
        client = _AsyncClientStub(
            result=_response(payload={"choices": [{"message": {"content": "Answer."}}]})
        )
        settings = AiSettings(
            NINEROUTER_URL="http://router",
            NINEROUTER_CHAT_MODEL="free-chat-route",
        )
        with patch("app.services.ninerouter.httpx.AsyncClient", return_value=client):
            result = await generate_ninerouter_chat_text("prompt", settings)

        self.assertEqual(result, "Answer.")
        self.assertEqual(client.post.await_args.kwargs["headers"], {})

    async def test_rejects_missing_url_or_chat_model_before_request(self) -> None:
        cases = (
            AiSettings(NINEROUTER_URL="", NINEROUTER_CHAT_MODEL="free-chat-route"),
            AiSettings(NINEROUTER_URL="http://router", NINEROUTER_CHAT_MODEL=""),
        )
        for settings in cases:
            with self.subTest(settings=settings), patch(
                "app.services.ninerouter.httpx.AsyncClient"
            ) as client_factory:
                with self.assertRaises(NineRouterChatError):
                    await generate_ninerouter_chat_text("prompt", settings)
            client_factory.assert_not_called()

    async def test_translates_http_and_timeout_failures(self) -> None:
        failures = (
            _response(status_code=503, payload={"error": {"message": "unavailable"}}),
            httpx.TimeoutException("router timed out"),
        )
        for failure in failures:
            options = (
                {"result": failure}
                if isinstance(failure, httpx.Response)
                else {"error": failure}
            )
            with self.subTest(failure=failure):
                client = _AsyncClientStub(**options)
                with patch(
                    "app.services.ninerouter.httpx.AsyncClient", return_value=client
                ):
                    with self.assertRaises(NineRouterChatError):
                        await generate_ninerouter_chat_text("prompt", self.settings)

    async def test_retries_chat_with_next_configured_model(self) -> None:
        settings = AiSettings(
            NINEROUTER_URL="http://router",
            NINEROUTER_CHAT_MODEL="chat-primary, chat-fallback",
            NINEROUTER_CHAT_TIMEOUT_SECONDS=24,
        )
        first = _AsyncClientStub(
            result=_response(status_code=503, payload={"error": "unavailable"})
        )
        second = _AsyncClientStub(
            result=_response(payload={"choices": [{"message": {"content": "Answer."}}]})
        )

        with patch(
            "app.services.ninerouter.httpx.AsyncClient",
            side_effect=[first, second],
        ) as client_factory:
            result = await generate_ninerouter_chat_text("prompt", settings)

        self.assertEqual(result, "Answer.")
        self.assertEqual(
            client_factory.call_args_list,
            [call(timeout=12), call(timeout=12)],
        )
        self.assertEqual(
            [
                first.post.await_args.kwargs["json"]["model"],
                second.post.await_args.kwargs["json"]["model"],
            ],
            ["chat-primary", "chat-fallback"],
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
                    with self.assertRaises(NineRouterChatError):
                        await generate_ninerouter_chat_text("prompt", self.settings)

if __name__ == "__main__":
    unittest.main()
