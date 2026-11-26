import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

from app.core.config import AiSettings, get_ai_settings
from app.main import app
from app.services.ninerouter import NineRouterSummaryError
from app.services.summarization import (
    SummarizationError,
    SummaryResult,
    build_final_summary_prompt,
    build_notes_prompt,
    chunk_transcript,
    summarize_transcript,
    validate_summary,
)
from app.services.transcription import (
    TranscriptionError,
    TranscriptionResult,
    download_media,
    transcribe_audio,
    transcribe_file,
    validate_media_url,
)


VALID_SUMMARY = """## Tổng quan

Bài học giới thiệu Dependency Injection và vai trò của cơ chế này trong ứng dụng.

## Nội dung chính

- Dependency Injection cung cấp các phụ thuộc cho lớp thông qua một cơ chế quản lý tập trung.
- Constructor injection giúp mã nguồn rõ ràng, dễ kiểm thử và bảo trì hơn.

## Kết luận và ghi nhớ

Người học cần ghi nhớ cách đăng ký và sử dụng dependency đúng phạm vi."""


class SummaryApiTests(unittest.TestCase):
    def setUp(self) -> None:
        app.dependency_overrides[get_ai_settings] = lambda: AiSettings(
            AI_TOKEN=None
        )

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_transcribe_returns_whisper_result_for_audio_url(self) -> None:
        with patch(
            "app.api.summary.transcribe_audio",
            new_callable=AsyncMock,
        ) as transcribe_audio_mock:
            transcribe_audio_mock.return_value = TranscriptionResult(
                transcript="Nội dung bài giảng.",
                language="vi",
                duration_seconds=42.5,
            )
            response = TestClient(app).post(
                "/transcribe",
                json={
                    "lessonId": "lesson-1",
                    "audioUrl": "http://minio/audio.ogg",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "transcript": "Nội dung bài giảng.",
                "language": "vi",
                "durationSeconds": 42.5,
            },
        )

    def test_transcribe_rejects_legacy_video_url_field(self) -> None:
        response = TestClient(app).post(
            "/transcribe",
            json={"lessonId": "lesson-1", "videoUrl": "http://minio/video.mp4"},
        )

        self.assertEqual(response.status_code, 422)

    def test_summarize_returns_provider_and_text(self) -> None:
        with patch(
            "app.api.summary.summarize_transcript",
            new_callable=AsyncMock,
        ) as summarize_transcript_mock:
            summarize_transcript_mock.return_value = SummaryResult(
                summary=VALID_SUMMARY,
                provider="9router",
            )
            response = TestClient(app).post(
                "/summarize",
                json={
                    "lessonId": "lesson-1",
                    "title": "Bài học",
                    "transcript": "Nội dung bài giảng đủ dài.",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["provider"], "9router")

    def test_summary_endpoints_reject_invalid_token(self) -> None:
        app.dependency_overrides[get_ai_settings] = lambda: AiSettings(
            AI_TOKEN="secret"
        )

        response = TestClient(app).post(
            "/summarize",
            json={
                "lessonId": "lesson-1",
                "title": "Bài học",
                "transcript": "Nội dung bài giảng đủ dài.",
            },
        )

        self.assertEqual(response.status_code, 401)

class SummarizationTests(unittest.IsolatedAsyncioTestCase):
    def test_final_prompt_requires_natural_complete_vietnamese_structure(self) -> None:
        prompt = build_final_summary_prompt(
            "Dependency Injection",
            "Xin chào, hôm nay chúng ta học dependency injection.",
            "NGUON_BAI_GIANG",
        )

        self.assertIn("không chép tuần tự từng câu", prompt)
        self.assertIn("câu đầy đủ", prompt)
        self.assertIn("## Tổng quan", prompt)
        self.assertIn("## Nội dung chính", prompt)
        self.assertIn("## Kết luận và ghi nhớ", prompt)

    def test_chunk_transcript_splits_unpunctuated_transcript_text(self) -> None:
        transcript = "nội dung bài giảng " * 80
        chunks = chunk_transcript(transcript, 500)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 500 for chunk in chunks))

    async def test_short_transcript_returns_only_validated_9router_output(self) -> None:
        settings = AiSettings(
            NINEROUTER_URL="http://router",
            NINEROUTER_SUMMARY_MODEL="free-chat-route",
        )
        with patch(
            "app.services.summarization.generate_ninerouter_text",
            new_callable=AsyncMock,
            return_value=VALID_SUMMARY,
            create=True,
        ):
            result = await summarize_transcript(
                "Dependency Injection",
                "Dependency Injection cung cấp dependency cho lớp.",
                settings,
            )

        self.assertEqual(result.provider, "9router")
        self.assertEqual(result.summary, VALID_SUMMARY)

    async def test_final_summary_passes_structure_validator_to_9router(self) -> None:
        settings = AiSettings(
            NINEROUTER_URL="http://router",
            NINEROUTER_SUMMARY_MODEL="summary-primary,summary-fallback",
        )
        with patch(
            "app.services.summarization.generate_ninerouter_text",
            new_callable=AsyncMock,
            return_value=VALID_SUMMARY,
        ) as generate:
            await summarize_transcript(
                "Dependency Injection",
                "Dependency Injection cung cấp dependency cho lớp.",
                settings,
            )

        validator = generate.await_args.kwargs.get("validator")
        self.assertTrue(callable(validator))
        self.assertEqual(
            validator(VALID_SUMMARY),
            VALID_SUMMARY,
        )
        with self.assertRaises(NineRouterSummaryError):
            validator("Invalid summary structure.")

    async def test_9router_failure_is_reported_for_retry(self) -> None:
        settings = AiSettings(
            NINEROUTER_URL="http://router",
            NINEROUTER_SUMMARY_MODEL="free-chat-route",
        )
        with patch(
            "app.services.summarization.generate_ninerouter_text",
            new_callable=AsyncMock,
            side_effect=NineRouterSummaryError("request failed"),
            create=True,
        ):
            with self.assertRaisesRegex(SummarizationError, "could not generate"):
                await summarize_transcript(
                    "Dependency Injection",
                    "Dependency Injection cung cấp dependency cho lớp.",
                    settings,
                )

    async def test_invalid_9router_structure_is_rejected(self) -> None:
        settings = AiSettings(
            NINEROUTER_URL="http://router",
            NINEROUTER_SUMMARY_MODEL="free-chat-route",
        )

        async def generate_invalid_summary(
            _prompt,
            _settings,
            *,
            validator=None,
            **_kwargs,
        ):
            content = "Dependency Injection cung cấp dependency cho lớp."
            return validator(content) if validator else content

        with patch(
            "app.services.summarization.generate_ninerouter_text",
            side_effect=generate_invalid_summary,
            create=True,
        ):
            with self.assertRaisesRegex(SummarizationError, "could not generate"):
                await summarize_transcript(
                    "Dependency Injection",
                    "Dependency Injection cung cấp dependency cho lớp.",
                    settings,
                )

    def test_transcript_copy_is_rejected_even_when_it_has_headings(self) -> None:
        with self.assertRaisesRegex(
            SummarizationError, "Model returned the transcript instead of a summary"
        ):
            validate_summary(VALID_SUMMARY, VALID_SUMMARY)

    def test_summary_rejects_out_of_order_headings(self) -> None:
        out_of_order = """## Tổng quan

Chủ đề bài học.

## Kết luận và ghi nhớ

Điều cần ghi nhớ.

## Nội dung chính

- Một ý chính.
"""
        with self.assertRaisesRegex(
            SummarizationError, "Summary is missing the required structure"
        ):
            validate_summary(out_of_order, "Nguồn bài học.")

    def test_summary_rejects_empty_section(self) -> None:
        empty_section = VALID_SUMMARY.replace(
            "Bài học giới thiệu Dependency Injection và vai trò của cơ chế này trong ứng dụng.",
            "",
        )
        with self.assertRaisesRegex(SummarizationError, "Summary contains an empty section"):
            validate_summary(empty_section, "Nguồn bài học.")

    def test_summary_requires_main_content_bullets(self) -> None:
        no_bullets = VALID_SUMMARY.replace("- Dependency Injection", "Dependency Injection").replace(
            "- Constructor injection", "Constructor injection"
        )
        with self.assertRaisesRegex(
            SummarizationError, "Summary has no main-content bullet points"
        ):
            validate_summary(no_bullets, "Nguồn bài học.")

    async def test_empty_transcript_is_rejected_before_router_call(self) -> None:
        with patch(
            "app.services.summarization.generate_ninerouter_text",
            new_callable=AsyncMock,
            create=True,
        ) as generate:
            with self.assertRaisesRegex(SummarizationError, "Transcript is empty"):
                await summarize_transcript("Bài học", " \n ", AiSettings())
        generate.assert_not_awaited()

    def test_valid_summary_normalizes_common_markdown_bullet_styles(self) -> None:
        star_bullets = VALID_SUMMARY.replace("- Dependency", "* Dependency").replace(
            "- Constructor", "2. Constructor"
        )

        result = validate_summary(star_bullets, "Transcript nguồn khác.")

        self.assertIn("- Dependency", result)
        self.assertIn("- Constructor", result)

    async def test_long_transcript_uses_notes_then_final_9router_call(self) -> None:
        settings = AiSettings(
            NINEROUTER_URL="http://router",
            NINEROUTER_SUMMARY_MODEL="free-chat-route",
            SUMMARY_CHUNK_SIZE=500,
        )
        transcript = " ".join(
            [
                "Phần một giải thích khái niệm quan trọng trong bài học.",
                "Phần hai đưa ra một ví dụ minh họa cho khái niệm.",
                "Phần ba kết luận nội dung cần ghi nhớ sau bài học.",
            ]
            * 12
        )
        chunks = chunk_transcript(transcript, settings.summary_chunk_size)
        with patch(
            "app.services.summarization.generate_ninerouter_text",
            new_callable=AsyncMock,
            create=True,
        ) as generate:
            generate.side_effect = [
                *["- Ghi chú sự kiện quan trọng."] * len(chunks),
                VALID_SUMMARY,
            ]
            result = await summarize_transcript("Bài học", transcript, settings)

        self.assertEqual(result.provider, "9router")
        self.assertEqual(generate.await_count, len(chunks) + 1)
        self.assertEqual(
            [call.args[0] for call in generate.await_args_list[:-1]],
            [
                build_notes_prompt("Bài học", chunk, index, len(chunks))
                for index, chunk in enumerate(chunks, start=1)
            ],
        )
        self.assertEqual(
            generate.await_args_list[-1].args[0],
            build_final_summary_prompt(
                "Bài học",
                "\n\n".join(["- Ghi chú sự kiện quan trọng."] * len(chunks)),
                "GHI_CHU_DA_TRICH_XUAT",
            ),
        )
        self.assertTrue(
            all(
                call.kwargs["max_output_tokens"] == settings.summary_max_output_tokens
                for call in generate.await_args_list
            )
        )


class _FakeDownloadResponse:
    headers: dict[str, str] = {}

    def __init__(self, url: str, chunks: list[bytes]) -> None:
        self.url = url
        self.chunks = iter(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def geturl(self) -> str:
        return self.url

    def read(self, _size: int) -> bytes:
        return next(self.chunks, b"")


class TranscriptionTests(unittest.IsolatedAsyncioTestCase):
    def test_rejects_media_from_an_unapproved_host(self) -> None:
        with self.assertRaisesRegex(TranscriptionError, "not allowed"):
            validate_media_url("https://example.com/audio.ogg", {"minio"})

    def test_download_enforces_byte_limit_and_removes_partial_file(self) -> None:
        settings = AiSettings(
            TRANSCRIBE_ALLOWED_HOSTS="minio",
            TRANSCRIBE_MAX_BYTES=5,
        )
        file_descriptor, path = tempfile.mkstemp()
        os.close(file_descriptor)
        Path(path).unlink()
        response = _FakeDownloadResponse(
            "http://minio/audio.ogg",
            [b"123", b"456"],
        )
        opener = SimpleNamespace(open=lambda *_args, **_kwargs: response)

        with (
            patch(
                "app.services.transcription.tempfile.mkstemp",
                return_value=(os.open(path, os.O_CREAT | os.O_WRONLY), path),
            ),
            patch(
                "app.services.transcription.urllib.request.build_opener",
                return_value=opener,
            ),
        ):
            with self.assertRaisesRegex(TranscriptionError, "size limit"):
                download_media("http://minio/audio.ogg", settings)

        self.assertFalse(Path(path).exists())

    def test_transcribe_file_posts_multipart_and_maps_metadata(self) -> None:
        audio_path = Path(__file__).resolve()
        response = httpx.Response(
            200,
            json={"text": " Xin chào ", "language": "vi", "duration": 8.5},
            request=httpx.Request("POST", "http://router/v1/audio/transcriptions"),
        )
        uploaded = {}

        def capture_request(url, **kwargs):
            uploaded["bytes"] = kwargs["files"]["file"][1].read()
            return response

        with patch("httpx.post", side_effect=capture_request) as post:
            result = transcribe_file(
                str(audio_path),
                AiSettings(
                    NINEROUTER_URL="http://router",
                    NINEROUTER_API_KEY="router-secret",
                    NINEROUTER_STT_MODEL="free-stt-route",
                    NINEROUTER_LANGUAGE="vi",
                    NINEROUTER_TIMEOUT_SECONDS=17,
                ),
            )

        self.assertEqual(result.transcript, "Xin chào")
        self.assertEqual(result.language, "vi")
        self.assertEqual(result.duration_seconds, 8.5)
        post.assert_called_once()
        self.assertEqual(post.call_args.args[0], "http://router/v1/audio/transcriptions")
        kwargs = post.call_args.kwargs
        self.assertEqual(
            kwargs["data"],
            {
                "model": "free-stt-route",
                "language": "vi",
                "response_format": "verbose_json",
            },
        )
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer router-secret")
        self.assertEqual(kwargs["timeout"], 17)
        self.assertEqual(kwargs["files"]["file"][0], audio_path.name)
        self.assertEqual(uploaded["bytes"], audio_path.read_bytes())

    def test_transcribe_file_retries_5xx_and_rate_limits_in_order(self) -> None:
        audio_path = Path(__file__).resolve()
        audio_bytes = audio_path.read_bytes()
        for status_code in (503, 429):
            with self.subTest(status_code=status_code):
                responses = [
                    httpx.Response(
                        status_code,
                        json={"error": {"message": "upstream detail"}},
                        request=httpx.Request(
                            "POST", "http://router/v1/audio/transcriptions"
                        ),
                    ),
                    httpx.Response(
                        200,
                        json={"text": "Fallback transcript"},
                        request=httpx.Request(
                            "POST", "http://router/v1/audio/transcriptions"
                        ),
                    ),
                ]
                requests = []

                def capture_request(_url, **kwargs):
                    uploaded = kwargs["files"]["file"][1].read()
                    requests.append(
                        (
                            kwargs["data"]["model"],
                            kwargs["timeout"],
                            uploaded,
                        )
                    )
                    return responses.pop(0)

                with patch("httpx.post", side_effect=capture_request) as post:
                    result = transcribe_file(
                        str(audio_path),
                        AiSettings(
                            NINEROUTER_URL="http://router",
                            NINEROUTER_STT_MODEL="stt-primary, stt-fallback",
                            NINEROUTER_TIMEOUT_SECONDS=24,
                        ),
                    )

                self.assertEqual(result.transcript, "Fallback transcript")
                self.assertEqual(post.call_count, 2)
                self.assertEqual(
                    requests,
                    [
                        ("stt-primary", 12, audio_bytes),
                        ("stt-fallback", 12, audio_bytes),
                    ],
                )

    def test_transcribe_file_retries_network_and_invalid_responses(self) -> None:
        audio_path = Path(__file__).resolve()
        endpoint = "http://router/v1/audio/transcriptions"
        failures = (
            httpx.TimeoutException("router timed out"),
            httpx.Response(200, content=b"not-json", request=httpx.Request("POST", endpoint)),
            httpx.Response(200, json={}, request=httpx.Request("POST", endpoint)),
            httpx.Response(
                200,
                json={"text": "  "},
                request=httpx.Request("POST", endpoint),
            ),
        )
        for failure in failures:
            with self.subTest(failure=failure):
                fallback = httpx.Response(
                    200,
                    json={"text": "Fallback transcript"},
                    request=httpx.Request("POST", endpoint),
                )
                with patch("httpx.post", side_effect=[failure, fallback]) as post:
                    result = transcribe_file(
                        str(audio_path),
                        AiSettings(
                            NINEROUTER_URL="http://router",
                            NINEROUTER_STT_MODEL="stt-primary,stt-fallback",
                        ),
                    )

                self.assertEqual(result.transcript, "Fallback transcript")
                self.assertEqual(post.call_count, 2)
                self.assertEqual(
                    [entry.kwargs["data"]["model"] for entry in post.call_args_list],
                    ["stt-primary", "stt-fallback"],
                )

    def test_transcribe_file_fails_fast_on_nonretryable_http_status(self) -> None:
        audio_path = Path(__file__).resolve()
        endpoint = "http://router/v1/audio/transcriptions"
        failure = httpx.Response(
            400,
            json={"error": {"message": "invalid request"}},
            request=httpx.Request("POST", endpoint),
        )
        fallback = httpx.Response(
            200,
            json={"text": "Should not be used"},
            request=httpx.Request("POST", endpoint),
        )
        with patch("httpx.post", side_effect=[failure, fallback]) as post:
            with self.assertRaises(TranscriptionError):
                transcribe_file(
                    str(audio_path),
                    AiSettings(
                        NINEROUTER_URL="http://router",
                        NINEROUTER_STT_MODEL="stt-primary,stt-fallback",
                    ),
                )

        self.assertEqual(post.call_count, 1)

    def test_transcribe_file_raises_after_all_candidates_fail(self) -> None:
        audio_path = Path(__file__).resolve()
        endpoint = "http://router/v1/audio/transcriptions"
        failures = [
            httpx.Response(
                503,
                json={"error": {"message": "upstream detail"}},
                request=httpx.Request("POST", endpoint),
            )
            for _ in range(2)
        ]
        with patch("httpx.post", side_effect=failures) as post:
            with self.assertRaises(TranscriptionError) as raised:
                transcribe_file(
                    str(audio_path),
                    AiSettings(
                        NINEROUTER_URL="http://router",
                        NINEROUTER_STT_MODEL="stt-primary,stt-fallback",
                    ),
                )

        self.assertEqual(post.call_count, 2)
        self.assertNotIn("upstream detail", str(raised.exception))

    def test_transcribe_file_uses_defaults_when_provider_omits_metadata(self) -> None:
        audio_path = Path(__file__).resolve()
        response = httpx.Response(
            200,
            json={"text": "Nội dung bài học"},
            request=httpx.Request("POST", "http://router/v1/audio/transcriptions"),
        )
        with patch("httpx.post", return_value=response) as post:
            result = transcribe_file(
                str(audio_path),
                AiSettings(
                    NINEROUTER_URL="http://router",
                    NINEROUTER_STT_MODEL="free-stt-route",
                    NINEROUTER_LANGUAGE="",
                ),
            )

        self.assertEqual(result.language, "vi")
        self.assertEqual(result.duration_seconds, 0)
        self.assertNotIn("Authorization", post.call_args.kwargs["headers"])
        self.assertEqual(post.call_args.kwargs["data"]["language"], "vi")

    def test_transcribe_file_rejects_malformed_or_empty_payloads(self) -> None:
        responses = [
            httpx.Response(
                200,
                content=b"not-json",
                request=httpx.Request("POST", "http://router/v1/audio/transcriptions"),
            ),
            httpx.Response(
                200,
                json={},
                request=httpx.Request("POST", "http://router/v1/audio/transcriptions"),
            ),
            httpx.Response(
                200,
                json={"text": "   "},
                request=httpx.Request("POST", "http://router/v1/audio/transcriptions"),
            ),
        ]
        audio_path = Path(__file__).resolve()
        for response in responses:
            with self.subTest(response=response), patch(
                "httpx.post", return_value=response
            ):
                with self.assertRaises(TranscriptionError):
                    transcribe_file(
                        str(audio_path),
                        AiSettings(
                            NINEROUTER_URL="http://router",
                            NINEROUTER_STT_MODEL="free-stt-route",
                        ),
                    )

    def test_transcribe_file_translates_http_and_network_failures(self) -> None:
        audio_path = Path(__file__).resolve()
        failures = [
            httpx.Response(
                503,
                request=httpx.Request("POST", "http://router/v1/audio/transcriptions"),
            ),
            httpx.TimeoutException("router timeout"),
        ]
        for failure in failures:
            with self.subTest(failure=failure):
                mock_options = (
                    {"return_value": failure}
                    if isinstance(failure, httpx.Response)
                    else {"side_effect": failure}
                )
                with patch("httpx.post", **mock_options):
                    with self.assertRaises(TranscriptionError):
                        transcribe_file(
                            str(audio_path),
                            AiSettings(
                                NINEROUTER_URL="http://router",
                                NINEROUTER_STT_MODEL="free-stt-route",
                            ),
                        )

    def test_transcribe_file_requires_router_url_and_model(self) -> None:
        audio_path = Path(__file__).resolve()
        with patch("httpx.post") as post:
            with self.assertRaises(TranscriptionError):
                transcribe_file(
                    str(audio_path),
                    AiSettings(NINEROUTER_URL="", NINEROUTER_STT_MODEL=""),
                )
        post.assert_not_called()

    async def test_transcribe_audio_cleans_up_after_router_failure(self) -> None:
        file_descriptor, path = tempfile.mkstemp()
        os.close(file_descriptor)
        with (
            patch(
                "app.services.transcription.download_media",
                return_value=path,
            ),
            patch(
                "app.services.transcription.transcribe_file",
                side_effect=TranscriptionError("router unavailable"),
            ),
        ):
            with self.assertRaisesRegex(TranscriptionError, "router unavailable"):
                await transcribe_audio("http://minio/audio.ogg", AiSettings())

        self.assertFalse(Path(path).exists())

    async def test_transcribe_audio_does_not_call_router_for_disallowed_media(self) -> None:
        with patch("httpx.post") as post:
            with self.assertRaisesRegex(TranscriptionError, "not allowed"):
                await transcribe_audio(
                    "https://example.com/audio.ogg",
                    AiSettings(TRANSCRIBE_ALLOWED_HOSTS="minio"),
                )
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
