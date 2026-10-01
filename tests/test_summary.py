import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.core.config import AiSettings, get_ai_settings
from app.main import app
from app.services.llm import call_gemini
from app.services.summarization import (
    SummarizationError,
    SummaryResult,
    build_final_summary_prompt,
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
                provider="gemini",
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
        self.assertEqual(response.json()["provider"], "gemini")

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

    def test_gemini_key_is_sent_in_header_not_url(self) -> None:
        class _GeminiResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self) -> bytes:
                return b'{"candidates":[{"content":{"parts":[{"text":"OK"}]}}]}'

        settings = AiSettings(GEMINI_API_KEY="private-key")
        with patch(
            "app.services.llm.urllib.request.urlopen",
            return_value=_GeminiResponse(),
        ) as urlopen:
            result = call_gemini("test", settings)

        request = urlopen.call_args.args[0]
        self.assertNotIn("private-key", request.full_url)
        self.assertEqual(request.get_header("X-goog-api-key"), "private-key")
        self.assertEqual(result, "OK")


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

    def test_chunk_transcript_splits_unpunctuated_whisper_text(self) -> None:
        transcript = "nội dung bài giảng " * 80
        chunks = chunk_transcript(transcript, 500)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 500 for chunk in chunks))

    async def test_short_transcript_returns_only_validated_gemini_output(self) -> None:
        settings = AiSettings(
            GEMINI_API_KEY="configured",
            SUMMARY_PROVIDER="gemini",
        )
        with patch(
            "app.services.summarization.generate_gemini_text",
            new_callable=AsyncMock,
            return_value=VALID_SUMMARY,
        ):
            result = await summarize_transcript(
                "Dependency Injection",
                "Dependency Injection cung cấp dependency cho lớp.",
                settings,
            )

        self.assertEqual(result.provider, "gemini")
        self.assertEqual(result.summary, VALID_SUMMARY)

    async def test_gemini_failure_is_not_replaced_by_extractive_transcript(self) -> None:
        settings = AiSettings(
            GEMINI_API_KEY="configured",
            SUMMARY_PROVIDER="gemini",
        )
        with patch(
            "app.services.summarization.generate_gemini_text",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with self.assertRaisesRegex(SummarizationError, "could not generate"):
                await summarize_transcript(
                    "Dependency Injection",
                    "Dependency Injection cung cấp dependency cho lớp.",
                    settings,
                )

    async def test_invalid_gemini_structure_is_rejected(self) -> None:
        settings = AiSettings(GEMINI_API_KEY="configured")
        with patch(
            "app.services.summarization.generate_gemini_text",
            new_callable=AsyncMock,
            return_value="Dependency Injection cung cấp dependency cho lớp.",
        ):
            with self.assertRaisesRegex(SummarizationError, "required structure"):
                await summarize_transcript(
                    "Dependency Injection",
                    "Dependency Injection cung cấp dependency cho lớp.",
                    settings,
                )

    def test_transcript_copy_is_rejected_even_when_it_has_headings(self) -> None:
        with self.assertRaisesRegex(SummarizationError, "instead of a summary"):
            validate_summary(VALID_SUMMARY, VALID_SUMMARY)

    def test_valid_summary_normalizes_common_markdown_bullet_styles(self) -> None:
        star_bullets = VALID_SUMMARY.replace("- Dependency", "* Dependency").replace(
            "- Constructor", "2. Constructor"
        )

        result = validate_summary(star_bullets, "Transcript nguồn khác.")

        self.assertIn("- Dependency", result)
        self.assertIn("- Constructor", result)

    async def test_missing_gemini_key_is_a_retryable_summary_error(self) -> None:
        with self.assertRaisesRegex(SummarizationError, "API key"):
            await summarize_transcript(
                "Bài học",
                "Nội dung bài học có thông tin.",
                AiSettings(GEMINI_API_KEY=None),
            )

    async def test_long_transcript_uses_notes_then_final_gemini_call(self) -> None:
        settings = AiSettings(
            GEMINI_API_KEY="configured",
            SUMMARY_PROVIDER="gemini",
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
            "app.services.summarization.generate_gemini_text",
            new_callable=AsyncMock,
        ) as generate:
            generate.side_effect = [
                *["- Ghi chú sự kiện quan trọng."] * len(chunks),
                VALID_SUMMARY,
            ]
            result = await summarize_transcript("Bài học", transcript, settings)

        self.assertEqual(result.provider, "gemini")
        self.assertEqual(generate.await_count, len(chunks) + 1)


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

    def test_whisper_segments_are_joined_in_order(self) -> None:
        model = SimpleNamespace(
            transcribe=lambda *_args, **_kwargs: (
                [
                    SimpleNamespace(text=" Xin chào "),
                    SimpleNamespace(text=" bài giảng. "),
                ],
                SimpleNamespace(language="vi", duration=8.5),
            )
        )
        with patch(
            "app.services.transcription.get_whisper_model",
            return_value=model,
        ):
            result = transcribe_file("audio.ogg", AiSettings())

        self.assertEqual(result.transcript, "Xin chào bài giảng.")
        self.assertEqual(result.duration_seconds, 8.5)

    async def test_transcribe_audio_cleans_up_after_whisper_failure(self) -> None:
        file_descriptor, path = tempfile.mkstemp()
        os.close(file_descriptor)
        with (
            patch(
                "app.services.transcription.download_media",
                return_value=path,
            ),
            patch(
                "app.services.transcription.transcribe_file",
                side_effect=TranscriptionError("empty transcript"),
            ),
        ):
            with self.assertRaisesRegex(TranscriptionError, "empty transcript"):
                await transcribe_audio("http://minio/audio.ogg", AiSettings())

        self.assertFalse(Path(path).exists())


if __name__ == "__main__":
    unittest.main()
