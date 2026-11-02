import asyncio
import math
import mimetypes
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from app.core.config import AiSettings


DOWNLOAD_CHUNK_SIZE = 1024 * 1024


class TranscriptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class TranscriptionResult:
    transcript: str
    language: str
    duration_seconds: float


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_hosts: set[str]) -> None:
        super().__init__()
        self.allowed_hosts = allowed_hosts

    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> urllib.request.Request | None:
        validate_media_url(new_url, self.allowed_hosts)
        return super().redirect_request(
            request,
            file_pointer,
            code,
            message,
            headers,
            new_url,
        )


def validate_media_url(url: str, allowed_hosts: set[str]) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise TranscriptionError("Media URL must use HTTP or HTTPS")

    hostname = (parsed.hostname or "").casefold()
    if not hostname or hostname not in allowed_hosts:
        raise TranscriptionError("Media host is not allowed")


def download_media(url: str, settings: AiSettings) -> str:
    allowed_hosts = settings.transcribe_allowed_host_set
    validate_media_url(url, allowed_hosts)
    request = urllib.request.Request(url, headers={"User-Agent": "kltn-ai-summary/1.0"})
    opener = urllib.request.build_opener(_SafeRedirectHandler(allowed_hosts))
    suffix = Path(urllib.parse.urlparse(url).path).suffix[:16] or ".media"
    file_descriptor, temporary_path = tempfile.mkstemp(suffix=suffix)
    bytes_written = 0

    try:
        with os.fdopen(file_descriptor, "wb") as destination:
            with opener.open(
                request,
                timeout=settings.transcribe_download_timeout_seconds,
            ) as response:
                final_url = response.geturl()
                validate_media_url(final_url, allowed_hosts)
                content_length = response.headers.get("Content-Length")
                if (
                    content_length
                    and int(content_length) > settings.transcribe_max_bytes
                ):
                    raise TranscriptionError("Media exceeds transcription size limit")

                while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                    bytes_written += len(chunk)
                    if bytes_written > settings.transcribe_max_bytes:
                        raise TranscriptionError(
                            "Media exceeds transcription size limit"
                        )
                    destination.write(chunk)
    except (OSError, ValueError, urllib.error.URLError) as error:
        Path(temporary_path).unlink(missing_ok=True)
        raise TranscriptionError("Could not download lesson audio") from error
    except Exception:
        Path(temporary_path).unlink(missing_ok=True)
        raise

    if bytes_written == 0:
        Path(temporary_path).unlink(missing_ok=True)
        raise TranscriptionError("Lesson audio is empty")

    return temporary_path


def transcribe_file(path: str, settings: AiSettings) -> TranscriptionResult:
    base_url = settings.ninerouter_url.strip().rstrip("/")
    model = settings.ninerouter_stt_model.strip()
    if not base_url or not model:
        raise TranscriptionError("9Router URL and STT model must be configured")

    parsed_url = urllib.parse.urlparse(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise TranscriptionError("9Router URL must use HTTP or HTTPS")

    endpoint = f"{base_url}/v1/audio/transcriptions"
    headers: dict[str, str] = {}
    api_key = settings.ninerouter_api_key
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    language_hint = settings.ninerouter_language.strip() or "vi"
    try:
        with open(path, "rb") as audio_file:
            response = httpx.post(
                endpoint,
                data={
                    "model": model,
                    "language": language_hint,
                    "response_format": "verbose_json",
                },
                files={
                    "file": (Path(path).name, audio_file, content_type),
                },
                headers=headers,
                timeout=settings.ninerouter_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, OSError) as error:
        raise TranscriptionError("9Router transcription request failed") from error
    except ValueError as error:
        raise TranscriptionError("9Router returned invalid transcription JSON") from error

    if not isinstance(payload, dict):
        raise TranscriptionError("9Router returned an invalid transcription payload")
    transcript = payload.get("text")
    if not isinstance(transcript, str) or not transcript.strip():
        raise TranscriptionError("9Router returned an empty transcript")

    language = payload.get("language")
    if not isinstance(language, str) or not language.strip():
        language = language_hint
    duration_value = payload.get("duration")
    duration = 0.0
    if isinstance(duration_value, (int, float)) and not isinstance(duration_value, bool):
        parsed_duration = float(duration_value)
        if math.isfinite(parsed_duration) and parsed_duration >= 0:
            duration = parsed_duration

    return TranscriptionResult(
        transcript=transcript.strip(),
        language=language,
        duration_seconds=duration,
    )


async def transcribe_audio(url: str, settings: AiSettings) -> TranscriptionResult:
    temporary_path = await asyncio.to_thread(download_media, url, settings)
    try:
        return await asyncio.to_thread(transcribe_file, temporary_path, settings)
    finally:
        Path(temporary_path).unlink(missing_ok=True)
