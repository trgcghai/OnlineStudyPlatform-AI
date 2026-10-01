import asyncio
import os
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


_model_lock = threading.Lock()
_model_cache: dict[tuple[str, str, str], Any] = {}


def get_whisper_model(settings: AiSettings) -> Any:
    key = (
        settings.whisper_model,
        settings.whisper_device,
        settings.whisper_compute_type,
    )
    with _model_lock:
        if key not in _model_cache:
            from faster_whisper import WhisperModel

            _model_cache[key] = WhisperModel(
                settings.whisper_model,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
                download_root=settings.whisper_download_root,
            )
        return _model_cache[key]


def transcribe_file(path: str, settings: AiSettings) -> TranscriptionResult:
    model = get_whisper_model(settings)
    segments, info = model.transcribe(
        path,
        language=settings.whisper_language,
        vad_filter=True,
        beam_size=5,
    )
    transcript = " ".join(
        segment.text.strip() for segment in segments if segment.text.strip()
    ).strip()
    if not transcript:
        raise TranscriptionError("Whisper returned an empty transcript")

    language = str(getattr(info, "language", settings.whisper_language))
    duration = float(getattr(info, "duration", 0) or 0)
    return TranscriptionResult(
        transcript=transcript,
        language=language,
        duration_seconds=duration,
    )


async def transcribe_audio(url: str, settings: AiSettings) -> TranscriptionResult:
    temporary_path = await asyncio.to_thread(download_media, url, settings)
    try:
        return await asyncio.to_thread(transcribe_file, temporary_path, settings)
    finally:
        Path(temporary_path).unlink(missing_ok=True)
