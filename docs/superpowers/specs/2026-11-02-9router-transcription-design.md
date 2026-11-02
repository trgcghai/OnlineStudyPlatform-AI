# 9Router transcription integration design

Date: 2026-11-02
Status: Proposed; awaiting design approval

## Intent

Replace local faster-whisper transcription because the application server does not have enough resources for the model. FastAPI will call the self-hosted 9Router service. 9Router is responsible for selecting upstream free STT providers and handling fallback.

## Goals and constraints

- Remove local speech-model inference and keep transcription available through 9Router.
- Preserve the current NestJS summary queue, FastAPI `/transcribe` route, response fields, transcript persistence, Gemini summary generation, and downstream transcript indexing.
- Keep FFmpeg audio extraction in NestJS. FastAPI continues to download the extracted audio URL with its current host allowlist and size limit, then uploads the audio file to 9Router using multipart form data.
- Keep 9Router provider selection/fallback outside the application. FastAPI sends one configured STT model/route and does not implement provider fallback or local Whisper fallback.
- Do not edit the active `.env`; document new settings in environment examples and wire them through Docker Compose.

## Approaches considered

1. **FastAPI calls 9Router (selected).** Preserves the existing NestJS-to-FastAPI boundary and keeps provider credentials and response mapping in the AI service. This is the requested architecture.
2. **NestJS calls 9Router directly.** Moves multipart upload and provider configuration into the backend and changes the current service boundary, so it is not selected.

## Proposed behavior

FastAPI sends the configured STT model identifier, the extracted audio file, Vietnamese language hint, and `verbose_json` response format to the self-hosted 9Router `/v1/audio/transcriptions` endpoint. The integration validates a non-empty transcript and maps returned language/duration metadata to the existing `transcript`, `language`, and `durationSeconds` response. If the upstream omits language, use the requested language (`vi`); if it omits duration, use `0` to mean unavailable while preserving the required numeric response field. Network errors, unsuccessful HTTP responses, malformed payloads, and empty transcripts become the existing transcription error, allowing the NestJS BullMQ retry policy to apply. Temporary audio files are still removed in a `finally` block.

Add AI settings and example values for the 9Router base URL, optional API key, required STT model/route ID, requested language, and request timeout. Pass those settings from root `.env.example` through root `docker-compose.yml` to the AI container. The configured URL must be reachable from FastAPI at runtime. The existing NestJS transcription timeout remains independently configured and must cover FastAPI download plus the 9Router request.

## Removal and retained pieces

- Remove faster-whisper model loading, local transcription functions, Whisper-only environment settings, and the `faster-whisper` dependency/lockfile entries if unused elsewhere.
- Retain FastAPI's guarded media download, size/time limits, safe redirect checks, temporary-file cleanup, and transcription result type/endpoint contract.
- Retain NestJS FFmpeg extraction, summary queue/retries, transcript and summary storage, and Gemini summarization.

## Non-goals

- Changing summary generation, Gemini settings, chatbot/RAG, Qdrant, or NestJS endpoints.
- Adding a 9Router service/container to the root Compose stack; the router is self-hosted separately and addressed by configuration.
- Guaranteeing permanent free provider availability; provider quota and routing configuration remain in 9Router.

## Implementation scope

Expected files: FastAPI config/transcription and tests; AI `pyproject.toml` and `uv.lock`; AI `.env.example` and `Dockerfile`; root `.env.example` and `docker-compose.yml`. Remove the Whisper cache directory and volume alongside the model dependency.

