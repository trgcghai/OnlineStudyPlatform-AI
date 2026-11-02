# 9Router Transcription Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace local faster-whisper transcription with a request from FastAPI to the self-hosted 9Router STT endpoint while preserving the existing NestJS summary flow and transcription response contract.

**Architecture:** FastAPI keeps its safe audio download and temporary-file lifecycle, then uploads that audio to 9Router using multipart form data. 9Router owns upstream provider routing/fallback; FastAPI validates and normalizes the response. Root Compose passes the configured endpoint and credentials to the AI container.

**Tech Stack:** Python 3.13, FastAPI, Pydantic Settings, HTTPX multipart client, uv, unittest/pytest, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-11-02-9router-transcription-design.md`

## Global Constraints

- FastAPI calls the self-hosted 9Router `/v1/audio/transcriptions` endpoint with a configured model/route ID.
- Do not implement provider routing/fallback or a local Whisper fallback in this application.
- Keep the existing NestJS `/transcribe` request/response contract and summary job behavior.
- Keep FFmpeg extraction in NestJS and FastAPI's media allowlist, size/time limits, redirect checks, and temporary-file cleanup.
- Do not edit the active `.env`; update only `.env.example` files and Compose wiring.
- Leave Gemini summary generation, transcript storage, chatbot/RAG, and Qdrant behavior unchanged.

## Review Focus

1. **9Router unavailable, times out, or returns non-2xx:** convert to `TranscriptionError`, clean the temporary file, and allow the existing NestJS retry policy to handle it. Test network/HTTP failures and cleanup.
2. **Malformed JSON or missing/blank transcript:** reject as a transcription failure rather than persisting an empty transcript. Test malformed, missing, and whitespace-only `text` values.
3. **Provider omits optional metadata:** use requested language `vi` and duration `0` when absent, preserving the required response fields. Test both omissions and valid provider metadata mapping.
4. **Unsafe or oversized source media:** preserve existing URL/redirect/size validation and do not call 9Router. Keep existing downloader tests.
5. **Missing endpoint or model configuration:** fail clearly before making an outbound request; API key remains optional when 9Router auth is disabled. Test missing configuration and optional auth header behavior.

---

### Task 1: Replace local Whisper inference with the 9Router STT adapter

**Files:**
- Modify: `app/core/config.py`
- Modify: `app/services/transcription.py`
- Modify: `app/api/summary.py` only if response adaptation requires it
- Modify: `tests/test_summary.py`
- Modify: `tests/test_settings.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Preserve `TranscriptionResult(transcript: str, language: str, duration_seconds: float)`.
- Preserve `transcribe_audio(url: str, settings: AiSettings) -> TranscriptionResult`.
- Preserve FastAPI `/transcribe` response keys `transcript`, `language`, and `durationSeconds`.
- Add settings for 9Router URL, optional API key, STT model/route ID, language (default `vi`), and positive request timeout.

- [x] **Step 1: Add failing adapter tests** in `tests/test_summary.py` for `transcribe_file(path: str, settings: AiSettings) -> TranscriptionResult` asserting multipart fields `model`, `language`, and `response_format=verbose_json`; uploaded file name/content; optional Bearer header; mapping of `text`, `language`, and `duration`; fallback language/duration; malformed/empty transcript rejection; HTTP/network error translation; and temporary-file cleanup after the upstream fails. Update or replace the old Whisper-mocked tests.
- [x] **Step 2: Run the focused transcription tests and confirm the new adapter cases fail** using `uv run python -m pytest tests/test_summary.py -k transcrib -q -p no:cacheprovider`.
- [x] **Step 3: Add explicit HTTPX dependency and 9Router settings** in `pyproject.toml` and `app/core/config.py`; keep the API key optional, require a usable URL/model at transcription time, default language to `vi`, and validate the timeout as positive. Regenerate `uv.lock` with `uv lock`.
- [x] **Step 4: Replace the Whisper code path** in `app/services/transcription.py` with an HTTPX multipart request to `<NINEROUTER_URL>/v1/audio/transcriptions`. Send the configured model, language, and `verbose_json`; parse non-empty `text`; use response metadata where valid and the spec's fallbacks otherwise; map HTTP, network, JSON, and payload errors to `TranscriptionError`. Keep the existing guarded download and `finally` cleanup. Remove model cache/loading and faster-whisper-specific code.
- [x] **Step 5: Run the focused transcription tests and confirm they pass** using `uv run python -m pytest tests/test_summary.py -k transcrib -q -p no:cacheprovider`.
- [x] **Step 6: Run the AI unit suite** using `uv run python -m unittest discover tests`; confirm the `/transcribe` API response contract and unrelated summary behavior still pass.
- [ ] **Step 7: Commit the AI service changes** in the AI submodule with a focused message such as `feat: transcribe lesson audio through 9router`.

### Task 2: Wire runtime configuration and remove Whisper deployment artifacts

**Files:**
- Modify: `.env.example` (AI repo)
- Modify: `Dockerfile` (AI repo)
- Modify: `../.env.example` (workspace root)
- Modify: `../docker-compose.yml` (workspace root)

**Interfaces:**
- AI settings consume `NINEROUTER_URL`, `NINEROUTER_API_KEY`, `NINEROUTER_STT_MODEL`, `NINEROUTER_LANGUAGE`, and `NINEROUTER_TIMEOUT_SECONDS`.
- Root Compose forwards those variables to the AI container. The router is separately hosted; no 9Router service is added to Compose.

- [x] **Step 1: Update the AI `.env.example`** with the five 9Router settings and remove `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`, `WHISPER_LANGUAGE`, and `WHISPER_DOWNLOAD_ROOT`; retain the `TRANSCRIBE_*` media safety settings.
- [x] **Step 2: Update root `.env.example` and `docker-compose.yml`** to add the 9Router settings and pass them only to the AI service; remove Whisper variables and the Whisper cache mount/volume while retaining embedding and AI model caches.
- [x] **Step 3: Remove Whisper cache directory creation from the AI `Dockerfile`** while keeping the embedding and model-artifact cache directories.
- [x] **Step 4: Check configuration wiring without exposing `.env` values** by reviewing the diff and running `docker compose --env-file .env.example config -q` from the root; expected result is valid Compose configuration with no Whisper references. Do not print resolved Compose configuration.
- [x] **Step 5: Search for stale Whisper references** in AI code, AI/root examples, Compose, and Dockerfile; expected remaining occurrences should be limited to historical docs or test names only if intentionally retained.
- [ ] **Step 6: Commit AI-repo deployment changes** in the AI submodule, then commit root example/Compose changes in the root repo on a non-main feature branch. Do not include or stage the active `.env`.

## Execution Notes

The AI submodule is on `feature-9router-stt`. Root-level files require a separate root feature branch before editing; the root worktree currently has pre-existing changes, so inspect and preserve them during execution. Do not use the active `.env` as the Compose validation source; use `.env.example` and do not print resolved configuration. The AI tests are pytest-compatible unittest tests; the commands above do not require a live 9Router service because upstream HTTP calls are mocked.


