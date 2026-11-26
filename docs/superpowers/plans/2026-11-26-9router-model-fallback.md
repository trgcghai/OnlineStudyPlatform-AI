# 9Router Model Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ordered comma-separated 9Router model fallbacks for chat, lesson summaries, and transcription while preserving each request's existing timeout budget and API contract.

**Architecture:** Parse each existing model environment variable into an ordered tuple of unique candidates. Chat and summary requests share an async candidate loop; transcription uses a matching synchronous loop that reopens the audio file for each candidate. Retry eligible upstream and content failures, validate final summaries inside the candidate loop, and split the existing timeout evenly among candidates.

**Tech Stack:** Python 3.13, FastAPI service, Pydantic Settings, httpx, unittest, uv.

---

## File map

- `app/core/config.py`: parse model lists and expose chat, summary, and STT candidate tuples.
- `app/services/ninerouter.py`: sequential chat-completions attempts, HTTP status classification, per-candidate timeout, and optional output validation callback.
- `app/services/transcription.py`: sequential multipart attempts with per-candidate timeout and a fresh file handle per attempt.
- `tests/test_settings.py`: parsing and configuration assertions.
- `tests/test_ninerouter.py`: async chat/summary fallback behavior and request assertions.
- `tests/test_summary.py`: final summary structure fallback and transcription failover regressions.
- `.env.example`, `README.md`: AI service configuration and operating documentation.
- Root `F:\Cntt\KLTN\.env.example`: document list values; do not touch unrelated hunks already modified in the worktree.
- `docker-compose.yml`: no change; existing environment forwarding passes comma-separated strings without transformation.

## Task 1: Parse ordered model candidate lists

**Files:**
- Modify: `app/core/config.py`
- Test: `tests/test_settings.py`

- [ ] **Step 1: Add a failing settings test**

Add a test to `SettingsTests` that verifies whitespace trimming, empty-item removal, duplicate removal, ordering, and the unchanged single-model case:

```python
def test_ai_settings_parse_ordered_9router_model_lists(self) -> None:
    from app.core.config import AiSettings

    settings = AiSettings(
        NINEROUTER_CHAT_MODEL=" chat-a, ,chat-b,chat-a ",
        NINEROUTER_SUMMARY_MODEL="summary-a, summary-b",
        NINEROUTER_STT_MODEL="stt-a",
    )

    self.assertEqual(settings.ninerouter_chat_models, ("chat-a", "chat-b"))
    self.assertEqual(settings.ninerouter_summary_models, ("summary-a", "summary-b"))
    self.assertEqual(settings.ninerouter_stt_models, ("stt-a",))
    self.assertEqual(AiSettings().ninerouter_chat_models, ())
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run from `F:\Cntt\KLTN\online-study-platform-AI`:

```powershell
uv run python -m unittest discover -s tests -p test_settings.py
```

Expected: FAIL because the candidate tuple properties do not exist yet.

- [ ] **Step 3: Implement the parser and properties**

In `app/core/config.py`, add:

```python
def _parse_model_candidates(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        candidate.strip()
        for candidate in value.split(",")
        if candidate.strip()
    ))
```

Add read-only `AiSettings` properties named `ninerouter_chat_models`, `ninerouter_summary_models`, and `ninerouter_stt_models`. Each property returns `_parse_model_candidates()` applied to its existing singular string setting; keep the existing Pydantic field names and environment aliases unchanged.

- [ ] **Step 4: Re-run the focused settings tests**

Run:

```powershell
uv run python -m unittest discover -s tests -p test_settings.py
```

Expected: PASS, including the existing single-model settings tests.

- [ ] **Step 5: Keep the settings change uncommitted**

Do not stage or commit; the user requested that no further commits be made.

## Task 2: Add async chat and summary model fallback

**Files:**
- Modify: `app/services/ninerouter.py`
- Test: `tests/test_ninerouter.py`

- [ ] **Step 1: Add failing async adapter tests**

Add tests to the existing `NineRouterChatAdapterTests` and `NineRouterSummaryAdapterTests` verifying:

1. A 503 or 429 from candidate one makes a second request with candidate two and returns its usable content.
2. A timeout/network exception or malformed/empty 2xx response makes a second request.
3. A 400 makes exactly one request and raises the existing operation-specific error.
4. With two candidates and a 24-second configured timeout, each `AsyncClient` is constructed with `timeout=12`.
5. One candidate still receives the full configured timeout.
6. If all candidates fail, the operation raises its existing typed error without exposing upstream response body text.

Use the existing `_AsyncClientStub`, `_response`, `httpx.Response`, and `AsyncMock` helpers; assert `json["model"]` across `client.post.await_args_list` to verify order.

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

```powershell
uv run python -m unittest discover -s tests -p test_ninerouter.py
```

Expected: New candidate-list tests FAIL against the current one-model adapter.

- [ ] **Step 3: Implement the candidate loop**

Update `_generate_ninerouter_text` to receive `models: tuple[str, ...]` rather than one model and an optional validator:

```python
validator: Callable[[str], str] | None = None
```

Reject an empty model tuple using the existing operation-specific configuration error. For each model in order, use `timeout_seconds / len(models)` as that attempt's `httpx.AsyncClient` timeout. Keep the existing request body unchanged except for `"model": model`.

Classify candidate errors as follows:

- Catch `httpx.HTTPStatusError` separately. Continue only for status `429` or `500 <= status_code < 600`; raise the existing operation-specific error immediately for every other 4xx.
- Fail fast for every other HTTP status, including 3xx responses.
- Continue on `httpx.RequestError` and `OSError`.
- Continue on invalid JSON, invalid response shape, empty content, or a `NineRouterSummaryError` from the optional validator.
- Return the first non-empty accepted content.
- After exhaustion, raise the operation-specific error from the last failure; do not include upstream response bodies in its message.

Update `generate_ninerouter_text` to use `settings.ninerouter_summary_models` and accept the optional validator. Update `generate_ninerouter_chat_text` to use `settings.ninerouter_chat_models`. Preserve endpoint URLs, headers, prompts, temperatures, token limits, and error classes.

Import `Callable` from `collections.abc` for the validator annotation.

- [ ] **Step 4: Re-run the focused adapter tests**

Run:

```powershell
uv run python -m unittest discover -s tests -p test_ninerouter.py
```

Expected: PASS, including all pre-existing single-model request tests and new multi-model cases.

- [ ] **Step 5: Keep the adapter change uncommitted**

Do not stage or commit; the user requested that no further commits be made.

## Task 3: Retry structurally invalid final summaries

**Files:**
- Modify: `app/services/summarization.py`
- Test: `tests/test_summary.py`
- Test: `tests/test_ninerouter.py`

- [ ] **Step 1: Add a failing summary fallback test**

In `tests/test_ninerouter.py`, add an adapter test with two summary models, a validator that calls `validate_summary`, an invalid first response, and `VALID_SUMMARY` as the second response. Assert the first output is rejected, the second is returned normalized, and requests use the configured model order. In `tests/test_summary.py`, add a test that calls `summarize_transcript` with a mocked `generate_ninerouter_text` and verifies the final-synthesis call receives a validator while chunk-notes calls do not.

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```powershell
uv run python -m unittest discover -s tests -p test_summary.py
```

Expected: FAIL because current summary validation happens only after the one model response has returned.

- [ ] **Step 3: Validate each final candidate within the adapter loop**

In `app/services/summarization.py`, import `Callable` from `collections.abc` and let `_generate_summary_text` accept and forward `validator: Callable[[str], str] | None`. `NineRouterSummaryError` is already imported in this module. Add a small callback for the final synthesis call:

```python
def _validate_summary_candidate(summary: str, transcript: str) -> str:
    try:
        return validate_summary(summary, transcript)
    except SummarizationError as error:
        raise NineRouterSummaryError(
            "9Router returned invalid summary structure"
        ) from error
```

Pass this callback to `_generate_summary_text` only for the final synthesis prompt. Keep notes-generation calls on their current non-empty text validation. Return the already-validated final string without applying a second validation pass. Keep the public `SummarizationError` behavior when every model is exhausted.

- [ ] **Step 4: Re-run summary tests**

Run:

```powershell
uv run python -m unittest discover -s tests -p test_summary.py
```

Expected: PASS, including existing heading, section, bullet, transcript-copy, and chunking tests.

- [ ] **Step 5: Keep the summary behavior uncommitted**

Do not stage or commit; the user requested that no further commits be made.

## Task 4: Add transcription model fallback

**Files:**
- Modify: `app/services/transcription.py`
- Test: `tests/test_summary.py`

- [ ] **Step 1: Add failing transcription fallback tests**

Add `TranscriptionTests` cases verifying:

1. A 503 or 429 from the first model retries with the second model and returns its non-empty transcript.
2. A timeout/network failure or malformed/empty success payload retries with the next model.
3. A 400 fails fast with exactly one request.
4. Each candidate request has the same audio bytes, a fresh file handle, its own model field, and `timeout=settings.ninerouter_timeout_seconds / len(models)`.
5. Exhausting candidates raises `TranscriptionError`.

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

```powershell
uv run python -m unittest discover -s tests -p test_summary.py
```

Expected: New transcription fallback tests FAIL because `transcribe_file` currently sends one request.

- [ ] **Step 3: Implement the synchronous candidate loop**

In `transcribe_file`, read `settings.ninerouter_stt_models` and fail before I/O if it is empty. For each candidate:

- Reopen `path` in binary mode so every multipart attempt can read the complete audio.
- Send the existing form fields and headers, changing only `data["model"]`.
- Divide `settings.ninerouter_timeout_seconds` evenly across candidate count.
- Continue for 429/5xx, `httpx.RequestError`/`OSError`, invalid JSON/payload, and empty transcript.
- Fail immediately for every other HTTP status, including other 4xx and 3xx responses.
- Return the first valid `TranscriptionResult`; after exhaustion raise the existing `TranscriptionError` without exposing upstream body text.

Keep media download, host validation, cleanup, metadata defaults, and endpoint response unchanged.

- [ ] **Step 4: Re-run transcription and adapter tests**

Run:

```powershell
uv run python -m unittest discover -s tests -p test_summary.py
uv run python -m unittest discover -s tests -p test_ninerouter.py
```

Expected: PASS for transcription, chat, summary, and existing response validation coverage.

- [ ] **Step 5: Keep transcription fallback uncommitted**

Do not stage or commit; the user requested that no further commits be made.

## Task 5: Document configuration and run the complete AI suite

**Files:**
- Modify: `F:\Cntt\KLTN\online-study-platform-AI\.env.example`
- Modify: `F:\Cntt\KLTN\online-study-platform-AI\README.md`
- Modify: `F:\Cntt\KLTN\.env.example`

- [ ] **Step 1: Update example comments**

For each `NINEROUTER_CHAT_MODEL`, `NINEROUTER_SUMMARY_MODEL`, and `NINEROUTER_STT_MODEL`, document that values are comma-separated, attempted left-to-right, and share the existing operation timeout budget. Keep the variable names unchanged and do not edit `.env`.

- [ ] **Step 2: Update the AI README**

Update the Vietnamese configuration descriptions for chat, lesson summaries, and transcription. Document one ordered example such as `model-primary,model-fallback`, immediate failover for 5xx/429, timeout/network and unusable-output failover, fast failure on other 4xx, and the shared per-call timeout budget.

- [ ] **Step 3: Review root env changes without staging unrelated work**

Update only the 9Router comments in `F:\Cntt\KLTN\.env.example`. The root `.env.example` already contains an unrelated unstaged RabbitMQ edit: preserve it and leave the root env file unstaged rather than mixing changes. Do not edit `docker-compose.yml`; it already forwards each variable as a string.

- [ ] **Step 4: Run the complete AI unit suite**

Run from the AI service directory:

```powershell
uv run python -m unittest discover tests
```

Expected: PASS for all AI service tests.

- [ ] **Step 5: Review diffs and keep documentation uncommitted**

Run:

```powershell
git diff --check
git status --short
```

Confirm active `.env` files and Compose files are unchanged, and the existing root RabbitMQ worktree change is preserved. Do not stage or commit changes in either repository.
