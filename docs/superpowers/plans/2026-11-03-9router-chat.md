# 9Router Chat Answer Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace direct Gemini chatbot answer generation with 9Router while preserving course RAG retrieval, scope checks, and citations.

**Architecture:** Keep Qdrant retrieval and `answer_with_rag` as-is. Add chat-specific 9Router model and timeout settings, reuse the existing 9Router URL/key and OpenAI-compatible chat-completions adapter pattern, and propagate generation failures so NestJS uses its current unavailable response.

**Tech Stack:** Python 3.13, FastAPI, Pydantic Settings, httpx, Qdrant, NestJS service integration, unittest.

**Spec:** `docs/superpowers/specs/2026-11-03-9router-chat-design.md`

## Global Constraints

- Keep the chat response contract `{ answer, inScope, similarity, citations }`.
- Use `NINEROUTER_CHAT_MODEL` and positive `NINEROUTER_CHAT_TIMEOUT_SECONDS` (default 120 seconds); reuse `NINEROUTER_URL` and optional `NINEROUTER_API_KEY`.
- Set the NestJS chat request timeout to 150,000 ms, leaving a 30-second margin beyond the default 120-second 9Router chat timeout; do not change the generic AI-client timeout.
- Send non-streaming requests to `<NINEROUTER_URL>/v1/chat/completions`, temperature `0.2`, and `max_tokens` 512.
- Do not fall back to Gemini; failures propagate to NestJS's existing unavailable-message handling.
- Do not change Qdrant, embeddings, chunking, retrieval ranking, summary generation, or transcription behavior.
- Do not edit a local `.env`; remove Gemini-only chatbot settings from config, both environment examples, Compose, and AI README.

## Review Focus

- URL with trailing slash: endpoint must still contain exactly one slash before `v1/chat/completions`; cover in `tests/test_ninerouter.py`.
- Missing URL or chat model: reject before making an HTTP request; cover in `tests/test_ninerouter.py`.
- Missing optional router key: send the request without `Authorization`; cover in `tests/test_ninerouter.py`.
- Non-2xx, timeout, invalid JSON, malformed choices, or empty message content: treat as generation failure with no Gemini fallback; cover adapter and chat endpoint tests.
- Out-of-scope retrieval: return the existing fixed reply and do not call 9Router; cover in `tests/test_chat.py`.
- 9Router response latency above the generic 4,500 ms NestJS timeout: chat's client override must remain 150,000 ms; cover in `online-study-platform-be/src/modules/ai-chat/ai-chat.client.spec.ts`.

---

### Task 1: Add chat settings and a reusable 9Router chat-completions adapter

**Files:**
- Modify: `app/core/config.py`
- Modify: `app/services/ninerouter.py`
- Test: `tests/test_settings.py`
- Test: `tests/test_ninerouter.py`

**Interfaces:**
- Produces `AiSettings.ninerouter_chat_model: str`, sourced from `NINEROUTER_CHAT_MODEL`, default `""`.
- Produces `AiSettings.ninerouter_chat_timeout_seconds: float`, sourced from `NINEROUTER_CHAT_TIMEOUT_SECONDS`, default `120`, validated greater than zero.
- Produces `generate_ninerouter_chat_text(prompt: str, settings: AiSettings, *, max_output_tokens: int = 512) -> str`, which returns non-empty assistant content or raises a chat-specific 9Router error.
- Preserve `generate_ninerouter_text` and `NineRouterSummaryError` behavior for lesson summaries.

- [x] **Step 1: Add failing settings tests** asserting both chat aliases/defaults and that timeout zero raises Pydantic `ValidationError`.
- [x] **Step 2: Add failing adapter tests** asserting URL/model, optional Bearer auth, `stream: false`, temperature `0.2`, `max_tokens`, timeout, parsed content, trailing-slash normalization, and rejection of missing URL/model, non-2xx, timeout, malformed JSON/shape, and blank content.
- [x] **Step 3: Run the focused tests and confirm the new expectations fail** with `uv run python -m unittest discover -s tests -p "test_settings.py"` and `uv run python -m unittest discover -s tests -p "test_ninerouter.py"`.
- [x] **Step 4: Add the two settings and implement `generate_ninerouter_chat_text`** using the existing `httpx.AsyncClient` request pattern. Share low-level request/response handling with summary calls where it preserves the summary error type and behavior; use a chat-specific error for chat failures.
- [x] **Step 5: Run the focused tests and confirm they pass** with `uv run python -m unittest discover -s tests -p "test_settings.py"` and `uv run python -m unittest discover -s tests -p "test_ninerouter.py"`.
- [x] **Step 6: Commit the AI adapter and settings changes** with `feat: add 9router chat adapter`.

### Task 2: Route grounded chatbot generation through 9Router

**Files:**
- Modify: `app/services/llm.py`
- Modify: `app/api/chat.py`
- Modify: `../online-study-platform-be/src/modules/ai-chat/ai-chat.client.ts`
- Modify: `tests/test_llm.py`
- Modify: `tests/test_chat.py`
- Test: `../online-study-platform-be/src/modules/ai-chat/ai-chat.client.spec.ts`
- Modify: `tests/test_summary.py` (remove the test-only legacy Gemini adapter reference)

**Interfaces:**
- Consumes `generate_ninerouter_chat_text` and the chat settings produced by Task 1.
- Produces `build_chat_prompt(question: str, history: list[RagHistoryMessage], citations: list[RagCitation]) -> str` and `generate_ninerouter_chat_answer(question, history, citations, settings) -> str | None`; no citation chunks preserve the existing deterministic answer path.
- Produces a NestJS `AiChatClient` request using `AiHttpClientService.post('/chat', request, { timeoutMs: 150_000 })`; the generic `AI_TIMEOUT_MS` stays unchanged.
- `generate_ninerouter_chat_answer` passes the existing Vietnamese grounding/safety instructions, selected history, and complete retrieved chunk text to the adapter; it raises on adapter failure and is only called for in-scope retrieval.

- [x] **Step 1: Replace Gemini-routing tests with failing 9Router tests** asserting prompt content includes the retrieved full chunk and selected history, and adapter receives the configured chat settings and 512 output-token limit.
- [x] **Step 2: Update chat tests to assert grounded in-scope answers use 9Router**, out-of-scope answers skip generation, and adapter errors fail the AI endpoint rather than return a Gemini or excerpt fallback. Use `TestClient(..., raise_server_exceptions=False)` to assert the endpoint returns 500 for a generation error.
- [x] **Step 2a: Remove the obsolete unit test for the unused direct Gemini helper from `tests/test_summary.py`** and update the summary 9Router failure test so it verifies summary error handling without patching the removed Gemini helper. Keep summary generation behavior unchanged.
- [x] **Step 2b: Add a failing NestJS chat-client test** asserting `/chat` is posted with `{ timeoutMs: 150_000 }`; retain coverage that a failed AI call returns `null` to the existing unavailable-message flow.
- [x] **Step 3: Run the focused tests and confirm the new expectations fail** with `uv run python -m unittest discover -s tests -p "test_llm.py"`, `uv run python -m unittest discover -s tests -p "test_chat.py"`, and from the BE repository `pnpm test --runInBand -- ai-chat.client.spec.ts`.
- [x] **Step 4: Replace Gemini prompt/generation helpers with the 9Router chat helpers** while keeping the Vietnamese system instructions, HISTORY/CONTEXT/QUESTION structure, and full citation text unchanged; update `app/api/chat.py` to call the new answer function only when `result.in_scope` is true.
- [x] **Step 5: Set the NestJS chat client timeout override to 150,000 ms** without changing the shared AI client timeout or other AI clients.
- [x] **Step 6: Run the focused tests and confirm they pass** with `uv run python -m unittest discover -s tests -p "test_llm.py"`, `uv run python -m unittest discover -s tests -p "test_chat.py"`, and from the BE repository `pnpm test --runInBand -- ai-chat.client.spec.ts`.
- [x] **Step 7: Commit chatbot routing changes inside the AI submodule** with `feat: route rag chat answers through 9router`, and commit the NestJS timeout change inside the BE submodule with `fix: allow time for 9router chat answers`.

### Task 3: Replace Gemini configuration references and document chat routing

**Files:**
- Modify: `app/core/config.py`
- Modify: `app/services/summarization.py` (make the stale provider-specific validation error generic)
- Modify: `.env.example`
- Modify: `../.env.example`
- Modify: `../docker-compose.yml`
- Modify: `README.md`
- Test: `tests/test_summary.py`

**Interfaces:**
- Consumes the exact setting names and 120-second default from Task 1.
- Produces deployment examples that pass `NINEROUTER_CHAT_MODEL` and `NINEROUTER_CHAT_TIMEOUT_SECONDS` to the AI container, with no Gemini-only chatbot variables.

- [x] **Step 1: Update the summary validation test** to expect provider-neutral transcript-copy wording, run `uv run python -m unittest discover -s tests -p "test_summary.py"`, and confirm it fails on the current Gemini-specific message.
- [x] **Step 2: Remove the four Gemini-only settings from `AiSettings`** and change the stale summary validation error text to `Model returned the transcript instead of a summary`; rerun the focused summary/settings tests.
- [x] **Step 3: Update both environment examples and Compose** to add `NINEROUTER_CHAT_MODEL=` and `NINEROUTER_CHAT_TIMEOUT_SECONDS=120`, remove `LLM_PROVIDER`, `GEMINI_API_KEY`, `GEMINI_MODEL`, and `GEMINI_TIMEOUT_SECONDS`, and preserve summary/transcription router settings.
- [x] **Step 4: Update the AI README** to describe 9Router-backed RAG answer generation and its new model/timeout settings; remove the claim that chat uses Gemini.
- [x] **Step 5: Search active source/configuration for obsolete Gemini chat/provider references** using `rg -n "LLM_PROVIDER|GEMINI_API_KEY|GEMINI_MODEL|GEMINI_TIMEOUT_SECONDS|Gemini|gemini" app tests README.md .env.example ../.env.example ../docker-compose.yml`; expected: no active production/configuration references; the settings test may name removed attributes to assert they are absent. Historical plans/specs may retain accurate records of the former design.
- [x] **Step 6: Commit the AI submodule config, examples, README, and tests** inside the AI repository with `docs: configure 9router for rag chat`.
- [x] **Step 7: Commit the root `.env.example` and Compose hunks plus the updated AI submodule pointer** inside the root repository with `docs: configure 9router for rag chat`. Preserve previously existing root edits by staging only the two 9Router hunks and the submodule pointer.

### Task 4: Verify the AI service and final configuration

**Files:**
- Verify: `tests/`
- Verify: `app/`, `README.md`, `.env.example`, `../.env.example`, `../docker-compose.yml`

- [ ] **Step 1: Run the full AI unit suite** from the AI repository with `$env:AI_TOKEN = ''; uv run python -m unittest discover tests`; expected: all tests pass. The process-only blank overrides the local `.env` token so recommendation tests run without auth, and it does not edit `.env`.
- [ ] **Step 2: Review the final diff and environment references** to confirm summary/transcription behavior and existing root worktree changes remain intact, the chat settings reach the `ai` Compose service, and no local `.env` was edited.
- [ ] **Step 3: Commit the final spec/plan timeout clarification and root pointer update** after review; any verification-driven code fix must first pass its focused test.
