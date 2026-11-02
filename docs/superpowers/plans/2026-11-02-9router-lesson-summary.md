# 9Router Lesson Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route lesson-summary generation through 9Router while keeping the RAG chatbot on direct Gemini calls.

**Architecture:** Add a dedicated asynchronous 9Router chat-completions adapter for lesson summaries and keep the existing Gemini helper for RAG. Preserve the summary prompts, chunking, and validation; report `provider: "9router"` and update the NestJS response validation and timeout for this summary path.

**Tech Stack:** Python 3.13, FastAPI, Pydantic Settings, HTTPX, pytest/unittest, NestJS 11, TypeScript, Jest, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-11-02-9router-summary-design.md`

## Global Constraints

- Only lesson-summary generation uses 9Router; RAG/chat generation continues using the existing Gemini implementation.
- Summary calls use `POST <NINEROUTER_URL>/v1/chat/completions`, `NINEROUTER_SUMMARY_MODEL`, optional Bearer auth, non-streaming OpenAI-compatible messages, and `SUMMARY_MAX_OUTPUT_TOKENS` as `max_tokens`.
- `NINEROUTER_SUMMARY_TIMEOUT_SECONDS` defaults to 120 seconds per model call; NestJS `/summarize` timeout defaults to 15 minutes.
- Do not add direct Gemini fallback for summaries.
- Keep transcript chunking, Vietnamese prompts, validation, `/summarize` request fields, and BullMQ retry behavior.
- Keep transcription, Qdrant, transcript storage, and unrelated behavior unchanged.
- Never edit the active `.env`; update only `.env.example` files and Compose wiring.
- Follow each repo's branch rules; preserve existing workspace changes and commit code within its own submodule before updating root pointers.

## Review Focus

1. **Router/network/HTTP failure:** convert non-2xx and transport failures into a summary error with no Gemini fallback; mock a 503 and a timeout in the adapter test.
2. **Malformed or empty chat response:** reject missing choices, missing message content, or blank content; cover each in adapter tests.
3. **Long transcript:** call the router once for each notes chunk and once for final synthesis; assert number/order of prompts and output-token setting in summary tests.
4. **Summary format rejection:** keep exact headings, non-empty sections, and main-content bullets; retain/add validation tests for malformed headings, empty sections, and missing bullets.
5. **RAG isolation:** RAG still calls `generate_gemini_text` and does not depend on summary model/provider settings; add a focused LLM test while leaving `app/services/llm.py` behavior unchanged.

---

### Task 1: Add a dedicated 9Router summary adapter and settings

**Files:**
- Create: `app/services/ninerouter.py`
- Modify: `app/core/config.py`
- Create: `tests/test_ninerouter.py`
- Modify: `tests/test_settings.py`

**Interfaces:**
- Add `AiSettings.ninerouter_summary_model: str` from `NINEROUTER_SUMMARY_MODEL`.
- Add `AiSettings.ninerouter_summary_timeout_seconds: float` from `NINEROUTER_SUMMARY_TIMEOUT_SECONDS`, default `120`, greater than zero.
- Define `NineRouterSummaryError(RuntimeError)` in `app/services/ninerouter.py` for adapter failures.
- Create `async def generate_ninerouter_text(prompt: str, settings: AiSettings, *, max_output_tokens: int) -> str`.
- The adapter posts to `<NINEROUTER_URL>/v1/chat/completions` with `model`, `messages=[{"role":"user","content":prompt}]`, `stream=false`, `temperature=0.2`, and `max_tokens=max_output_tokens`; it returns non-empty `choices[0].message.content` or raises `NineRouterSummaryError` for HTTP, network, JSON, or content failures.

- [x] **Step 1: Write adapter tests** asserting exact URL, Bearer header when configured, omitted auth header when absent, request JSON fields, configured timeout, string content parsing, missing URL/model rejection before HTTP, and failures for non-2xx, timeout, malformed JSON, missing choices/content, and blank content.
- [x] **Step 2: Write settings tests** for the new env aliases, default timeout `120`, and rejection of zero/negative timeout.
- [x] **Step 3: Run adapter and settings tests and confirm they fail** using `uv run python -m pytest tests/test_ninerouter.py tests/test_settings.py -q -p no:cacheprovider` from `online-study-platform-AI/`.
- [x] **Step 4: Implement the settings and `generate_ninerouter_text` adapter** in the specified files; use an HTTPX async JSON POST (no file upload) and do not alter `app/services/llm.py`.
- [x] **Step 5: Run adapter and settings tests** with the same command; expect all selected tests to pass.
- [x] **Step 6: Commit AI adapter/config/tests** with `feat: add 9router summary adapter`.

### Task 2: Route lesson summaries through 9Router and preserve RAG on Gemini

**Files:**
- Modify: `app/services/summarization.py`
- Modify: `tests/test_summary.py`
- Create: `tests/test_llm.py`

**Interfaces:**
- Rename `summarize_with_gemini` to `summarize_with_ninerouter`; it calls `generate_ninerouter_text` for every notes chunk and final summary.
- `summarize_transcript(title, transcript, settings) -> SummaryResult` returns `provider="9router"`.
- Keep `app/services/llm.py` and `generate_gemini_answer` behavior unchanged for RAG.

- [x] **Step 1: Update summary tests to expect 9Router**; mock the new adapter for one-chunk and multi-chunk transcripts and assert calls use the final-summary prompt and every notes prompt in order with `summary_max_output_tokens`; update the `/summarize` API test to expect `provider: "9router"`.
- [x] **Step 2: Add a test that summary adapter failures become `SummarizationError`** and that no Gemini helper is called as fallback.
- [x] **Step 3: Add/adjust tests for empty transcript and required summary structure failures** (missing/out-of-order headings, empty section, missing main-content bullets, transcript echoed back).
- [x] **Step 4: Add `tests/test_llm.py` coverage that `generate_gemini_answer` still delegates to the Gemini helper**, proving summary routing did not change RAG behavior.
- [x] **Step 5: Run the summary and LLM tests and confirm new expectations fail before implementation** using `uv run python -m pytest tests/test_summary.py tests/test_llm.py -q -p no:cacheprovider`.
- [x] **Step 6: Replace the summary path with `generate_ninerouter_text`**, retain chunking/prompts/validation, translate `NineRouterSummaryError` to `SummarizationError("9Router could not generate a summary")`, remove the `SUMMARY_PROVIDER`/Gemini-key guard from summary generation, and return provider `9router`.
- [x] **Step 7: Run focused summary/LLM tests** with the same command; expect all tests to pass, including existing `/chat` behavior tests if run.
- [x] **Step 8: Commit AI summary integration/tests** with `feat: generate lesson summaries through 9router`.

### Task 3: Update the NestJS summary response contract and timeout

**Files:**
- Modify: `../online-study-platform-be/src/modules/ai-summary/dto/ai-summary.dto.ts`
- Modify: `../online-study-platform-be/src/modules/ai-summary/ai-summary.client.ts`
- Modify: `../online-study-platform-be/src/modules/ai-summary/ai-summary.constants.ts`
- Modify: `../online-study-platform-be/src/modules/ai-summary/ai-summary.client.spec.ts`

**Interfaces:**
- `AiSummarizeResponse.provider` accepts the literal `"9router"`.
- `DEFAULT_AI_SUMMARIZE_TIMEOUT_MS` becomes `900_000` (15 minutes); leave `DEFAULT_AI_TRANSCRIBE_TIMEOUT_MS` unchanged.

- [x] **Step 1: Add a client test accepting `{ summary: "...", provider: "9router" }` and update the invalid-response test to continue rejecting empty summary/unknown provider values.
- [x] **Step 2: Assert the `/summarize` request uses the 900,000 ms timeout and `/transcribe` retains its current timeout.
- [ ] **Step 3: Run the focused NestJS test** from `online-study-platform-be/` using `pnpm test --runInBand -- ai-summary.client.spec.ts`; expect it to fail before the DTO/timeout update.
- [x] **Step 4: Update the response type/validator and summary timeout only**; do not alter the transcription client or RAG API.
- [ ] **Step 5: Re-run the focused NestJS test** and expect it to pass.
- [x] **Step 6: Commit backend changes** with `feat: accept 9router lesson summary responses`.

### Task 4: Wire summary configuration and verify deployment

**Files:**
- Modify: `.env.example` (AI repo)
- Modify: `README.md` (AI repo)
- Modify: `../.env.example` (workspace root)
- Modify: `../docker-compose.yml`
- Modify: `app/core/config.py`
- Modify: `tests/test_settings.py`

**Interfaces:**
- Add `NINEROUTER_SUMMARY_MODEL=` and `NINEROUTER_SUMMARY_TIMEOUT_SECONDS=120` to both examples.
- Pass both summary settings to the AI Compose service.
- Remove `SUMMARY_PROVIDER` from AI settings/examples/Compose; retain `GEMINI_API_KEY`, `GEMINI_MODEL`, and `LLM_PROVIDER` for RAG.

- [x] **Step 1: Update AI/root environment examples and AI README**, removing only the obsolete summary-provider selector and adding the dedicated summary model/timeout entries; document that lesson summaries use 9Router and preserve Gemini settings/documentation for RAG.
- [x] **Step 2: Update root Compose** to pass the two summary settings only to the AI service and remove the old `SUMMARY_PROVIDER` mapping.
- [x] **Step 3: Verify Compose without printing resolved values** using `docker compose --env-file .env.example config -q` from the workspace root.
- [x] **Step 4: Search AI/root source and examples** to ensure summary code no longer depends on Gemini while RAG still uses `generate_gemini_answer`; confirm there are no active `.env` edits.
- [ ] **Step 5: Run AI suite** using `uv run python -m unittest discover tests` from the AI repo (with `AI_TOKEN` cleared only for the test process if the local `.env` affects unrelated tests), then rerun focused NestJS coverage.
- [x] **Step 6: Commit AI example/config changes**, then commit root example/Compose changes and the AI/BE submodule pointers on the existing root feature branch; stage only intended hunks so pre-existing root edits remain unstaged.

## Execution Notes

- Implement the adapter and summary refactor only inside the AI submodule; keep `app/services/llm.py` unchanged so `/chat` remains Gemini-backed.
- The summary model ID must be one returned by the deployed 9Router chat model list; it is independent of the STT model ID.
- Never edit or display the active `.env` values. Compose validation uses `.env.example` and `config -q` only.
- Before execution, verify AI, backend, and root branches are not `main`; preserve all pre-existing changes and do not use git worktrees.

### Execution results

- AI suite passed: 104 tests. Compose validation passed with warnings for unrelated optional variables missing from `.env.example`.
- NestJS focused tests could not execute because the BE worktree has no installed Jest binary and pnpm package fetches failed with `ECONNREFUSED`. The test changes cover 9Router provider acceptance, unknown-provider rejection, a 900,000 ms summary timeout, and the unchanged 1,800,000 ms transcription timeout.
- AI/BE submodule changes and root config/pointer changes are committed. The active `.env` was not edited.
