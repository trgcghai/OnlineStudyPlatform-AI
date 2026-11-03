# 9Router Lesson Summary Design

## Intent

Move lesson-summary generation from direct Gemini API calls to the self-hosted 9Router chat-completions API. The goal is to use the free-provider routing already managed by 9Router while leaving the RAG chatbot on Gemini.

## Goals and boundaries

- Send summary prompts to `POST <NINEROUTER_URL>/v1/chat/completions` using the OpenAI-compatible JSON request/response shape.
- Keep the current transcript chunking, Vietnamese prompts, summary validation, `/summarize` request fields, and NestJS retry flow.
- Report `provider: "9router"` from the AI response and update the NestJS summary client to accept that provider value.
- Keep RAG/chat calls on the existing direct Gemini implementation.
- Do not add automatic fallback to Gemini for summaries, so a failed 9Router call cannot silently switch to another provider.
- Do not change transcription, Qdrant, transcript storage, or unrelated API behavior.

## Approaches considered

1. **Summary-only 9Router adapter (selected):** introduce a separate summary call using the existing 9Router URL/key and a dedicated summary model setting. RAG continues to use Gemini. This isolates the requested change and supports a free-only route.
2. **Selectable Gemini/9Router summary provider:** retain both summary implementations behind a setting. This is more flexible but keeps direct Gemini in the summary path and adds configuration the requested 9Router-only behavior does not need.

## Design

- Add `NINEROUTER_SUMMARY_MODEL` and `NINEROUTER_SUMMARY_TIMEOUT_SECONDS` (default 120 seconds per model call); reuse `NINEROUTER_URL` and optional `NINEROUTER_API_KEY`. Do not reuse `NINEROUTER_STT_MODEL` for text generation.
- Add an isolated 9Router text-generation adapter for summaries. Send non-streaming chat-completions messages and the current output-token limit; parse `choices[0].message.content` and convert HTTP, network, malformed-response, and empty-content failures into `SummarizationError`.
- Keep the summary prompts and validation unchanged. For long transcripts, each notes-generation chunk and the final synthesis call use the same configured 9Router summary model.
- Remove the obsolete `SUMMARY_PROVIDER` selector from AI configuration/examples/Compose because summaries have one provider after this change. Retain Gemini key/model settings for RAG.
- Update the NestJS summary response type/validator to accept `9router`. Keep the request contract unchanged. Set the NestJS `/summarize` client timeout to 15 minutes for sequential chunk calls; leave the transcription timeout unchanged.
- Update AI and root `.env.example` plus Compose wiring. Never edit the active `.env`.

## Acceptance checks

- Mocked tests verify the 9Router URL, authorization, model, message prompt, non-streaming request, response parsing, and upstream/error handling.
- Summary tests still enforce required headings, non-empty sections, and main-content bullets.
- Tests verify the RAG path still calls the Gemini adapter and is unaffected by summary-provider settings.
- NestJS accepts a `provider: "9router"` summary response while retaining its current response validation.
- Compose validation succeeds using `.env.example`; no active secrets are printed or staged.

## Implementation calls for review

- The dedicated summary model and timeout are separate from the transcription model and timeout because chat and audio routes have different latency and model requirements.
- `provider` will be the literal `"9router"` so stored/API behavior identifies the actual summary provider.
- The existing asynchronous BullMQ retry behavior remains the recovery path when 9Router fails; there is no provider fallback.
