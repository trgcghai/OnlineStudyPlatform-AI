# 9Router Chat Answer Generation Design

## Intent

Use the existing 9Router service to generate chatbot answers, replacing direct Gemini calls. Keep the course RAG retrieval and citation flow intact.

## Goals

- Generate grounded chatbot answers through the OpenAI-compatible 9Router chat-completions endpoint.
- Give chat its own `NINEROUTER_CHAT_MODEL` and `NINEROUTER_CHAT_TIMEOUT_SECONDS` settings while reusing `NINEROUTER_URL` and optional `NINEROUTER_API_KEY`.
- Remove obsolete Gemini chat settings (`LLM_PROVIDER`, `GEMINI_API_KEY`, `GEMINI_MODEL`, and `GEMINI_TIMEOUT_SECONDS`) from AI configuration, environment examples, Compose wiring, and user-facing configuration documentation.
- Keep prompt safety instructions, selected history, retrieved full chunk context, response schema, citations, scope threshold, and out-of-scope response behavior.
- Keep failure handling explicit: a 9Router failure or invalid response does not fall back to Gemini; the AI endpoint fails and the existing NestJS client returns its unavailable response.

## Non-goals

- Changing embeddings, Qdrant, chunking, retrieval ranking, enrollment checks, or the NestJS chat contract.
- Changing lesson summary or transcription routing and their model/timeout settings.
- Adding provider selection or a second chat provider.

## Approaches

1. **Dedicated 9Router chat model and timeout (chosen):** reuse the established router URL/key and add chat-specific settings. This lets chatbot and summary requests be tuned independently.
2. **Share summary model and timeout:** minimizes settings but couples chat latency/model changes to lesson summaries.

## Design

Replace the Gemini answer adapter with a chat adapter that sends the existing grounded prompt as a non-streaming `POST <NINEROUTER_URL>/v1/chat/completions` request. Use `NINEROUTER_CHAT_MODEL`, the existing optional Bearer key, temperature 0.2, and a positive timeout from `NINEROUTER_CHAT_TIMEOUT_SECONDS` (proposed default: 120 seconds). Parse non-empty `choices[0].message.content`; treat transport, HTTP, JSON, and content errors as generation failures. Only call the adapter when retrieval is in scope. Generation failures propagate to NestJS's existing unavailable-message handling; there is no Gemini fallback.

Update the AI `.env.example`, root `.env.example`, root `docker-compose.yml`, and AI README for the new chat settings and removal of Gemini-only chat variables. Do not edit a local `.env`. Add focused tests for 9Router request/response handling, grounded prompt/context, and error propagation while retaining the existing retrieval and citation tests.

## Constraints and open implementation details

- `NINEROUTER_URL` may include a trailing slash; endpoint joining must avoid duplicate separators.
- The chat response contract remains `{ answer, inScope, similarity, citations }`.
- NestJS already treats non-success/invalid AI chat responses as unavailable and persists its fallback message.
- The 120-second default is a design default; the implementation should validate the setting as greater than zero.
