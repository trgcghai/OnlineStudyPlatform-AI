# 9Router Model Fallback Design

## Intent

Allow the AI service to recover when a configured 9Router model route is
unavailable, without changing the existing chat, summary, or transcription API
contracts.

## Goals

- Accept ordered comma-separated model IDs in `NINEROUTER_CHAT_MODEL`,
  `NINEROUTER_SUMMARY_MODEL`, and `NINEROUTER_STT_MODEL`.
- Try configured models sequentially for each 9Router request.
- Fall back immediately on HTTP 5xx and 429 responses, and also on network
  errors, timeouts, malformed responses, and empty output.
- For lesson summaries, treat failure of required summary-structure validation
  as an unusable model response and try the next model.
- Fail fast on other HTTP 4xx responses because they usually indicate a shared
  authentication or request-configuration problem.
- Preserve the current API error contracts and use the existing single-model
  environment values unchanged.

## Non-goals

- Changing the 9Router endpoint, credentials, prompts, output schemas, or
  provider.
- Adding local-model or alternate-provider fallback.
- Changing queue retry behavior or public API response shapes.
- Modifying Compose model-variable wiring; Compose already passes these values
  through as strings.

## Approaches

1. **Share the existing timeout budget across candidates (selected):** divide
   each configured `NINEROUTER_*_TIMEOUT_SECONDS` budget evenly among the model
   IDs. This gives every candidate time to run while retaining the current
   maximum timeout for one 9Router call and avoids requiring larger fixed
   NestJS request timeouts.
2. **Give each candidate the full existing timeout:** preserves per-model
   latency but multiplies the maximum request duration by the candidate count;
   existing NestJS request deadlines could abort before fallback finishes.
3. **Configure fallback only inside 9Router:** delegates selection to the
   router, but does not support the requested application-level comma-separated
   environment list.

## Design

Parse each model setting as an ordered comma-separated list. Trim surrounding
whitespace, ignore empty items, and avoid retrying duplicate IDs. A single
non-empty value continues to behave as it does today. If no model remains,
raise the existing configuration error without making a request.

For chat and summaries, submit the existing OpenAI-compatible chat-completions
request once per candidate until one returns a usable non-empty
`choices[0].message.content`. For transcription, upload the same audio to the
existing transcription endpoint once per candidate until it returns a valid
non-empty `text` field. Continue on HTTP 5xx or 429, network/timeout failures,
invalid JSON or response shapes, and empty content. Other HTTP 4xx errors fail
immediately. A 5xx/429 response advances without waiting for a timeout; a
timeout has no HTTP status and advances only after that candidate's share of
the configured total timeout expires.

For the final lesson-summary response, run the existing structural validator
as part of candidate acceptance. A candidate missing required headings,
sections, or main-content bullets is rejected and the next model is tried.
Notes-generation calls keep their existing non-empty-content requirement.
Once candidates are exhausted, raise the existing domain error so the current
AI API and NestJS queue/error handling remain unchanged.

Update the AI and root `.env.example` comments and the AI README to document
comma-separated values, order, and timeout budgeting. Do not edit active
`.env` files. Keep Docker Compose unchanged because its current environment
forwarding supports list strings already.

## Testing

- Verify settings parse single and comma-separated values, trimming whitespace,
  dropping empty entries and duplicates, and preserving order.
- Verify chat, summary, and transcription request each candidate in order,
  changing only the model field for each attempt.
- Verify immediate fallback on 5xx/429, fallback on network/timeouts and
  malformed/empty responses, and fail-fast behavior on other 4xx responses.
- Verify summary structural validation failure advances to the next candidate
  and that exhaustion still raises the existing summary error.
- Verify the configured total timeout is shared across candidates and that a
  single-model setting retains the existing timeout.
- Verify existing one-model behavior and API response contracts remain intact.

## Constraints

- Keep all current prompts, response validation rules, endpoints, and
  authorization behavior unchanged.
- Never include upstream error bodies, API keys, or credentials in logs or
  client-visible errors.
- Keep each fallback list scoped to its corresponding operation; chat,
  summaries, and transcription may use different model types.
