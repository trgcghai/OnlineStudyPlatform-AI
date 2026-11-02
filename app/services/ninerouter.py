import httpx

import urllib.parse

import httpx

from app.core.config import AiSettings


class NineRouterSummaryError(RuntimeError):
    pass


async def generate_ninerouter_text(
    prompt: str,
    settings: AiSettings,
    *,
    max_output_tokens: int,
) -> str:
    base_url = settings.ninerouter_url.strip().rstrip("/")
    model = settings.ninerouter_summary_model.strip()
    if not base_url or not model:
        raise NineRouterSummaryError(
            "9Router URL and summary model must be configured"
        )

    parsed_url = urllib.parse.urlparse(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise NineRouterSummaryError("9Router URL must use HTTP or HTTPS")

    headers: dict[str, str] = {}
    api_key = settings.ninerouter_api_key
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    request_json = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.2,
        "max_tokens": max_output_tokens,
    }
    try:
        async with httpx.AsyncClient(
            timeout=settings.ninerouter_summary_timeout_seconds
        ) as client:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                headers=headers,
                json=request_json,
            )
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, OSError) as error:
        raise NineRouterSummaryError("9Router summary request failed") from error
    except ValueError as error:
        raise NineRouterSummaryError(
            "9Router returned invalid summary JSON"
        ) from error

    if not isinstance(payload, dict):
        raise NineRouterSummaryError("9Router returned an invalid summary response")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise NineRouterSummaryError("9Router returned an invalid summary response")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise NineRouterSummaryError("9Router returned an invalid summary response")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise NineRouterSummaryError("9Router returned empty summary content")
    return content.strip()
