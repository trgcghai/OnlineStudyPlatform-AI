import httpx

import urllib.parse

import httpx

from app.core.config import AiSettings


class NineRouterSummaryError(RuntimeError):
    pass


class NineRouterChatError(RuntimeError):
    pass


async def _generate_ninerouter_text(
    prompt: str,
    settings: AiSettings,
    *,
    model: str,
    timeout_seconds: float,
    max_output_tokens: int,
    error_type: type[RuntimeError],
    operation: str,
) -> str:
    base_url = settings.ninerouter_url.strip().rstrip("/")
    model = model.strip()
    if not base_url or not model:
        raise error_type(f"9Router URL and {operation} model must be configured")

    parsed_url = urllib.parse.urlparse(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise error_type("9Router URL must use HTTP or HTTPS")

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
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                headers=headers,
                json=request_json,
            )
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, OSError) as error:
        raise error_type(f"9Router {operation} request failed") from error
    except ValueError as error:
        raise error_type(f"9Router returned invalid {operation} JSON") from error

    if not isinstance(payload, dict):
        raise error_type(f"9Router returned an invalid {operation} response")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise error_type(f"9Router returned an invalid {operation} response")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise error_type(f"9Router returned an invalid {operation} response")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise error_type(f"9Router returned empty {operation} content")
    return content.strip()


async def generate_ninerouter_text(
    prompt: str,
    settings: AiSettings,
    *,
    max_output_tokens: int,
) -> str:
    return await _generate_ninerouter_text(
        prompt,
        settings,
        model=settings.ninerouter_summary_model,
        timeout_seconds=settings.ninerouter_summary_timeout_seconds,
        max_output_tokens=max_output_tokens,
        error_type=NineRouterSummaryError,
        operation="summary",
    )


async def generate_ninerouter_chat_text(
    prompt: str,
    settings: AiSettings,
    *,
    max_output_tokens: int = 512,
) -> str:
    return await _generate_ninerouter_text(
        prompt,
        settings,
        model=settings.ninerouter_chat_model,
        timeout_seconds=settings.ninerouter_chat_timeout_seconds,
        max_output_tokens=max_output_tokens,
        error_type=NineRouterChatError,
        operation="chat",
    )
