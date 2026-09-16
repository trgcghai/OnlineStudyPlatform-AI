import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request

from app.core.config import AiSettings
from app.services.rag import RagCitation, RagHistoryMessage


def build_gemini_prompt(
    question: str,
    history: list[RagHistoryMessage],
    citations: list[RagCitation],
) -> str:
    history_text = "\n".join(
        f"{message.role}: {message.content}" for message in history[-6:]
    )
    context_text = "\n".join(
        f"[{index + 1}] {citation.lesson_title}: {citation.excerpt}"
        for index, citation in enumerate(citations)
    )

    return (
        "Ban la tro ly AI cho nen tang hoc truc tuyen. "
        "Chi tra loi dua tren CONTEXT cua khoa hoc. "
        "Neu CONTEXT khong du de tra loi, hay noi ro rang rang tai lieu chua du thong tin. "
        "Khong tu bia kien thuc ngoai khoa hoc. Tra loi bang tieng Viet, ngan gon, de hieu.\n\n"
        f"HISTORY:\n{history_text or 'Khong co'}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:"
    )


def parse_gemini_text(payload: dict) -> str | None:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None

    content = candidates[0].get("content")
    if not isinstance(content, dict):
        return None

    parts = content.get("parts")
    if not isinstance(parts, list):
        return None

    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    normalized = text.strip()
    return normalized or None


def call_gemini(prompt: str, settings: AiSettings) -> str | None:
    if settings.llm_provider.casefold() != "gemini" or not settings.gemini_api_key:
        return None

    model = urllib.parse.quote(settings.gemini_model, safe="")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={settings.gemini_api_key}"
    )
    body = json.dumps(
        {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.8,
                "maxOutputTokens": 512,
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=settings.gemini_timeout_seconds,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
        return None

    return parse_gemini_text(payload)


async def generate_gemini_answer(
    question: str,
    history: list[RagHistoryMessage],
    citations: list[RagCitation],
    settings: AiSettings,
) -> str | None:
    if not citations:
        return None

    prompt = build_gemini_prompt(question, history, citations)
    return await asyncio.to_thread(call_gemini, prompt, settings)
