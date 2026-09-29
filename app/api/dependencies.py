from fastapi import Depends, Header, HTTPException, status

from app.core.config import AiSettings, get_ai_settings


def verify_token(
    authorization: str | None = Header(default=None),
    settings: AiSettings = Depends(get_ai_settings),
) -> None:
    if not settings.ai_token:
        return

    expected = f"Bearer {settings.ai_token}"
    if authorization != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid AI service token",
        )
