from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.api.dependencies import verify_token
from app.core.config import AiSettings, get_ai_settings
from app.services.summarization import SummarizationError, summarize_transcript
from app.services.transcription import TranscriptionError, transcribe_audio


router = APIRouter(tags=["summary"])


class TranscribeRequest(BaseModel):
    lesson_id: str = Field(alias="lessonId", min_length=1, max_length=100)
    audio_url: str = Field(alias="audioUrl", min_length=1, max_length=2048)


class TranscribeResponse(BaseModel):
    transcript: str
    language: str
    duration_seconds: float = Field(alias="durationSeconds")


class SummarizeRequest(BaseModel):
    lesson_id: str = Field(alias="lessonId", min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=200)
    transcript: str = Field(min_length=1)


class SummarizeResponse(BaseModel):
    summary: str
    provider: str


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    dependencies=[Depends(verify_token)],
)
async def transcribe(
    request: TranscribeRequest,
    settings: AiSettings = Depends(get_ai_settings),
) -> TranscribeResponse:
    try:
        result = await transcribe_audio(request.audio_url, settings)
    except TranscriptionError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    return TranscribeResponse(
        transcript=result.transcript,
        language=result.language,
        durationSeconds=result.duration_seconds,
    )


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    dependencies=[Depends(verify_token)],
)
async def summarize(
    request: SummarizeRequest,
    settings: AiSettings = Depends(get_ai_settings),
) -> SummarizeResponse:
    try:
        result = await summarize_transcript(
            request.title,
            request.transcript,
            settings,
        )
    except SummarizationError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    return SummarizeResponse(summary=result.summary, provider=result.provider)
