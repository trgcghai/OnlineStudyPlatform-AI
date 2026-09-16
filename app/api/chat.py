from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import AiSettings, get_ai_settings
from app.services.llm import generate_gemini_answer
from app.services.rag import answer_with_rag


router = APIRouter(prefix="/chat", tags=["chat"])

class ChatSource(BaseModel):
    lesson_id: str | None = Field(default=None, alias="lessonId")
    lesson_title: str = Field(alias="lessonTitle")
    content: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    learner_id: str = Field(alias="learnerId")
    course_id: str = Field(alias="courseId")
    question: str = Field(min_length=1, max_length=1000)
    similarity_threshold: float = Field(alias="similarityThreshold", ge=0, le=1)
    history: list[ChatMessage] = Field(default_factory=list)
    sources: list[ChatSource]


class ChatCitation(BaseModel):
    lesson_id: str | None = Field(default=None, alias="lessonId")
    lesson_title: str = Field(alias="lessonTitle")
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    in_scope: bool = Field(alias="inScope")
    similarity: float
    citations: list[ChatCitation]


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


@router.post("", response_model=ChatResponse, dependencies=[Depends(verify_token)])
async def chat(
    request: ChatRequest,
    settings: AiSettings = Depends(get_ai_settings),
) -> ChatResponse:
    result = answer_with_rag(
        question=request.question,
        similarity_threshold=request.similarity_threshold,
        history=request.history,
        sources=request.sources,
    )
    answer = result.answer

    if result.in_scope:
        generated_answer = await generate_gemini_answer(
            question=request.question,
            history=request.history,
            citations=result.citations,
            settings=settings,
        )
        answer = generated_answer or answer

    return ChatResponse(
        answer=answer,
        inScope=result.in_scope,
        similarity=result.similarity,
        citations=[
            ChatCitation(
                lessonId=citation.lesson_id,
                lessonTitle=citation.lesson_title,
                excerpt=citation.excerpt,
            )
            for citation in result.citations
        ],
    )
