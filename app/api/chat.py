from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import verify_token
from app.core.config import AiSettings, get_ai_settings
from app.services.llm import generate_ninerouter_chat_answer
from app.services.rag import answer_with_rag, build_query, select_history
from app.services.vector_store import search_course


router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    learner_id: str = Field(alias="learnerId")
    course_id: str = Field(alias="courseId")
    question: str = Field(min_length=1, max_length=1000)
    similarity_threshold: float = Field(alias="similarityThreshold", ge=0, le=1)
    history: list[ChatMessage] = Field(default_factory=list)


class ChatCitation(BaseModel):
    lesson_id: str | None = Field(default=None, alias="lessonId")
    lesson_title: str = Field(alias="lessonTitle")
    excerpt: str
    source_type: Literal["transcript", "material"] = Field(alias="sourceType")
    locator: str | None = None


class ChatResponse(BaseModel):
    answer: str
    in_scope: bool = Field(alias="inScope")
    similarity: float
    citations: list[ChatCitation]


@router.post("", response_model=ChatResponse, dependencies=[Depends(verify_token)])
async def chat(
    request: ChatRequest,
    settings: AiSettings = Depends(get_ai_settings),
) -> ChatResponse:
    history = select_history(request.question, request.history)
    matches = await search_course(
        settings,
        request.course_id,
        build_query(request.question, history),
        request.similarity_threshold,
    )
    result = answer_with_rag(
        question=request.question,
        similarity_threshold=request.similarity_threshold,
        history=history,
        matches=matches,
    )
    answer = result.answer

    if result.in_scope:
        generated_answer = await generate_ninerouter_chat_answer(
            question=request.question,
            history=history,
            citations=result.citations,
            settings=settings,
        )
        if generated_answer is not None:
            answer = generated_answer

    return ChatResponse(
        answer=answer,
        inScope=result.in_scope,
        similarity=result.similarity,
        citations=[
            ChatCitation(
                lessonId=citation.lesson_id,
                lessonTitle=citation.lesson_title,
                excerpt=citation.excerpt,
                sourceType=citation.source_type,
                locator=citation.locator,
            )
            for citation in result.citations
        ],
    )
