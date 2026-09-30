import asyncio
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from app.api.dependencies import verify_token
from app.core.config import AiSettings, get_ai_settings
from app.services.knowledge_documents import (
    KnowledgeDocumentError,
    chunk_transcript,
    extract_and_chunk_url,
)
from app.services.vector_store import (
    KnowledgePayload,
    delete_course,
    delete_source,
    index_chunks,
)


router = APIRouter(prefix="/knowledge", tags=["knowledge"])


class KnowledgeIndexRequest(BaseModel):
    source_id: str = Field(alias="sourceId")
    course_id: str = Field(alias="courseId")
    lesson_id: str = Field(alias="lessonId")
    lesson_title: str = Field(alias="lessonTitle", min_length=1, max_length=255)
    source_type: Literal["transcript", "material"] = Field(alias="sourceType")
    content_hash: str = Field(alias="contentHash", min_length=64, max_length=64)
    source_url: str | None = Field(default=None, alias="sourceUrl")
    transcript: str | None = None

    @model_validator(mode="after")
    def validate_source(self):
        if self.source_type == "material" and not self.source_url:
            raise ValueError("sourceUrl is required for material")
        if self.source_type == "transcript" and not self.transcript:
            raise ValueError("transcript is required for transcript source")
        return self


class KnowledgeIndexResponse(BaseModel):
    source_id: str = Field(alias="sourceId")
    content_hash: str = Field(alias="contentHash")
    chunk_count: int = Field(alias="chunkCount")


class KnowledgeDeleteResponse(BaseModel):
    deleted: bool


@router.post(
    "/index",
    response_model=KnowledgeIndexResponse,
    dependencies=[Depends(verify_token)],
)
async def index_knowledge(
    request: KnowledgeIndexRequest,
    settings: AiSettings = Depends(get_ai_settings),
) -> KnowledgeIndexResponse:
    try:
        if request.source_type == "transcript":
            chunks = chunk_transcript(request.transcript or "", settings)
        else:
            chunks = await asyncio.to_thread(
                extract_and_chunk_url, request.source_url or "", settings
            )
        count = await index_chunks(
            settings,
            KnowledgePayload(
                source_id=request.source_id,
                course_id=request.course_id,
                lesson_id=request.lesson_id,
                lesson_title=request.lesson_title,
                source_type=request.source_type,
                content_hash=request.content_hash,
            ),
            chunks,
        )
    except KnowledgeDocumentError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge indexing is unavailable",
        ) from error
    return KnowledgeIndexResponse(
        sourceId=request.source_id,
        contentHash=request.content_hash,
        chunkCount=count,
    )


@router.delete(
    "/sources/{source_id}",
    response_model=KnowledgeDeleteResponse,
    dependencies=[Depends(verify_token)],
)
async def delete_knowledge_source(
    source_id: str,
    settings: AiSettings = Depends(get_ai_settings),
) -> KnowledgeDeleteResponse:
    try:
        await delete_source(settings, source_id)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge deletion is unavailable",
        ) from error
    return KnowledgeDeleteResponse(deleted=True)


@router.delete(
    "/courses/{course_id}",
    response_model=KnowledgeDeleteResponse,
    dependencies=[Depends(verify_token)],
)
async def delete_course_knowledge(
    course_id: str,
    settings: AiSettings = Depends(get_ai_settings),
) -> KnowledgeDeleteResponse:
    try:
        await delete_course(settings, course_id)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge deletion is unavailable",
        ) from error
    return KnowledgeDeleteResponse(deleted=True)
