import asyncio
from dataclasses import dataclass
from functools import lru_cache
from uuid import NAMESPACE_URL, uuid5

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from app.core.config import AiSettings
from app.services.knowledge_documents import DocumentChunk


@dataclass(frozen=True)
class KnowledgePayload:
    source_id: str
    course_id: str
    lesson_id: str
    lesson_title: str
    source_type: str
    content_hash: str


@dataclass(frozen=True)
class VectorSearchResult:
    score: float
    lesson_id: str
    lesson_title: str
    source_type: str
    locator: str | None
    text: str


class VectorKnowledgeStore:
    def __init__(self, settings: AiSettings) -> None:
        self.collection = settings.qdrant_collection
        self.client = QdrantClient(url=settings.qdrant_url)
        self.embedder = TextEmbedding(
            model_name=settings.embedding_model,
            cache_dir=settings.embedding_cache_dir,
        )

    def index(self, payload: KnowledgePayload, chunks: list[DocumentChunk]) -> int:
        texts = [f"{payload.lesson_title}\n{chunk.text}" for chunk in chunks]
        vectors = [vector.tolist() for vector in self.embedder.embed(texts)]
        if not vectors:
            raise RuntimeError("Embedding model returned no vectors")
        self._ensure_collection(len(vectors[0]))
        self.delete_source(payload.source_id)
        points = [
            models.PointStruct(
                id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"{payload.source_id}:{payload.content_hash}:{chunk.chunk_index}",
                    )
                ),
                vector=vector,
                payload={
                    "sourceId": payload.source_id,
                    "courseId": payload.course_id,
                    "lessonId": payload.lesson_id,
                    "lessonTitle": payload.lesson_title,
                    "sourceType": payload.source_type,
                    "contentHash": payload.content_hash,
                    "locator": chunk.locator,
                    "chunkIndex": chunk.chunk_index,
                    "text": chunk.text,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    def search(
        self,
        course_id: str,
        query: str,
        limit: int,
        score_threshold: float,
    ) -> list[VectorSearchResult]:
        if not self.client.collection_exists(self.collection):
            return []
        vector = next(iter(self.embedder.query_embed(query))).tolist()
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="courseId", match=models.MatchValue(value=course_id)
                    )
                ]
            ),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        results: list[VectorSearchResult] = []
        for point in response.points:
            payload = point.payload or {}
            text = payload.get("text")
            lesson_id = payload.get("lessonId")
            lesson_title = payload.get("lessonTitle")
            source_type = payload.get("sourceType")
            if not all(isinstance(value, str) for value in [text, lesson_id, lesson_title, source_type]):
                continue
            locator = payload.get("locator")
            results.append(
                VectorSearchResult(
                    score=float(point.score),
                    lesson_id=lesson_id,
                    lesson_title=lesson_title,
                    source_type=source_type,
                    locator=locator if isinstance(locator, str) else None,
                    text=text,
                )
            )
        return results

    def delete_source(self, source_id: str) -> None:
        if not self.client.collection_exists(self.collection):
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="sourceId", match=models.MatchValue(value=source_id)
                        )
                    ]
                )
            ),
            wait=True,
        )

    def delete_course(self, course_id: str) -> None:
        if not self.client.collection_exists(self.collection):
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="courseId", match=models.MatchValue(value=course_id)
                        )
                    ]
                )
            ),
            wait=True,
        )

    def _ensure_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        )


@lru_cache
def _cached_store(
    qdrant_url: str,
    collection: str,
    model: str,
    cache_dir: str,
) -> VectorKnowledgeStore:
    return VectorKnowledgeStore(
        AiSettings(
            QDRANT_URL=qdrant_url,
            QDRANT_COLLECTION=collection,
            EMBEDDING_MODEL=model,
            EMBEDDING_CACHE_DIR=cache_dir,
        )
    )


def get_vector_store(settings: AiSettings) -> VectorKnowledgeStore:
    return _cached_store(
        settings.qdrant_url,
        settings.qdrant_collection,
        settings.embedding_model,
        settings.embedding_cache_dir,
    )


async def index_chunks(
    settings: AiSettings,
    payload: KnowledgePayload,
    chunks: list[DocumentChunk],
) -> int:
    return await asyncio.to_thread(get_vector_store(settings).index, payload, chunks)


async def search_course(
    settings: AiSettings,
    course_id: str,
    query: str,
    score_threshold: float,
) -> list[VectorSearchResult]:
    return await asyncio.to_thread(
        get_vector_store(settings).search,
        course_id,
        query,
        settings.rag_top_k,
        score_threshold,
    )


async def delete_source(settings: AiSettings, source_id: str) -> None:
    await asyncio.to_thread(get_vector_store(settings).delete_source, source_id)


async def delete_course(settings: AiSettings, course_id: str) -> None:
    await asyncio.to_thread(get_vector_store(settings).delete_course, course_id)
