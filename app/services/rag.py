import re
from collections import Counter
from dataclasses import dataclass
from math import sqrt
from typing import Protocol


OUT_OF_SCOPE_REPLY = (
    "Xin loi, cau hoi nay nam ngoai noi dung kien thuc cua khoa hoc. "
    "Ban hay hoi ve bai hoc, tai lieu hoac noi dung trong khoa hoc nay."
)

CHUNK_TOKEN_LIMIT = 90
CHUNK_TOKEN_OVERLAP = 18
MAX_CITATIONS = 3


class RagSource(Protocol):
    lesson_id: str | None
    lesson_title: str
    content: str


class RagHistoryMessage(Protocol):
    role: str
    content: str


@dataclass(frozen=True)
class RagChunk:
    lesson_id: str | None
    lesson_title: str
    content: str
    searchable_text: str


@dataclass(frozen=True)
class RagCitation:
    lesson_id: str | None
    lesson_title: str
    excerpt: str


@dataclass(frozen=True)
class RagResult:
    answer: str
    in_scope: bool
    similarity: float
    citations: list[RagCitation]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)


def cosine_similarity(left: str, right: str) -> float:
    left_counts = Counter(tokenize(left))
    right_counts = Counter(tokenize(right))

    if not left_counts or not right_counts:
        return 0.0

    common = set(left_counts).intersection(right_counts)
    dot_product = sum(left_counts[token] * right_counts[token] for token in common)
    left_norm = sqrt(sum(count * count for count in left_counts.values()))
    right_norm = sqrt(sum(count * count for count in right_counts.values()))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def excerpt(text: str, max_length: int = 220) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 3].rstrip()}..."


def chunk_source(source: RagSource) -> list[RagChunk]:
    normalized = " ".join(source.content.split())
    tokens = tokenize(normalized)

    if not tokens:
        return []

    if len(tokens) <= CHUNK_TOKEN_LIMIT:
        return [
            RagChunk(
                lesson_id=source.lesson_id,
                lesson_title=source.lesson_title,
                content=normalized,
                searchable_text=f"{source.lesson_title}\n{normalized}",
            )
        ]

    chunks: list[RagChunk] = []
    words = normalized.split()
    start = 0
    step = CHUNK_TOKEN_LIMIT - CHUNK_TOKEN_OVERLAP

    while start < len(words):
        chunk_text = " ".join(words[start : start + CHUNK_TOKEN_LIMIT])
        chunks.append(
            RagChunk(
                lesson_id=source.lesson_id,
                lesson_title=source.lesson_title,
                content=chunk_text,
                searchable_text=f"{source.lesson_title}\n{chunk_text}",
            )
        )
        start += step

    return chunks


def build_query(question: str, history: list[RagHistoryMessage]) -> str:
    recent_user_messages = [
        message.content
        for message in history[-6:]
        if message.role == "user" and message.content.strip()
    ]

    if not recent_user_messages:
        return question

    return "\n".join([*recent_user_messages[-2:], question])


def retrieve_chunks(
    question: str,
    history: list[RagHistoryMessage],
    sources: list[RagSource],
    top_k: int = MAX_CITATIONS,
) -> list[tuple[float, RagChunk]]:
    query = build_query(question, history)
    chunks = [chunk for source in sources for chunk in chunk_source(source)]
    ranked_chunks = sorted(
        (
            (cosine_similarity(query, chunk.searchable_text), chunk)
            for chunk in chunks
        ),
        key=lambda item: item[0],
        reverse=True,
    )

    return ranked_chunks[:top_k]


def build_citations(ranked_chunks: list[tuple[float, RagChunk]]) -> list[RagCitation]:
    citations: list[RagCitation] = []
    seen: set[tuple[str | None, str]] = set()

    for similarity, chunk in ranked_chunks:
        if similarity <= 0:
            continue

        key = (chunk.lesson_id, excerpt(chunk.content))
        if key in seen:
            continue

        seen.add(key)
        citations.append(
            RagCitation(
                lesson_id=chunk.lesson_id,
                lesson_title=chunk.lesson_title,
                excerpt=excerpt(chunk.content),
            )
        )

    return citations


def build_answer(question: str, ranked_chunks: list[tuple[float, RagChunk]], has_history: bool) -> str:
    context_lines = [
        f"- Bai \"{chunk.lesson_title}\": {excerpt(chunk.content, 320)}"
        for similarity, chunk in ranked_chunks
        if similarity > 0
    ]
    context = "\n".join(context_lines)
    prefix = "Dua tren ngu canh trao doi truoc va tai lieu khoa hoc" if has_history else "Dua tren tai lieu khoa hoc"

    return (
        f"{prefix}, noi dung lien quan nhat cho cau hoi \"{question}\" la:\n"
        f"{context}\n\n"
        "Hay doi chieu cau tra loi voi cac trich dan kem theo va uu tien noi dung trong khoa hoc."
    )


def answer_with_rag(
    question: str,
    similarity_threshold: float,
    history: list[RagHistoryMessage],
    sources: list[RagSource],
) -> RagResult:
    if not sources:
        return RagResult(
            answer=OUT_OF_SCOPE_REPLY,
            in_scope=False,
            similarity=0,
            citations=[],
        )

    ranked_chunks = retrieve_chunks(question, history, sources)
    best_similarity = ranked_chunks[0][0] if ranked_chunks else 0

    if best_similarity < similarity_threshold:
        return RagResult(
            answer=OUT_OF_SCOPE_REPLY,
            in_scope=False,
            similarity=round(best_similarity, 4),
            citations=[],
        )

    citations = build_citations(ranked_chunks)

    return RagResult(
        answer=build_answer(question, ranked_chunks, bool(history)),
        in_scope=True,
        similarity=round(best_similarity, 4),
        citations=citations,
    )
