from dataclasses import dataclass
from typing import Protocol


OUT_OF_SCOPE_REPLY = (
    "Xin lỗi, câu hỏi này nằm ngoài nội dung kiến thức của khóa học. "
    "Bạn hãy hỏi về bài học, tài liệu hoặc nội dung trong khóa học này."
)


class RagHistoryMessage(Protocol):
    role: str
    content: str


class RagSearchMatch(Protocol):
    score: float
    lesson_id: str
    lesson_title: str
    source_type: str
    locator: str | None
    text: str


@dataclass(frozen=True)
class RagCitation:
    lesson_id: str
    lesson_title: str
    excerpt: str
    source_type: str
    locator: str | None


@dataclass(frozen=True)
class RagResult:
    answer: str
    in_scope: bool
    similarity: float
    citations: list[RagCitation]


def excerpt(text: str, max_length: int = 360) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 3].rstrip()}..."


def build_query(question: str, history: list[RagHistoryMessage]) -> str:
    recent_user_messages = [
        message.content.strip()
        for message in history[-6:]
        if message.role == "user" and message.content.strip()
    ]
    if not recent_user_messages:
        return question.strip()
    return "\n".join([*recent_user_messages[-2:], question.strip()])


def build_citations(matches: list[RagSearchMatch]) -> list[RagCitation]:
    citations: list[RagCitation] = []
    seen: set[tuple[str, str, str | None]] = set()
    for match in matches:
        key = (match.lesson_id, match.source_type, match.locator)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            RagCitation(
                lesson_id=match.lesson_id,
                lesson_title=match.lesson_title,
                excerpt=excerpt(match.text),
                source_type=match.source_type,
                locator=match.locator,
            )
        )
    return citations


def answer_with_rag(
    question: str,
    similarity_threshold: float,
    history: list[RagHistoryMessage],
    matches: list[RagSearchMatch],
) -> RagResult:
    best_similarity = matches[0].score if matches else 0.0
    if not matches or best_similarity < similarity_threshold:
        return RagResult(
            answer=OUT_OF_SCOPE_REPLY,
            in_scope=False,
            similarity=round(best_similarity, 4),
            citations=[],
        )

    citations = build_citations(matches)
    context_lines: list[str] = []
    for citation in citations:
        locator = f" ({citation.locator})" if citation.locator else ""
        context_lines.append(
            f'- Bài "{citation.lesson_title}"{locator}: {citation.excerpt}'
        )
    prefix = (
        "Dựa trên ngữ cảnh trao đổi trước và tài liệu khóa học"
        if history
        else "Dựa trên tài liệu khóa học"
    )
    context = "\n".join(context_lines)
    return RagResult(
        answer=(
            f'{prefix}, nội dung liên quan nhất cho câu hỏi "{question}" là:\n'
            f"{context}\n\n"
            "Hãy đối chiếu câu trả lời với các trích dẫn kèm theo."
        ),
        in_scope=True,
        similarity=round(best_similarity, 4),
        citations=citations,
    )
