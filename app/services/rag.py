import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol


OUT_OF_SCOPE_REPLY = (
    "Xin lỗi, câu hỏi này nằm ngoài nội dung kiến thức của khóa học. "
    "Bạn hãy hỏi về bài học, tài liệu hoặc nội dung trong khóa học này."
)
MAX_HISTORY_MESSAGES = 4
MAX_CONTEXT_CHARACTERS = 12_000
FOLLOW_UP_QUESTIONS = {
    "giai thich them",
    "giai thich ro hon",
    "noi ro hon",
    "chi tiet hon",
    "huong dan chi tiet hon",
    "cho vi du",
    "vi du",
    "tai sao",
    "nhu the nao",
    "lam the nao",
    "ket hop nhu nao",
    "ket hop nhu the nao",
    "buoc tiep theo",
    "tiep tuc",
}
FOLLOW_UP_REFERENCE = re.compile(
    r"\b(?:no|chung|cai (?:nay|do)|dieu (?:nay|do)|buoc (?:nay|do)"
    r"|cach (?:nay|do)|phuong phap (?:nay|do))\b"
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
    text: str


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


def is_follow_up_question(question: str) -> bool:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFD", question.casefold())
        if not unicodedata.combining(character)
    ).replace("đ", "d")
    normalized = " ".join(re.sub(r"[^\w\s]", " ", normalized).split())
    return normalized in FOLLOW_UP_QUESTIONS or bool(
        FOLLOW_UP_REFERENCE.search(normalized)
    )


def select_history(
    question: str, history: list[RagHistoryMessage]
) -> list[RagHistoryMessage]:
    if not is_follow_up_question(question):
        return []
    recent = history[-MAX_HISTORY_MESSAGES:]
    for index in range(len(recent) - 1, -1, -1):
        message = recent[index]
        if message.role == "user" and not is_follow_up_question(message.content):
            return recent[index:]
    return recent if any(message.role == "user" for message in recent) else []


def build_query(
    question: str, history: list[RagHistoryMessage] | None = None
) -> str:
    recent_user_questions = [
        message.content.strip()
        for message in select_history(question, history or [])
        if message.role == "user" and message.content.strip()
    ]
    return "\n".join([*recent_user_questions, question.strip()])


def build_citations(matches: list[RagSearchMatch]) -> list[RagCitation]:
    citations: list[RagCitation] = []
    seen: set[tuple[str, str, str | None, str]] = set()
    context_characters = 0
    for match in matches:
        text = " ".join(match.text.split())
        key = (
            match.lesson_id,
            match.source_type,
            match.locator,
            text,
        )
        if key in seen:
            continue
        if not text or context_characters + len(text) > MAX_CONTEXT_CHARACTERS:
            continue
        seen.add(key)
        context_characters += len(text)
        citations.append(
            RagCitation(
                lesson_id=match.lesson_id,
                lesson_title=match.lesson_title,
                excerpt=excerpt(text),
                source_type=match.source_type,
                locator=match.locator,
                text=text,
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
