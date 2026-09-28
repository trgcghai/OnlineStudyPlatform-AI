from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import AiSettings, get_ai_settings


router = APIRouter(prefix="/recommend", tags=["recommend"])


class RecommendationSignals(BaseModel):
    categories: dict[str, float] = Field(default_factory=dict)
    completed_course_ids: list[str] = Field(
        default_factory=list,
        alias="completedCourseIds",
    )
    engagement_by_course_id: dict[str, float] = Field(
        default_factory=dict,
        alias="engagementByCourseId",
    )
    quiz_performance_by_course_id: dict[str, float] = Field(
        default_factory=dict,
        alias="quizPerformanceByCourseId",
    )


class RecommendationCandidate(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    category: str = Field(min_length=1)
    description: str = Field(min_length=1)
    popularity: float = Field(ge=0)


class RecommendationRequest(BaseModel):
    learner_id: str = Field(min_length=1, alias="learnerId")
    limit: int = Field(ge=1, le=20)
    signals: RecommendationSignals
    candidates: list[RecommendationCandidate]


class RecommendationResponse(BaseModel):
    course_ids: list[str] = Field(alias="courseIds")


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


def dedupe_candidates(
    candidates: list[RecommendationCandidate],
) -> list[RecommendationCandidate]:
    seen: set[str] = set()
    deduped: list[RecommendationCandidate] = []

    for candidate in candidates:
        course_id = candidate.id.strip()
        if not course_id or course_id in seen:
            continue

        seen.add(course_id)
        deduped.append(candidate)

    return deduped


def rank_candidates(request: RecommendationRequest) -> list[str]:
    deduped_candidates = dedupe_candidates(request.candidates)
    ranked_candidates = sorted(
        deduped_candidates,
        key=lambda candidate: (
            -request.signals.categories.get(candidate.category, 0),
            -candidate.popularity,
            candidate.title,
            candidate.id,
        ),
    )

    return [candidate.id for candidate in ranked_candidates[: request.limit]]


@router.post(
    "",
    response_model=RecommendationResponse,
    dependencies=[Depends(verify_token)],
)
async def recommend(request: RecommendationRequest) -> RecommendationResponse:
    return RecommendationResponse(courseIds=rank_candidates(request))
