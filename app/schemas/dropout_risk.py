from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.ml.dropout_constants import FEATURE_COLUMNS


RiskBand = Literal["low", "medium", "high"]


class DropoutRiskFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    last_activity_days: float = Field(alias="lastActivityDays")
    activity_count_7d: float = Field(alias="activityCount7d")
    expected_progress_gap: float = Field(alias="expectedProgressGap")
    course_progress_percent: float = Field(alias="courseProgressPercent")
    quiz_score_avg_percent: float = Field(alias="quizScoreAvgPercent")
    quiz_pass_rate: float = Field(alias="quizPassRate")
    watch_ratio_avg: float = Field(alias="watchRatioAvg")
    abandoned_attempt_count: float = Field(alias="abandonedAttemptCount")
    late_quiz_count: float = Field(alias="lateQuizCount")
    completion_velocity: float = Field(alias="completionVelocity")
    total_watched_seconds: float = Field(alias="totalWatchedSeconds")
    pause_count: float = Field(alias="pauseCount")

    def to_ordered_list(self) -> list[float]:
        by_alias = self.model_dump(by_alias=True)
        return [float(by_alias[column]) for column in FEATURE_COLUMNS]


class DropoutRiskPredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    learner_id: str = Field(min_length=1, alias="learnerId")
    course_id: str = Field(min_length=1, alias="courseId")
    snapshot_id: str | None = Field(default=None, alias="snapshotId")
    features: DropoutRiskFeatures


class DropoutRiskPredictResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    learner_id: str = Field(alias="learnerId")
    course_id: str = Field(alias="courseId")
    snapshot_id: str | None = Field(alias="snapshotId")
    risk_score: float = Field(alias="riskScore")
    risk_band: RiskBand = Field(alias="riskBand")
    top_risk_feature: str = Field(alias="topRiskFeature")
    recommended_action: str = Field(alias="recommendedAction")
    shap_values: dict[str, float] = Field(alias="shapValues")
    model_version: str = Field(alias="modelVersion")
    predicted_at: datetime = Field(alias="predictedAt")


class ActiveDropoutModelResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_name: str = Field(alias="modelName")
    model_version: str = Field(alias="modelVersion")
    feature_version: str = Field(alias="featureVersion")
    trained_at: datetime = Field(alias="trainedAt")
    sample_count: int = Field(alias="sampleCount")
    metrics: dict[str, float]
    artifact_storage: str = Field(alias="artifactStorage")
    artifact_bucket: str = Field(alias="artifactBucket")
    artifact_object_key: str = Field(alias="artifactObjectKey")
    artifact_etag: str | None = Field(default=None, alias="artifactEtag")
    artifact_path: str = Field(alias="artifactPath")
    is_active: bool = Field(alias="isActive")
