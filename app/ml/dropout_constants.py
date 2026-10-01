FEATURE_VERSION = "dropout_features_v1"
MODEL_NAME = "dropout-risk-xgboost"
MODEL_VERSION = "dropout-xgb-v1"
MODEL_ARTIFACT_NAME = "model.json"

FEATURE_COLUMNS = [
    "lastActivityDays",
    "activityCount7d",
    "expectedProgressGap",
    "courseProgressPercent",
    "quizScoreAvgPercent",
    "quizPassRate",
    "watchRatioAvg",
    "abandonedAttemptCount",
    "lateQuizCount",
    "completionVelocity",
    "totalWatchedSeconds",
    "pauseCount",
]

RECOMMENDED_ACTIONS = {
    "lastActivityDays": "Trigger slow-progress reminder and send motivation nudge.",
    "expectedProgressGap": "Recommend catch-up study schedule and milestone reminders.",
    "quizScoreAvgPercent": "Recommend foundational review lessons and practice quizzes.",
    "abandonedAttemptCount": "Suggest introductory mock tests and tutor consultation.",
    "activityCount7d": "Send weekly engagement nudge and display progress tracker.",
    "lateQuizCount": "Recommend calendar usage and submission alerts.",
    "watchRatioAvg": "Recommend micro-video summaries or shorter review content.",
    "pauseCount": "Suggest slower playback speed or lesson transcripts.",
}

DEFAULT_RECOMMENDED_ACTION = "Recommend personalized review materials and check-in."


def risk_band_for_score(score: float) -> str:
    if score >= 70:
        return "high"
    if score >= 30:
        return "medium"
    return "low"


def clamp_score(score: float) -> float:
    return round(min(100, max(0, score)), 2)


def recommended_action_for_feature(feature_name: str) -> str:
    return RECOMMENDED_ACTIONS.get(feature_name, DEFAULT_RECOMMENDED_ACTION)
