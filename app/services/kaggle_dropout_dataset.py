import csv
from pathlib import Path

import numpy as np

from app.schemas.dropout_risk import DropoutRiskFeatures


class KaggleDropoutDatasetError(ValueError):
    pass


REQUIRED_KAGGLE_COLUMNS = {
    "login_frequency_per_week",
    "avg_session_minutes",
    "course_progress_percent",
    "assignments_completed",
    "quiz_average_score",
    "forum_interactions",
    "video_watch_percent",
    "dropout_risk",
}


LABEL_SCORE_TARGETS = {
    "Low": 20.0,
    "Medium": 55.0,
    "High": 85.0,
}


def map_kaggle_dropout_label_to_score(label: str) -> float:
    try:
        return LABEL_SCORE_TARGETS[label]
    except KeyError as exc:
        raise KaggleDropoutDatasetError(
            f"Unsupported dropout_risk label: {label}"
        ) from exc


def _parse_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    try:
        return float(value)
    except ValueError as exc:
        raise KaggleDropoutDatasetError(
            f"Invalid numeric value for {field}: {value}"
        ) from exc


def map_kaggle_row_to_features(row: dict[str, str]) -> DropoutRiskFeatures:
    login_frequency = _parse_float(row, "login_frequency_per_week")
    avg_session_minutes = _parse_float(row, "avg_session_minutes")
    course_progress_percent = _parse_float(row, "course_progress_percent")
    assignments_completed = _parse_float(row, "assignments_completed")
    quiz_average_score = _parse_float(row, "quiz_average_score")
    forum_interactions = _parse_float(row, "forum_interactions")
    video_watch_percent = _parse_float(row, "video_watch_percent")
    watch_ratio = video_watch_percent / 100

    return DropoutRiskFeatures(
        lastActivityDays=round(max(0, 7 - login_frequency) / 7 * 14, 4),
        activityCount7d=round(login_frequency + forum_interactions, 4),
        expectedProgressGap=round(max(0, 70 - course_progress_percent), 4),
        courseProgressPercent=round(course_progress_percent, 4),
        quizScoreAvgPercent=round(quiz_average_score, 4),
        quizPassRate=1 if quiz_average_score >= 50 else 0,
        watchRatioAvg=round(watch_ratio, 4),
        abandonedAttemptCount=round(max(0, 6 - assignments_completed), 4),
        lateQuizCount=0,
        completionVelocity=round(
            course_progress_percent / 100 * assignments_completed / 7,
            4,
        ),
        totalWatchedSeconds=round(
            avg_session_minutes * 60 * login_frequency * watch_ratio,
            4,
        ),
        pauseCount=0,
    )


def _validate_columns(fieldnames: list[str] | None) -> None:
    columns = set(fieldnames or [])
    missing = sorted(REQUIRED_KAGGLE_COLUMNS - columns)
    if missing:
        raise KaggleDropoutDatasetError(
            f"Missing required Kaggle columns: {', '.join(missing)}"
        )


def load_kaggle_dropout_samples(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not csv_path.exists():
        raise KaggleDropoutDatasetError(f"Kaggle dataset not found: {csv_path}")

    matrices: list[list[float]] = []
    targets: list[float] = []

    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        _validate_columns(reader.fieldnames)

        for row in reader:
            features = map_kaggle_row_to_features(row)
            matrices.append(features.to_ordered_list())
            targets.append(map_kaggle_dropout_label_to_score(row["dropout_risk"]))

    if not matrices:
        raise KaggleDropoutDatasetError("No valid Kaggle dropout rows found")

    return np.asarray(matrices, dtype=float), np.asarray(targets, dtype=float)
