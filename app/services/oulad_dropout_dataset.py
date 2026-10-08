import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.schemas.dropout_risk import DropoutRiskFeatures


DEFAULT_OULAD_SNAPSHOT_DAY = 60
OULAD_TARGET_POLICY = "withdrawn_fail_vs_pass_distinction"

REQUIRED_OULAD_FILES = {
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentVle.csv",
    "studentAssessment.csv",
    "assessments.csv",
    "courses.csv",
}


class OuladDropoutDatasetError(ValueError):
    pass


def map_oulad_final_result_to_score(final_result: str) -> float:
    if final_result in {"Withdrawn", "Fail"}:
        return 85.0
    if final_result in {"Pass", "Distinction"}:
        return 20.0
    raise OuladDropoutDatasetError(f"Unsupported final_result: {final_result}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _parse_int(value: str, default: int = 0) -> int:
    if value == "?" or value == "":
        return default
    return int(float(value))


def _parse_float(value: str, default: float = 0) -> float:
    if value == "?" or value == "":
        return default
    return float(value)


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _validate_required_files(data_dir: Path) -> None:
    missing = sorted(
        file_name
        for file_name in REQUIRED_OULAD_FILES
        if not (data_dir / file_name).exists()
    )
    if missing:
        raise OuladDropoutDatasetError(f"Missing OULAD file: {', '.join(missing)}")


def load_oulad_dropout_samples(
    data_dir: Path,
    snapshot_day: int = DEFAULT_OULAD_SNAPSHOT_DAY,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | str]]:
    _validate_required_files(data_dir)

    courses = {
        (row["code_module"], row["code_presentation"]): _parse_int(
            row["module_presentation_length"],
            snapshot_day,
        )
        for row in _read_csv(data_dir / "courses.csv")
    }
    assessments_by_course: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    assessment_course: dict[str, tuple[str, str]] = {}
    for row in _read_csv(data_dir / "assessments.csv"):
        key = (row["code_module"], row["code_presentation"])
        assessments_by_course[key].append(row)
        assessment_course[row["id_assessment"]] = key

    registrations = {
        (row["code_module"], row["code_presentation"], row["id_student"]): row
        for row in _read_csv(data_dir / "studentRegistration.csv")
    }

    submissions: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(data_dir / "studentAssessment.csv"):
        course_key = assessment_course.get(row["id_assessment"])
        if not course_key:
            continue
        submissions[(course_key[0], course_key[1], row["id_student"])].append(row)

    vle_events: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(data_dir / "studentVle.csv"):
        date = _parse_int(row["date"])
        if date <= snapshot_day:
            vle_events[(row["code_module"], row["code_presentation"], row["id_student"])].append(
                row
            )

    matrices: list[list[float]] = []
    targets: list[float] = []
    skipped_rows = 0

    for row in _read_csv(data_dir / "studentInfo.csv"):
        key = (row["code_module"], row["code_presentation"], row["id_student"])
        course_key = (row["code_module"], row["code_presentation"])
        if course_key not in courses or key not in registrations:
            skipped_rows += 1
            continue

        features = _build_features(
            course_length=courses[course_key],
            registration=registrations[key],
            assessments=assessments_by_course[course_key],
            submissions=submissions.get(key, []),
            events=vle_events.get(key, []),
            snapshot_day=snapshot_day,
        )
        matrices.append(features.to_ordered_list())
        targets.append(map_oulad_final_result_to_score(row["final_result"]))

    if not matrices:
        raise OuladDropoutDatasetError("No valid OULAD dropout rows found")

    metadata = {
        "sampleCount": len(matrices),
        "snapshotDay": snapshot_day,
        "skippedRows": skipped_rows,
    }
    return np.asarray(matrices, dtype=float), np.asarray(targets, dtype=float), metadata


def _build_features(
    *,
    course_length: int,
    registration: dict[str, str],
    assessments: list[dict[str, str]],
    submissions: list[dict[str, str]],
    events: list[dict[str, str]],
    snapshot_day: int,
) -> DropoutRiskFeatures:
    registration_day = _parse_int(registration["date_registration"])
    active_days = max(1, snapshot_day - registration_day)
    event_dates = [_parse_int(event["date"]) for event in events]
    latest_activity = max(event_dates) if event_dates else registration_day
    total_clicks = sum(_parse_int(event["sum_click"]) for event in events)
    activity_count_7d = sum(
        _parse_int(event["sum_click"])
        for event in events
        if snapshot_day - 7 <= _parse_int(event["date"]) <= snapshot_day
    )

    due_assessments = [
        assessment
        for assessment in assessments
        if _parse_int(assessment["date"], snapshot_day + 1) <= snapshot_day
    ]
    submitted_by_id = {
        submission["id_assessment"]: submission
        for submission in submissions
        if _parse_int(submission["date_submitted"], snapshot_day + 1) <= snapshot_day
    }
    submitted_due = [
        submitted_by_id[assessment["id_assessment"]]
        for assessment in due_assessments
        if assessment["id_assessment"] in submitted_by_id
    ]
    scores = [_parse_float(submission["score"]) for submission in submitted_due]
    pass_count = sum(1 for score in scores if score >= 40)
    total_assessment_count = max(1, len(assessments))
    course_progress_percent = len(submitted_by_id) / total_assessment_count * 100
    expected_progress_percent = min(100, snapshot_day / max(1, course_length) * 100)
    expected_clicks = max(1, snapshot_day * 0.5)
    late_quiz_count = 0
    for assessment in due_assessments:
        submission = submitted_by_id.get(assessment["id_assessment"])
        if submission and _parse_int(submission["date_submitted"]) > _parse_int(
            assessment["date"]
        ):
            late_quiz_count += 1

    return DropoutRiskFeatures(
        lastActivityDays=round(max(0, snapshot_day - latest_activity), 4),
        activityCount7d=round(activity_count_7d, 4),
        expectedProgressGap=round(max(0, expected_progress_percent - course_progress_percent), 4),
        courseProgressPercent=round(course_progress_percent, 4),
        quizScoreAvgPercent=round(_average(scores), 4),
        quizPassRate=round(pass_count / len(scores), 4) if scores else 0,
        watchRatioAvg=round(min(1, total_clicks / expected_clicks), 4),
        abandonedAttemptCount=0,
        lateQuizCount=late_quiz_count,
        completionVelocity=round(len(submitted_by_id) / active_days, 4),
        totalWatchedSeconds=round(total_clicks * 30, 4),
        pauseCount=0,
    )
