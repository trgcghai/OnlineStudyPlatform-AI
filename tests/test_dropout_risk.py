import unittest
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.dropout_risk import get_dropout_database
from app.core.config import get_ai_settings
from app.main import app
from app.ml.dropout_constants import (
    DROPOUT_KAGGLE_MODEL_VERSION,
    DROPOUT_OULAD_MODEL_VERSION,
    FEATURE_COLUMNS,
    recommended_action_for_feature,
    risk_band_for_score,
)
from app.schemas.dropout_risk import DropoutRiskFeatures, DropoutRiskPredictRequest
from app.services.dropout_explainability import explain_prediction, top_feature_from_shap
from app.services.oulad_dropout_dataset import (
    DEFAULT_OULAD_SNAPSHOT_DAY,
    OuladDropoutDatasetError,
    load_oulad_dropout_samples,
    map_oulad_final_result_to_score,
)
from app.services.dropout_model import (
    build_feature_matrix,
    resolve_model_artifact_path,
)
from app.services.model_artifact_store import (
    ModelArtifactLocation,
    build_model_object_key,
    cached_model_path,
    download_model_artifact,
    upload_model_artifact,
)
from app.services.kaggle_dropout_dataset import (
    KaggleDropoutDatasetError,
    load_kaggle_dropout_samples,
    map_kaggle_dropout_label_to_score,
    map_kaggle_row_to_features,
)
from app.services.dropout_model_registry import register_dropout_model
from scripts.train_kaggle_dropout import train_kaggle_dropout_model
from scripts.train_oulad_dropout import train_oulad_dropout_model


def valid_features() -> dict:
    return {
        "lastActivityDays": 5,
        "activityCount7d": 12,
        "expectedProgressGap": 18.5,
        "courseProgressPercent": 42,
        "quizScoreAvgPercent": 68.2,
        "quizPassRate": 0.75,
        "watchRatioAvg": 0.61,
        "abandonedAttemptCount": 1,
        "lateQuizCount": 2,
        "completionVelocity": 0.35,
        "totalWatchedSeconds": 7200,
        "pauseCount": 14,
    }


def valid_kaggle_row() -> dict[str, str]:
    return {
        "student_id": "1",
        "course_category": "Design",
        "login_frequency_per_week": "14",
        "avg_session_minutes": "17",
        "course_progress_percent": "51",
        "assignments_completed": "7",
        "quiz_average_score": "60.42",
        "forum_interactions": "14",
        "video_watch_percent": "23",
        "device_type": "Desktop",
        "dropout_risk": "Low",
    }


def write_minimal_oulad_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "courses.csv").write_text(
        "code_module,code_presentation,module_presentation_length\n"
        "AAA,2013J,100\n",
        encoding="utf-8",
    )
    (root / "assessments.csv").write_text(
        "code_module,code_presentation,id_assessment,assessment_type,date,weight\n"
        "AAA,2013J,1,TMA,20,30\n"
        "AAA,2013J,2,TMA,50,30\n"
        "AAA,2013J,3,TMA,80,40\n",
        encoding="utf-8",
    )
    (root / "studentInfo.csv").write_text(
        "code_module,code_presentation,id_student,gender,region,highest_education,"
        "imd_band,age_band,num_of_prev_attempts,studied_credits,disability,final_result\n"
        "AAA,2013J,10,M,Region,A Level,50-60%,0-35,0,60,N,Fail\n"
        "AAA,2013J,11,F,Region,A Level,50-60%,0-35,0,60,N,Pass\n",
        encoding="utf-8",
    )
    (root / "studentRegistration.csv").write_text(
        "code_module,code_presentation,id_student,date_registration,date_unregistration\n"
        "AAA,2013J,10,0,?\n"
        "AAA,2013J,11,0,?\n",
        encoding="utf-8",
    )
    (root / "studentAssessment.csv").write_text(
        "id_assessment,id_student,date_submitted,is_banked,score\n"
        "1,10,22,0,80\n"
        "2,10,55,0,30\n"
        "1,11,20,0,90\n"
        "2,11,48,0,85\n",
        encoding="utf-8",
    )
    (root / "studentVle.csv").write_text(
        "code_module,code_presentation,id_student,id_site,date,sum_click\n"
        "AAA,2013J,10,100,10,20\n"
        "AAA,2013J,10,101,54,3\n"
        "AAA,2013J,10,102,59,7\n"
        "AAA,2013J,11,100,58,20\n"
        "AAA,2013J,11,101,60,10\n",
        encoding="utf-8",
    )
    return root


class DropoutRiskSchemaTests(unittest.TestCase):
    def test_dataset_specific_model_versions_are_defined(self) -> None:
        self.assertEqual(DROPOUT_KAGGLE_MODEL_VERSION, "dropout-kaggle-v1")
        self.assertEqual(DROPOUT_OULAD_MODEL_VERSION, "dropout-oulad-v1")

    def test_feature_order_matches_backend_contract(self) -> None:
        self.assertEqual(
            FEATURE_COLUMNS,
            [
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
            ],
        )

    def test_risk_band_mapping_uses_30_70_thresholds(self) -> None:
        self.assertEqual(risk_band_for_score(29.99), "low")
        self.assertEqual(risk_band_for_score(30), "medium")
        self.assertEqual(risk_band_for_score(69.99), "medium")
        self.assertEqual(risk_band_for_score(70), "high")

    def test_recommended_action_mapping_for_known_feature(self) -> None:
        self.assertEqual(
            recommended_action_for_feature("lastActivityDays"),
            "Trigger slow-progress reminder and send motivation nudge.",
        )

    def test_feature_validation_rejects_missing_feature(self) -> None:
        payload = valid_features()
        del payload["pauseCount"]

        with self.assertRaises(ValidationError):
            DropoutRiskFeatures(**payload)

    def test_predict_request_accepts_camel_case_features(self) -> None:
        request = DropoutRiskPredictRequest(
            learnerId="learner-1",
            courseId="course-1",
            snapshotId=None,
            features=valid_features(),
        )

        self.assertEqual(request.features.last_activity_days, 5)
        self.assertEqual(request.features.to_ordered_list()[0], 5)


class DropoutRiskServiceTests(unittest.TestCase):
    def test_build_feature_matrix_uses_canonical_order(self) -> None:
        matrix = build_feature_matrix(DropoutRiskFeatures(**valid_features()))

        self.assertEqual(matrix.shape, (1, len(FEATURE_COLUMNS)))
        self.assertEqual(matrix[0][0], 5)
        self.assertEqual(matrix[0][-1], 14)

    def test_resolve_model_artifact_path_downloads_minio_artifact(self) -> None:
        client = FakeMinioClient()
        metadata = {
            "modelVersion": "dropout-oulad-v1",
            "artifactStorage": "minio",
            "artifactBucket": "ai-models",
            "artifactObjectKey": "dropout-risk/dropout-oulad-v1/model.json",
            "artifactPath": "minio://ai-models/dropout-risk/dropout-oulad-v1/model.json",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = resolve_model_artifact_path(
                metadata,
                artifact_client=client,
                cache_root=Path(tmp),
            )

            self.assertTrue(path.exists())

        self.assertEqual(client.fget_object_calls[0][0], "ai-models")
        self.assertEqual(
            client.fget_object_calls[0][1],
            "dropout-risk/dropout-oulad-v1/model.json",
        )

    def test_resolve_model_artifact_path_rejects_legacy_local_metadata(self) -> None:
        metadata = {
            "modelVersion": "dropout-xgb-v1",
            "artifactPath": "models/dropout-risk/dropout-xgb-v1/model.json",
        }

        with self.assertRaisesRegex(ValueError, "Unsupported model artifact storage"):
            resolve_model_artifact_path(
                metadata,
                artifact_client=FakeMinioClient(),
                cache_root=Path("/cache"),
            )

    def test_top_feature_prefers_highest_positive_shap_value(self) -> None:
        shap_values = {
            "lastActivityDays": -5.0,
            "expectedProgressGap": 8.5,
            "pauseCount": 1.0,
        }

        self.assertEqual(top_feature_from_shap(shap_values), "expectedProgressGap")

    def test_top_feature_uses_absolute_value_when_no_positive_values_exist(self) -> None:
        shap_values = {
            "lastActivityDays": -5.0,
            "expectedProgressGap": -2.0,
        }

        self.assertEqual(top_feature_from_shap(shap_values), "lastActivityDays")

    def test_explain_prediction_returns_feature_values_from_explainer(self) -> None:
        explainer = Mock()
        explainer.shap_values.return_value = [[1.5, -2.0] + [0.0] * 10]
        matrix = build_feature_matrix(DropoutRiskFeatures(**valid_features()))

        shap_values = explain_prediction(explainer, matrix)

        self.assertEqual(shap_values["lastActivityDays"], 1.5)
        self.assertEqual(shap_values["activityCount7d"], -2.0)


class FakeMinioClient:
    def __init__(self) -> None:
        self.buckets: set[str] = set()
        self.bucket_exists_calls: list[str] = []
        self.make_bucket_calls: list[str] = []
        self.fput_object_calls: list[tuple[str, str, str]] = []
        self.fget_object_calls: list[tuple[str, str, str]] = []

    def bucket_exists(self, bucket_name: str) -> bool:
        self.bucket_exists_calls.append(bucket_name)
        return bucket_name in self.buckets

    def make_bucket(self, bucket_name: str) -> None:
        self.make_bucket_calls.append(bucket_name)
        self.buckets.add(bucket_name)

    def fput_object(self, bucket_name: str, object_name: str, file_path: str) -> object:
        self.fput_object_calls.append((bucket_name, object_name, file_path))

        class Result:
            etag = "unit-etag"

        return Result()

    def fget_object(self, bucket_name: str, object_name: str, file_path: str) -> None:
        self.fget_object_calls.append((bucket_name, object_name, file_path))
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(file_path).write_text("downloaded-model", encoding="utf-8")


class ModelArtifactStoreTests(unittest.TestCase):
    def test_build_model_object_key_uses_dropout_risk_prefix(self) -> None:
        self.assertEqual(
            build_model_object_key("dropout-oulad-v1"),
            "dropout-risk/dropout-oulad-v1/model.json",
        )

    def test_upload_model_artifact_creates_bucket_and_returns_location(self) -> None:
        client = FakeMinioClient()
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "model.json"
            artifact.write_text("model", encoding="utf-8")

            location = upload_model_artifact(
                client=client,
                bucket_name="ai-models",
                model_version="dropout-oulad-v1",
                artifact_path=artifact,
            )

        self.assertEqual(client.make_bucket_calls, ["ai-models"])
        self.assertEqual(
            client.fput_object_calls[0][1],
            "dropout-risk/dropout-oulad-v1/model.json",
        )
        self.assertEqual(location.storage, "minio")
        self.assertEqual(location.bucket, "ai-models")
        self.assertEqual(location.etag, "unit-etag")
        self.assertEqual(
            location.uri,
            "minio://ai-models/dropout-risk/dropout-oulad-v1/model.json",
        )

    def test_cached_model_path_uses_cache_root_and_model_version(self) -> None:
        self.assertEqual(
            cached_model_path(Path("/cache"), "dropout-oulad-v1"),
            Path("/cache/dropout-risk/dropout-oulad-v1/model.json"),
        )

    def test_download_model_artifact_reuses_cached_file(self) -> None:
        client = FakeMinioClient()
        location = ModelArtifactLocation(
            storage="minio",
            bucket="ai-models",
            object_key="dropout-risk/dropout-oulad-v1/model.json",
            uri="minio://ai-models/dropout-risk/dropout-oulad-v1/model.json",
            etag="unit-etag",
        )
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp)
            cached = cached_model_path(cache_root, "dropout-oulad-v1")
            cached.parent.mkdir(parents=True)
            cached.write_text("cached-model", encoding="utf-8")

            result = download_model_artifact(
                client=client,
                location=location,
                cache_root=cache_root,
                model_version="dropout-oulad-v1",
            )

        self.assertEqual(result, cached)
        self.assertEqual(client.fget_object_calls, [])


class DropoutModelRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_register_dropout_model_writes_versioned_artifact_and_metadata(self) -> None:
        model = Mock()
        database = FakeDatabase([])
        trained_at = datetime(2026, 10, 6, tzinfo=UTC)

        with tempfile.TemporaryDirectory() as tmp:
            client = FakeMinioClient()
            response = await register_dropout_model(
                database=database,
                model=model,
                artifact_client=client,
                artifact_bucket="ai-models",
                temp_dir=Path(tmp) / "tmp-artifacts",
                model_version="dropout-test-v1",
                sample_count=12,
                metrics={"mae": 1.23, "rmse": 2.34},
                trained_at=trained_at,
                extra_metadata={"trainingSource": "unit-test"},
            )

            expected_artifact = (
                Path(tmp) / "tmp-artifacts" / "dropout-test-v1" / "model.json"
            )
            model.save_model.assert_called_once_with(expected_artifact)
            self.assertEqual(client.fput_object_calls[0][0], "ai-models")
            self.assertEqual(
                client.fput_object_calls[0][1],
                "dropout-risk/dropout-test-v1/model.json",
            )

        self.assertEqual(response.model_version, "dropout-test-v1")
        self.assertEqual(
            response.artifact_path,
            "minio://ai-models/dropout-risk/dropout-test-v1/model.json",
        )
        self.assertTrue(response.is_active)
        self.assertEqual(len(database.ai_model_versions.update_many_calls), 1)
        self.assertEqual(len(database.ai_model_versions.replace_one_calls), 1)
        query, metadata, upsert = database.ai_model_versions.replace_one_calls[0]
        self.assertEqual(query, {"_id": "dropout-test-v1"})
        self.assertTrue(upsert)
        self.assertEqual(metadata["trainingSource"], "unit-test")
        self.assertEqual(metadata["artifactStorage"], "minio")
        self.assertEqual(metadata["artifactBucket"], "ai-models")
        self.assertEqual(
            metadata["artifactObjectKey"],
            "dropout-risk/dropout-test-v1/model.json",
        )
        self.assertEqual(metadata["artifactEtag"], "unit-etag")


class KaggleDropoutDatasetTests(unittest.TestCase):
    def test_label_mapping_converts_risk_bands_to_score_targets(self) -> None:
        self.assertEqual(map_kaggle_dropout_label_to_score("Low"), 20.0)
        self.assertEqual(map_kaggle_dropout_label_to_score("Medium"), 55.0)
        self.assertEqual(map_kaggle_dropout_label_to_score("High"), 85.0)

    def test_label_mapping_rejects_unknown_label(self) -> None:
        with self.assertRaisesRegex(
            KaggleDropoutDatasetError,
            "Unsupported dropout_risk label",
        ):
            map_kaggle_dropout_label_to_score("Critical")

    def test_maps_low_risk_kaggle_row_to_current_feature_contract(self) -> None:
        features = map_kaggle_row_to_features(valid_kaggle_row())

        self.assertEqual(features.last_activity_days, 0)
        self.assertEqual(features.activity_count_7d, 28)
        self.assertEqual(features.expected_progress_gap, 19)
        self.assertEqual(features.course_progress_percent, 51)
        self.assertEqual(features.quiz_score_avg_percent, 60.42)
        self.assertEqual(features.quiz_pass_rate, 1)
        self.assertEqual(features.watch_ratio_avg, 0.23)
        self.assertEqual(features.abandoned_attempt_count, 0)
        self.assertEqual(features.late_quiz_count, 0)
        self.assertEqual(features.completion_velocity, 0.51)
        self.assertEqual(features.total_watched_seconds, 3284.4)
        self.assertEqual(features.pause_count, 0)

    def test_maps_high_risk_kaggle_row_to_current_feature_contract(self) -> None:
        row = valid_kaggle_row()
        row.update(
            {
                "login_frequency_per_week": "1",
                "avg_session_minutes": "87",
                "course_progress_percent": "34",
                "assignments_completed": "4",
                "quiz_average_score": "43.08",
                "forum_interactions": "26",
                "video_watch_percent": "75",
                "dropout_risk": "High",
            }
        )

        features = map_kaggle_row_to_features(row)

        self.assertEqual(features.last_activity_days, 12)
        self.assertEqual(features.activity_count_7d, 27)
        self.assertEqual(features.expected_progress_gap, 36)
        self.assertEqual(features.course_progress_percent, 34)
        self.assertEqual(features.quiz_score_avg_percent, 43.08)
        self.assertEqual(features.quiz_pass_rate, 0)
        self.assertEqual(features.watch_ratio_avg, 0.75)
        self.assertEqual(features.abandoned_attempt_count, 2)
        self.assertEqual(features.late_quiz_count, 0)
        self.assertEqual(features.completion_velocity, 0.1943)
        self.assertEqual(features.total_watched_seconds, 3915)
        self.assertEqual(features.pause_count, 0)

    def test_load_kaggle_dropout_samples_reads_csv_in_feature_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "kaggle.csv"
            csv_path.write_text(
                "student_id,course_category,login_frequency_per_week,avg_session_minutes,"
                "course_progress_percent,assignments_completed,quiz_average_score,"
                "forum_interactions,video_watch_percent,device_type,dropout_risk\n"
                "1,Design,14,17,51,7,60.42,14,23,Desktop,Low\n",
                encoding="utf-8",
            )

            matrix, targets = load_kaggle_dropout_samples(csv_path)

        self.assertEqual(matrix.shape, (1, len(FEATURE_COLUMNS)))
        self.assertEqual(
            matrix[0].tolist(),
            [0, 28, 19, 51, 60.42, 1, 0.23, 0, 0, 0.51, 3284.4, 0],
        )
        self.assertEqual(targets.tolist(), [20.0])

    def test_load_kaggle_dropout_samples_rejects_missing_required_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "kaggle.csv"
            csv_path.write_text(
                "student_id,dropout_risk\n"
                "1,Low\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                KaggleDropoutDatasetError,
                "Missing required Kaggle columns",
            ):
                load_kaggle_dropout_samples(csv_path)

    def test_load_kaggle_dropout_samples_rejects_malformed_numeric_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "kaggle.csv"
            csv_path.write_text(
                "student_id,course_category,login_frequency_per_week,avg_session_minutes,"
                "course_progress_percent,assignments_completed,quiz_average_score,"
                "forum_interactions,video_watch_percent,device_type,dropout_risk\n"
                "1,Design,not-a-number,17,51,7,60.42,14,23,Desktop,Low\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                KaggleDropoutDatasetError,
                "Invalid numeric value for login_frequency_per_week",
            ):
                load_kaggle_dropout_samples(csv_path)


class OuladDropoutDatasetTests(unittest.TestCase):
    def test_oulad_target_mapping_uses_at_risk_policy(self) -> None:
        self.assertEqual(map_oulad_final_result_to_score("Withdrawn"), 85.0)
        self.assertEqual(map_oulad_final_result_to_score("Fail"), 85.0)
        self.assertEqual(map_oulad_final_result_to_score("Pass"), 20.0)
        self.assertEqual(map_oulad_final_result_to_score("Distinction"), 20.0)

    def test_oulad_target_mapping_rejects_unknown_result(self) -> None:
        with self.assertRaisesRegex(OuladDropoutDatasetError, "Unsupported final_result"):
            map_oulad_final_result_to_score("Unknown")

    def test_load_oulad_dropout_samples_builds_features_from_relational_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = write_minimal_oulad_dataset(Path(tmp) / "oulad")
            matrix, targets, metadata = load_oulad_dropout_samples(data_dir)

        self.assertEqual(matrix.shape, (2, len(FEATURE_COLUMNS)))
        self.assertEqual(targets.tolist(), [85.0, 20.0])
        self.assertEqual(metadata["sampleCount"], 2)
        self.assertEqual(metadata["snapshotDay"], DEFAULT_OULAD_SNAPSHOT_DAY)
        self.assertEqual(metadata["skippedRows"], 0)
        self.assertEqual(
            matrix[0].tolist(),
            [1, 10, 0, 66.6667, 55.0, 0.5, 1.0, 0, 2, 0.0333, 900, 0],
        )

    def test_load_oulad_dropout_samples_rejects_missing_required_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "oulad"
            data_dir.mkdir()
            with self.assertRaisesRegex(OuladDropoutDatasetError, "Missing OULAD file"):
                load_oulad_dropout_samples(data_dir)


class KaggleDropoutTrainingScriptTests(unittest.IsolatedAsyncioTestCase):
    async def test_train_kaggle_dropout_model_writes_artifact_and_metadata(self) -> None:
        rows = [
            "student_id,course_category,login_frequency_per_week,avg_session_minutes,"
            "course_progress_percent,assignments_completed,quiz_average_score,"
            "forum_interactions,video_watch_percent,device_type,dropout_risk\n",
        ]
        labels = ["Low", "Medium", "High"] * 4
        for index, label in enumerate(labels, start=1):
            rows.append(
                f"{index},Design,{1 + index % 14},30,"
                f"{20 + index},5,60,{index % 10},70,Desktop,{label}\n"
            )

        database = FakeDatabase([])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "kaggle.csv"
            client = FakeMinioClient()
            csv_path.write_text("".join(rows), encoding="utf-8")

            response = await train_kaggle_dropout_model(
                database=database,
                csv_path=csv_path,
                artifact_client=client,
                artifact_bucket="ai-models",
                temp_dir=root / "tmp-artifacts",
            )

            self.assertTrue(response.artifact_path.startswith("minio://ai-models/"))
            self.assertEqual(client.fput_object_calls[0][0], "ai-models")

        self.assertEqual(response.model_version, DROPOUT_KAGGLE_MODEL_VERSION)
        self.assertEqual(response.feature_version, "dropout_features_v1")
        self.assertEqual(response.sample_count, 12)
        self.assertTrue(response.is_active)
        self.assertIn("mae", response.metrics)
        self.assertIn("rmse", response.metrics)
        self.assertEqual(len(database.ai_model_versions.update_many_calls), 1)
        self.assertEqual(len(database.ai_model_versions.replace_one_calls), 1)
        _, metadata, upsert = database.ai_model_versions.replace_one_calls[0]
        self.assertTrue(upsert)
        self.assertEqual(metadata["trainingSource"], "kaggle-1")
        self.assertEqual(metadata["sampleCount"], 12)


class OuladDropoutTrainingScriptTests(unittest.IsolatedAsyncioTestCase):
    async def test_train_oulad_dropout_model_writes_artifact_and_metadata(self) -> None:
        database = FakeDatabase([])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = write_minimal_oulad_dataset(root / "oulad")
            client = FakeMinioClient()

            response = await train_oulad_dropout_model(
                database=database,
                data_dir=data_dir,
                artifact_client=client,
                artifact_bucket="ai-models",
                temp_dir=root / "tmp-artifacts",
            )

            self.assertTrue(response.artifact_path.startswith("minio://ai-models/"))
            self.assertEqual(client.fput_object_calls[0][0], "ai-models")

        self.assertEqual(response.model_version, DROPOUT_OULAD_MODEL_VERSION)
        self.assertEqual(response.feature_version, "dropout_features_v1")
        self.assertEqual(response.sample_count, 2)
        self.assertTrue(response.is_active)
        self.assertIn("mae", response.metrics)
        self.assertIn("rmse", response.metrics)
        self.assertEqual(len(database.ai_model_versions.update_many_calls), 1)
        self.assertEqual(len(database.ai_model_versions.replace_one_calls), 1)
        _, metadata, upsert = database.ai_model_versions.replace_one_calls[0]
        self.assertTrue(upsert)
        self.assertEqual(metadata["trainingSource"], "oulad")
        self.assertEqual(metadata["targetPolicy"], "withdrawn_fail_vs_pass_distinction")
        self.assertEqual(metadata["snapshotDay"], DEFAULT_OULAD_SNAPSHOT_DAY)


class FakeCursor:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    async def to_list(self, length: int | None = None) -> list[dict]:
        return self.rows


class FakeCollection:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.rows = rows or []
        self.update_many_calls: list[tuple] = []
        self.replace_one_calls: list[tuple] = []

    def find(self, query: dict) -> FakeCursor:
        self.last_find_query = query
        return FakeCursor(self.rows)

    async def update_many(self, query: dict, update: dict) -> None:
        self.update_many_calls.append((query, update))

    async def replace_one(self, query: dict, replacement: dict, upsert: bool) -> None:
        self.replace_one_calls.append((query, replacement, upsert))


class FakeDatabase:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.ai_model_versions = FakeCollection()


class DropoutRiskApiTests(unittest.TestCase):
    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def allow_without_token(self) -> None:
        class Settings:
            ai_token = None
            dropout_model_dir = "models/dropout-risk"
            dropout_min_training_samples = 10
            minio_endpoint = "http://localhost:9000"
            minio_access_key = "minioadmin"
            minio_secret_key = "minioadmin"
            minio_secure = False
            dropout_model_cache_dir = "/var/cache/ai-models"

        app.dependency_overrides[get_ai_settings] = lambda: Settings()

    def test_invalid_token_is_rejected_when_ai_token_is_configured(self) -> None:
        class Settings:
            ai_token = "secret"
            dropout_model_dir = "models/dropout-risk"
            dropout_min_training_samples = 10
            minio_endpoint = "http://localhost:9000"
            minio_access_key = "minioadmin"
            minio_secret_key = "minioadmin"
            minio_secure = False
            dropout_model_cache_dir = "/var/cache/ai-models"

        app.dependency_overrides[get_ai_settings] = lambda: Settings()

        response = TestClient(app).post(
            "/analytics/dropout-risk/predict",
            json={
                "learnerId": "learner-1",
                "courseId": "course-1",
                "features": valid_features(),
            },
            headers={"Authorization": "Bearer wrong"},
        )

        self.assertEqual(response.status_code, 401)

    def test_train_endpoint_is_not_registered(self) -> None:
        self.allow_without_token()

        response = TestClient(app).post("/analytics/dropout-risk/train")

        self.assertEqual(response.status_code, 404)

    def test_active_model_endpoint_returns_metadata(self) -> None:
        self.allow_without_token()
        trained_at = datetime(2026, 9, 30, tzinfo=UTC)

        async def fake_active_model(database: object) -> dict:
            return {
                "modelName": "dropout-risk-xgboost",
                "modelVersion": "dropout-xgb-v1",
                "featureVersion": "dropout_features_v1",
                "trainedAt": trained_at,
                "sampleCount": 12,
                "metrics": {"mae": 1.0, "rmse": 2.0},
                "artifactStorage": "minio",
                "artifactBucket": "ai-models",
                "artifactObjectKey": "dropout-risk/dropout-xgb-v1/model.json",
                "artifactEtag": "unit-etag",
                "artifactPath": "minio://ai-models/dropout-risk/dropout-xgb-v1/model.json",
                "isActive": True,
            }

        app.dependency_overrides[get_dropout_database] = lambda: FakeDatabase([])
        with patch("app.api.dropout_risk.get_active_model_metadata", new=fake_active_model):
            response = TestClient(app).get("/analytics/dropout-risk/model/active")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["modelVersion"], "dropout-xgb-v1")
        self.assertEqual(response.json()["artifactStorage"], "minio")
        self.assertEqual(response.json()["artifactBucket"], "ai-models")

    def test_predict_returns_model_response_when_services_are_mocked(self) -> None:
        self.allow_without_token()
        metadata = {
            "modelVersion": "dropout-xgb-v1",
            "artifactStorage": "minio",
            "artifactBucket": "ai-models",
            "artifactObjectKey": "dropout-risk/dropout-xgb-v1/model.json",
            "artifactPath": "minio://ai-models/dropout-risk/dropout-xgb-v1/model.json",
        }
        model = Mock()
        model.predict.return_value = [82.4]
        explainer = Mock()
        explainer.shap_values.return_value = [[23.1, 0.0, 15.4] + [0.0] * 9]

        async def fake_active_model(database: object) -> dict:
            return metadata

        app.dependency_overrides[get_dropout_database] = lambda: FakeDatabase([])
        with patch("app.api.dropout_risk.get_active_model_metadata", new=fake_active_model):
            with patch(
                "app.api.dropout_risk.resolve_model_artifact_path",
                return_value=Path("/cache/model.json"),
            ):
                with patch("app.api.dropout_risk.load_xgboost_model", return_value=model):
                    with patch(
                        "app.api.dropout_risk.create_tree_explainer",
                        return_value=explainer,
                    ):
                        response = TestClient(app).post(
                            "/analytics/dropout-risk/predict",
                            json={
                                "learnerId": "learner-1",
                                "courseId": "course-1",
                                "snapshotId": None,
                                "features": valid_features(),
                            },
                        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["riskScore"], 82.4)
        self.assertEqual(body["riskBand"], "high")
        self.assertEqual(body["topRiskFeature"], "lastActivityDays")
        self.assertEqual(body["modelVersion"], "dropout-xgb-v1")
        self.assertIn("shapValues", body)


if __name__ == "__main__":
    unittest.main()
