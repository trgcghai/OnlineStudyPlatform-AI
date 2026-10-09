import asyncio
from pathlib import Path

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from app.core.config import get_ai_settings
from app.db.mongo import get_mongo_database
from app.ml.dropout_constants import DROPOUT_OULAD_MODEL_VERSION
from app.schemas.dropout_risk import ActiveDropoutModelResponse
from app.services.model_artifact_store import create_minio_client
from app.services.dropout_model_registry import register_dropout_model
from app.services.oulad_dropout_dataset import (
    DEFAULT_OULAD_SNAPSHOT_DAY,
    OULAD_TARGET_POLICY,
    load_oulad_dropout_samples,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OULAD_DATA_DIR = PROJECT_ROOT / "data" / "Open-University-Learning-Analytics"


async def train_oulad_dropout_model(
    database: object,
    data_dir: Path,
    artifact_client: object,
    artifact_bucket: str,
    temp_dir: Path,
    snapshot_day: int = DEFAULT_OULAD_SNAPSHOT_DAY,
) -> ActiveDropoutModelResponse:
    matrix, targets, dataset_metadata = load_oulad_dropout_samples(
        data_dir,
        snapshot_day=snapshot_day,
    )
    sample_count = int(targets.shape[0])

    if sample_count >= 10:
        x_train, x_test, y_train, y_test = train_test_split(
            matrix,
            targets,
            test_size=0.2,
            random_state=42,
            stratify=targets,
        )
    else:
        x_train, x_test, y_train, y_test = matrix, matrix, targets, targets

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse": round(float(mean_squared_error(y_test, predictions) ** 0.5), 4),
    }

    return await register_dropout_model(
        database=database,
        model=model,
        artifact_client=artifact_client,
        artifact_bucket=artifact_bucket,
        temp_dir=temp_dir,
        model_version=DROPOUT_OULAD_MODEL_VERSION,
        sample_count=sample_count,
        metrics=metrics,
        extra_metadata={
            "trainingSource": "oulad",
            "datasetPath": data_dir.as_posix(),
            "snapshotDay": snapshot_day,
            "targetPolicy": OULAD_TARGET_POLICY,
            "skippedRows": dataset_metadata["skippedRows"],
        },
    )


async def main() -> None:
    settings = get_ai_settings()
    artifact_client = create_minio_client(
        settings.minio_endpoint,
        settings.minio_access_key,
        settings.minio_secret_key,
        settings.minio_secure,
    )
    response = await train_oulad_dropout_model(
        database=get_mongo_database(),
        data_dir=DEFAULT_OULAD_DATA_DIR,
        artifact_client=artifact_client,
        artifact_bucket=settings.minio_bucket_ai_models,
        temp_dir=Path(settings.dropout_model_cache_dir) / "training-artifacts",
    )
    print(
        "Trained OULAD dropout model "
        f"version={response.model_version} "
        f"samples={response.sample_count} "
        f"mae={response.metrics['mae']} "
        f"rmse={response.metrics['rmse']} "
        f"artifact={response.artifact_path}"
    )


if __name__ == "__main__":
    asyncio.run(main())
