import asyncio
from datetime import UTC, datetime
from pathlib import Path

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from app.core.config import get_ai_settings
from app.db.mongo import get_mongo_database
from app.ml.dropout_constants import (
    FEATURE_VERSION,
    MODEL_ARTIFACT_NAME,
    MODEL_NAME,
    MODEL_VERSION,
)
from app.schemas.dropout_risk import DropoutRiskTrainResponse
from app.services.kaggle_dropout_dataset import load_kaggle_dropout_samples


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KAGGLE_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "kaggle-1"
    / "online_learning_engagement_dropout_risk.csv"
)


async def train_kaggle_dropout_model(
    database: object,
    csv_path: Path,
    model_dir: Path,
) -> DropoutRiskTrainResponse:
    matrix, targets = load_kaggle_dropout_samples(csv_path)
    sample_count = int(targets.shape[0])

    x_train, x_test, y_train, y_test = train_test_split(
        matrix,
        targets,
        test_size=0.2,
        random_state=42,
        stratify=targets,
    )

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

    artifact_dir = model_dir / MODEL_VERSION
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / MODEL_ARTIFACT_NAME
    model.save_model(artifact_path)
    trained_at = datetime.now(UTC)
    metadata = {
        "_id": MODEL_VERSION,
        "modelName": MODEL_NAME,
        "modelVersion": MODEL_VERSION,
        "featureVersion": FEATURE_VERSION,
        "artifactPath": artifact_path.as_posix(),
        "trainedAt": trained_at,
        "sampleCount": sample_count,
        "metrics": metrics,
        "isActive": True,
        "trainingSource": "kaggle-1",
        "datasetPath": csv_path.as_posix(),
    }

    await database.ai_model_versions.update_many(
        {"modelName": MODEL_NAME},
        {"$set": {"isActive": False}},
    )
    await database.ai_model_versions.replace_one(
        {"_id": MODEL_VERSION},
        metadata,
        upsert=True,
    )

    return DropoutRiskTrainResponse(**metadata)


async def main() -> None:
    settings = get_ai_settings()
    response = await train_kaggle_dropout_model(
        database=get_mongo_database(),
        csv_path=DEFAULT_KAGGLE_CSV_PATH,
        model_dir=Path(settings.dropout_model_dir),
    )
    print(
        "Trained Kaggle dropout model "
        f"version={response.model_version} "
        f"samples={response.sample_count} "
        f"mae={response.metrics['mae']} "
        f"rmse={response.metrics['rmse']} "
        f"artifact={response.artifact_path}"
    )


if __name__ == "__main__":
    asyncio.run(main())
