from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xgboost as xgb
from motor.motor_asyncio import AsyncIOMotorDatabase
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from app.ml.dropout_constants import (
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    MODEL_ARTIFACT_NAME,
    MODEL_NAME,
    MODEL_VERSION,
)
from app.schemas.dropout_risk import DropoutRiskFeatures, DropoutRiskTrainResponse


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


async def load_training_samples(
    database: AsyncIOMotorDatabase,
) -> tuple[np.ndarray, np.ndarray]:
    cursor = database.risk_scores.find(
        {
            "source": "heuristic",
            "featureVersion": FEATURE_VERSION,
            "features": {"$exists": True},
            "score": {"$type": "number"},
        }
    )
    rows = await cursor.to_list(length=None)
    matrices: list[list[float]] = []
    targets: list[float] = []

    for row in rows:
        features = row.get("features")
        score = row.get("score")
        if not isinstance(features, dict) or not _is_number(score):
            continue
        try:
            parsed = DropoutRiskFeatures(**features)
        except Exception:
            continue
        matrices.append(parsed.to_ordered_list())
        targets.append(float(score))

    if not matrices:
        return np.empty((0, len(FEATURE_COLUMNS))), np.empty((0,))

    return np.asarray(matrices, dtype=float), np.asarray(targets, dtype=float)


async def train_dropout_model(
    database: AsyncIOMotorDatabase,
    model_dir: Path,
    min_samples: int,
) -> DropoutRiskTrainResponse:
    matrix, targets = await load_training_samples(database)
    sample_count = int(targets.shape[0])
    if sample_count < min_samples:
        raise ValueError(
            f"At least {min_samples} valid samples are required; found {sample_count}"
        )

    if sample_count >= 20:
        x_train, x_test, y_train, y_test = train_test_split(
            matrix,
            targets,
            test_size=0.2,
            random_state=42,
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
        "artifactPath": str(artifact_path).replace("\\", "/"),
        "trainedAt": trained_at,
        "sampleCount": sample_count,
        "metrics": metrics,
        "isActive": True,
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
