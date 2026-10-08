from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.ml.dropout_constants import (
    FEATURE_VERSION,
    MODEL_ARTIFACT_NAME,
    MODEL_NAME,
)
from app.schemas.dropout_risk import ActiveDropoutModelResponse


async def register_dropout_model(
    *,
    database: object,
    model: object,
    model_dir: Path,
    model_version: str,
    sample_count: int,
    metrics: dict[str, float],
    trained_at: datetime | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> ActiveDropoutModelResponse:
    artifact_dir = model_dir / model_version
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / MODEL_ARTIFACT_NAME
    model.save_model(artifact_path)

    metadata = {
        "_id": model_version,
        "modelName": MODEL_NAME,
        "modelVersion": model_version,
        "featureVersion": FEATURE_VERSION,
        "artifactPath": artifact_path.as_posix(),
        "trainedAt": trained_at or datetime.now(UTC),
        "sampleCount": sample_count,
        "metrics": metrics,
        "isActive": True,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    await database.ai_model_versions.update_many(
        {"modelName": MODEL_NAME},
        {"$set": {"isActive": False}},
    )
    await database.ai_model_versions.replace_one(
        {"_id": model_version},
        metadata,
        upsert=True,
    )

    return ActiveDropoutModelResponse(**metadata)
