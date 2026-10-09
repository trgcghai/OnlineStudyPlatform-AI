from pathlib import Path

import numpy as np
import shap
import xgboost as xgb
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.ml.dropout_constants import MODEL_NAME
from app.schemas.dropout_risk import DropoutRiskFeatures
from app.services.model_artifact_store import (
    ModelArtifactLocation,
    download_model_artifact,
)


def build_feature_matrix(features: DropoutRiskFeatures) -> np.ndarray:
    return np.asarray([features.to_ordered_list()], dtype=float)


def load_xgboost_model(artifact_path: str) -> xgb.XGBRegressor:
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

    model = xgb.XGBRegressor()
    model.load_model(path)
    return model


def resolve_model_artifact_path(
    metadata: dict,
    *,
    artifact_client: object,
    cache_root: Path,
) -> Path:
    if metadata.get("artifactStorage") != "minio":
        raise ValueError(
            "Unsupported model artifact storage; retrain the model to create MinIO metadata"
        )

    model_version = metadata.get("modelVersion")
    bucket = metadata.get("artifactBucket")
    object_key = metadata.get("artifactObjectKey")
    artifact_path = metadata.get("artifactPath")
    if not all(
        isinstance(value, str) and value
        for value in [model_version, bucket, object_key, artifact_path]
    ):
        raise ValueError("Incomplete MinIO model artifact metadata")

    return download_model_artifact(
        client=artifact_client,
        location=ModelArtifactLocation(
            storage="minio",
            bucket=bucket,
            object_key=object_key,
            uri=artifact_path,
            etag=metadata.get("artifactEtag"),
        ),
        cache_root=cache_root,
        model_version=model_version,
    )


def create_tree_explainer(model: xgb.XGBRegressor) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


async def get_active_model_metadata(database: AsyncIOMotorDatabase) -> dict | None:
    return await database.ai_model_versions.find_one(
        {"modelName": MODEL_NAME, "isActive": True},
    )
