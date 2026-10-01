from pathlib import Path

import numpy as np
import shap
import xgboost as xgb
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.ml.dropout_constants import MODEL_NAME
from app.schemas.dropout_risk import DropoutRiskFeatures


def build_feature_matrix(features: DropoutRiskFeatures) -> np.ndarray:
    return np.asarray([features.to_ordered_list()], dtype=float)


def load_xgboost_model(artifact_path: str) -> xgb.XGBRegressor:
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

    model = xgb.XGBRegressor()
    model.load_model(path)
    return model


def create_tree_explainer(model: xgb.XGBRegressor) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


async def get_active_model_metadata(database: AsyncIOMotorDatabase) -> dict | None:
    return await database.ai_model_versions.find_one(
        {"modelName": MODEL_NAME, "isActive": True},
    )
