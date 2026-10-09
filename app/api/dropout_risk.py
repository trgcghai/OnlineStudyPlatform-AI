from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.config import AiSettings, get_ai_settings
from app.db.mongo import get_mongo_database
from app.ml.dropout_constants import (
    clamp_score,
    recommended_action_for_feature,
    risk_band_for_score,
)
from app.schemas.dropout_risk import (
    ActiveDropoutModelResponse,
    DropoutRiskPredictRequest,
    DropoutRiskPredictResponse,
)
from app.services.dropout_explainability import explain_prediction, top_feature_from_shap
from app.services.dropout_model import (
    build_feature_matrix,
    create_tree_explainer,
    get_active_model_metadata,
    load_xgboost_model,
    resolve_model_artifact_path,
)
from app.services.model_artifact_store import create_minio_client


router = APIRouter(prefix="/analytics/dropout-risk", tags=["dropout-risk"])


def get_dropout_database() -> AsyncIOMotorDatabase:
    return get_mongo_database()


def verify_token(
    authorization: str | None = Header(default=None),
    settings: AiSettings = Depends(get_ai_settings),
) -> None:
    if not settings.ai_token:
        return

    expected = f"Bearer {settings.ai_token}"
    if authorization != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid AI service token",
        )


@router.post(
    "/predict",
    response_model=DropoutRiskPredictResponse,
    dependencies=[Depends(verify_token)],
)
async def predict_dropout_risk(
    request: DropoutRiskPredictRequest,
    database: AsyncIOMotorDatabase = Depends(get_dropout_database),
    settings: AiSettings = Depends(get_ai_settings),
) -> DropoutRiskPredictResponse:
    metadata = await get_active_model_metadata(database)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active dropout risk model found",
        )

    try:
        artifact_client = create_minio_client(
            settings.minio_endpoint,
            settings.minio_access_key,
            settings.minio_secret_key,
            settings.minio_secure,
        )
        artifact_path = resolve_model_artifact_path(
            metadata,
            artifact_client=artifact_client,
            cache_root=Path(settings.dropout_model_cache_dir),
        )
        model = load_xgboost_model(artifact_path.as_posix())
        matrix = build_feature_matrix(request.features)
        raw_score = float(model.predict(matrix)[0])
        risk_score = clamp_score(raw_score)
        explainer = create_tree_explainer(model)
        shap_values = explain_prediction(explainer, matrix)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dropout risk prediction failed: {exc}",
        ) from exc

    top_feature = top_feature_from_shap(shap_values)

    return DropoutRiskPredictResponse(
        learnerId=request.learner_id,
        courseId=request.course_id,
        snapshotId=request.snapshot_id,
        riskScore=risk_score,
        riskBand=risk_band_for_score(risk_score),
        topRiskFeature=top_feature,
        recommendedAction=recommended_action_for_feature(top_feature),
        shapValues=shap_values,
        modelVersion=metadata["modelVersion"],
        predictedAt=datetime.now(UTC),
    )


@router.get(
    "/model/active",
    response_model=ActiveDropoutModelResponse,
    dependencies=[Depends(verify_token)],
)
async def get_active_dropout_model(
    database: AsyncIOMotorDatabase = Depends(get_dropout_database),
) -> ActiveDropoutModelResponse:
    metadata = await get_active_model_metadata(database)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active dropout risk model found",
        )

    return ActiveDropoutModelResponse(**metadata)
