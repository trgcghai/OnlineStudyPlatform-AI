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
    DropoutRiskTrainResponse,
)
from app.services.dropout_explainability import explain_prediction, top_feature_from_shap
from app.services.dropout_model import (
    build_feature_matrix,
    create_tree_explainer,
    get_active_model_metadata,
    load_xgboost_model,
)
from app.services.dropout_training import train_dropout_model


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
) -> DropoutRiskPredictResponse:
    metadata = await get_active_model_metadata(database)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active dropout risk model found",
        )

    try:
        model = load_xgboost_model(metadata["artifactPath"])
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


@router.post(
    "/train",
    response_model=DropoutRiskTrainResponse,
    dependencies=[Depends(verify_token)],
)
async def train_dropout_risk(
    database: AsyncIOMotorDatabase = Depends(get_dropout_database),
    settings: AiSettings = Depends(get_ai_settings),
) -> DropoutRiskTrainResponse:
    try:
        return await train_dropout_model(
            database,
            Path(settings.dropout_model_dir),
            settings.dropout_min_training_samples,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


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
