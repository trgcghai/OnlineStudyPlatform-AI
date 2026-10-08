import asyncio
from pathlib import Path

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from app.core.config import get_ai_settings
from app.db.mongo import get_mongo_database
from app.ml.dropout_constants import DROPOUT_KAGGLE_MODEL_VERSION
from app.schemas.dropout_risk import DropoutRiskTrainResponse
from app.services.dropout_model_registry import register_dropout_model
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

    return await register_dropout_model(
        database=database,
        model=model,
        model_dir=model_dir,
        model_version=DROPOUT_KAGGLE_MODEL_VERSION,
        sample_count=sample_count,
        metrics=metrics,
        extra_metadata={
            "trainingSource": "kaggle-1",
            "datasetPath": csv_path.as_posix(),
        },
    )


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
